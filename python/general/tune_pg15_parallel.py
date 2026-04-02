import torch
import pandas as pd
import numpy as np
from skopt import gp_minimize
from sklearn.model_selection import StratifiedKFold
from scipy.stats import poisson
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch_optimizer as optimz
from pathlib import Path
import pickle
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KernelDensity
from skopt import BayesSearchCV
import sys, pathlib

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT/"python/general").exists():
    ROOT = ROOT.parent

if not (ROOT / "python/general").exists():
    raise FileNotFoundError(f"Could not find 'general' folder starting from {Path.cwd()}")

sys.path.insert(0, str(ROOT/"python/general"))

from NN import PoissonRegressor
from dcovs import u_centered_dist, sq_dcov_unbiased, JdCov_sq_unbiased
from metrics import rps_poisson
from utils import hp_list_to_dict

def _reg_none(*_):
    """Return zero: no regularisation (baseline cross-entropy only)."""
    return 0.0
    

def _reg_none(*_):  return 0.0

def _reg_ccd(y, z1, z2):
    return sq_dcov_unbiased(y, torch.cat((z1, z2.unsqueeze(1)), 1))

def _reg_jd(y, z1, z2):
    return JdCov_sq_unbiased(y, z1, z2)

_REG = {"none": _reg_none, "ccdcov": _reg_ccd, "jdcov": _reg_jd}


class FairPoissonReg(BaseEstimator, RegressorMixin):
    """
    Feed-forward Poisson regressor + optional dCov fairness penalty.
    Last column of X must be log-offset.
    """

    # ────────────────────────────────────────────────────────────────────
    def __init__(self, *, input_dim,
                 num_layers=3, hidden_dim=64, activation="ReLU",
                 dropout=0.2, batch_size=256,
                 lr=1e-3, betas="(0.9,0.999)", hessian_power=0.75,
                 reg_type="none", lambda_reg=0.0,
                 max_epochs=5000, patience=15,
                 random_state=42, device="cpu",
                 checkpoint_dir=None):

        for k, v in locals().items():
            if k not in {"self", "__class__"}:
                setattr(self, k, v)
        self.n_epochs_ = 0                           # total epochs of best fold

    # ────────────────────────────────────────────────────────────────────
    def _split_X(self, X):
        if X.ndim != 2 or X.shape[1] != self.input_dim + 1:
            raise ValueError(f"Expect shape (n,{self.input_dim+1}) – last col offset")
        return X[:, :-1], X[:, -1]

    # ────────────────────────────────────────────────────────────────────
    def _build_model(self, dev):
        return PoissonRegressor(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            activation=self.activation if isinstance(self.activation, str)
                      else self.activation.__name__,
            dropout=self.dropout,
        ).to(dev)

    # ────────────────────────────────────────────────────────────────────
    def _fit_one_fold(self, model, optim, dl_tr, X_val, y_val, off_val,
                      z1_val, z2_val, reg_fn):

        best_rps, best_state = np.inf, None
        no_imp, n_epochs = 0, 0

        for epoch in range(self.max_epochs):

            # ---- training loop
            model.train()
            for xb, yb, offb, zz1, zz2 in dl_tr:
                optim.zero_grad()
                pred = model(xb, offb).squeeze()
                loss = (nn.PoissonNLLLoss(log_input=False)(pred, yb) +
                        self.lambda_reg * reg_fn(pred, zz1, zz2))
                loss.backward(create_graph=True)
                optim.step()

            # ---- validation
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val, off_val).squeeze()
                rps = rps_poisson(pred_val, y_val).item()

            n_epochs += 1
            if rps + 1e-3 < best_rps:
                best_rps = rps
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break

        return best_state, best_rps, n_epochs

    # ────────────────────────────────────────────────────────────────────
    def fit(self, X, y, *, z1, z2, strat_key):

        import torch, os, numpy as np
        from torch.utils.data import TensorDataset, DataLoader
        from pathlib import Path
        from sklearn.model_selection import StratifiedKFold

        # ---- reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.set_num_threads(1); os.environ.setdefault("OMP_NUM_THREADS", "1")

        # ---- tensorise
        X_cov, off = self._split_X(np.asarray(X))
        dev = torch.device(self.device)
        X_cov = torch.tensor(X_cov, dtype=torch.float32, device=dev)
        y     = torch.tensor(y,     dtype=torch.float32, device=dev)
        z1    = torch.tensor(z1,    dtype=torch.float32, device=dev)
        z2    = torch.tensor(z2,    dtype=torch.float32, device=dev)
        off   = torch.tensor(off,   dtype=torch.float32, device=dev)

        reg_fn = {"none": _reg_none, "ccdcov": _reg_ccd, "jdcov": _reg_jd}[self.reg_type]

        # ---- cross-validation
        kf = StratifiedKFold(5, shuffle=True, random_state=self.random_state)
        best_rps_global = np.inf
        best_state_global = None
        best_epochs_global = 0

        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.empty(len(X_cov)), strat_key)):

            model = self._build_model(dev)
            optim = optimz.Adahessian(
                model.parameters(), lr=self.lr,
                betas=eval(self.betas) if isinstance(self.betas, str) else self.betas,
                hessian_power=self.hessian_power)

            ds_tr = TensorDataset(X_cov[tr_idx], y[tr_idx], off[tr_idx],
                                  z1[tr_idx], z2[tr_idx])
            dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)

            best_state_fold, rps_fold, epochs_fold = self._fit_one_fold(
                model, optim, dl_tr,
                X_cov[va_idx], y[va_idx], off[va_idx],
                z1[va_idx], z2[va_idx], reg_fn)

            if rps_fold < best_rps_global:
                best_rps_global = rps_fold
                best_state_global = best_state_fold
                best_epochs_global = epochs_fold

        # ---- restore global best
        self.model_ = self._build_model(dev)
        self.model_.load_state_dict(best_state_global)
        self.n_epochs_ = best_epochs_global

        # ---- save ONE checkpoint if path supplied
        if self.checkpoint_dir:
            ckpt_path = Path(self.checkpoint_dir) / "best_checkpoint.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "best_rps": best_rps_global,
                "total_epochs": best_epochs_global,
                "model_state_dict": best_state_global,
                "hyperparameters": self.get_params(deep=False)},
                ckpt_path)

        return self

    # ────────────────────────────────────────────────────────────────────
    def predict(self, X):
        X_cov, off = self._split_X(np.asarray(X))
        dev = next(self.model_.parameters()).device
        X_cov = torch.tensor(X_cov, dtype=torch.float32, device=dev)
        off   = torch.tensor(off,   dtype=torch.float32, device=dev)
        with torch.no_grad():
            lam = self.model_(X_cov, off).cpu().numpy()
        return lam

    def score(self, X, y):
        return -rps_poisson(torch.tensor(self.predict(X)), torch.tensor(y)).item()

def neg_rps_poisson(estimator, X, y_true, *, sample_weight):
    """
    This functino returns a negative RPS score for the skopt estimator object.
    """
    return estimator.score(X, y_true, offset=sample_weight)
  
  
def fit_poisson_pg15(X_tr, y_tr, Z1_tr, Z2_tr, off_tr,
                     X_te, y_te, off_te, hp, lambda_reg, reg_type):
    """
    Creating a skopt object for running BayesSearchCV with this object.
    """
    _REG = {"none": _reg_none, "ccdcov": _reg_ccd, "jdcov": _reg_jd}
    reg_fn = _REG[reg_type]
    train_dt = TensorDataset(X_tr , y_tr, off_tr , Z1_tr , Z2_tr)
    train_ld = DataLoader(train_dt, batch_size = hp['iterator_train__batch_size'],
                          shuffle=True)
    input_dim = X_tr.shape[1]
    model = PoissonRegressor(input_dim=input_dim,
                             num_layers=hp['module__num_layers'],
                             hidden_dim=hp['module__hidden_dim'],
                             activation=hp['module__activation'],
                             dropout=hp['module__dropout'])
    optim = optimz.Adahessian(model.parameters(), lr=hp['lr'],
                              betas=hp['optimizer__betas'],
                              hessian_power=hp['optimizer__hessian_power'])
    for _ in range(hp["max_epochs"]):
        model.train()
        for Xb, yb, offb, Z1b, Z2b in train_ld:
            optim.zero_grad()
            preds = model(Xb,offb)
            loss = nn.PoissonNLLLoss(log_input=False)(preds.squeeze(), yb) + lambda_reg*reg_fn(preds.squeeze(),Z1b, Z2b)
            loss.backward(create_graph=True)
            optim.step()
    model.eval()
    with torch.no_grad():
        train_out = model(X_tr, off_tr)
        test_out  = model(X_te, off_te)
    results = {
        'lambda': lambda_reg,
        'train_output': train_out,
        'train_rps': rps_poisson(train_out, y_tr),
        'train_pdev': nn.PoissonNLLLoss(log_input=False)(train_out, y_tr),
        'val_output': test_out,
        'val_rps': rps_poisson(test_out, y_te),
        'val_pdev': nn.PoissonNLLLoss(log_input=False)(test_out, y_te)
    }
    return results

def make_interaction_dfs(
    outputs: torch.Tensor,
    region:  torch.Tensor,                  # shape (N, 10)
    gender:    torch.Tensor,                  # one-hot (N,)
    region_labels   = ['Region_L', 'Region_M', 'Region_N', 'Region_O',
                       'Region_P', 'Region_Q', 'Region_R', 'Region_S', 
                       'Region_T', 'Region_U']
):
    """Return data frames with all subgroup combinations used in analysis.

    The function converts tensor inputs to Pandas series, maps categorical
    codes to human-readable labels, then builds six different frames whose
    second column is the grouping key required by fairness metrics.

    Output dict keys
    ----------------
    gender            : 2-level        – 'Male' / 'Female'  
    region            : 10-level        – as in *region*  
    gender_region     : 20-level      

    All frames contain two columns: ['Output', <group_key>].

    Args:
        outputs : torch.Tensor (N,)
            Model scores or probabilities.
        gender  : torch.Tensor (N,)
            Binary indicator 0 = Male, 1 = Female.
        race    : torch.Tensor (N, 4)
            One-hot encoded race.
        age     : torch.Tensor (N,)
            Continuous age in years.
        age_bins, age_labels, race_labels
            Custom bin edges / labels.  Length of *race_labels* must match
            race.shape[1].

    Returns:
        dict(str, pd.DataFrame)
            Mapping from subgroup name to frame.
    """
    # decode tensors
    out_np   = outputs.numpy()
    gender_s = pd.Series(gender.numpy()).map({0: 'Male', 1: 'Female'})
    region_s   = pd.Series(torch.argmax(region, dim=1).numpy()).map(dict(enumerate(region_labels)))

    # build interaction columns
    df_base = pd.DataFrame({'Output': out_np})
    return {
        "gender":             df_base.assign(Gender = gender_s),
        "region":             df_base.assign(region   = region_s),
        "gender_region":      df_base.assign(Gender_region = gender_s + ', ' + region_s)
    }


def build_df(outputs_tensor, gender_tensor, region_tensor):
    """
    For CDF: cols = ['Predicted claim frequency', 'Gender_region']
    """
    region_labels = ['Region_L', 'Region_M', 'Region_N', 'Region_O', 'Region_P',
                     'Region_Q', 'Region_R', 'Region_S', 'Region_T', 'Region_U']
                 
    df = pd.DataFrame({
        'Predicted claim frequency': outputs_tensor.numpy().squeeze(),
        'Gender': gender_tensor.numpy().squeeze()
    })
    df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})
    region_class = torch.argmax(region_tensor, dim=1)
    df_region = pd.DataFrame({'Region': region_class.numpy()})
    df_region['Region'] = df_region['Region'].apply(lambda x: region_labels[x])
    df['Gender_region'] = df['Gender'] + ', ' + df_region['Region']
    return df

def plot_kde_by_region_gender(ax, df, interaction_col, output_col, bandwidth=0.05):
    """Returns dict(region→handle), reference_handle"""
    region_labels = ['Region_L', 'Region_M', 'Region_N', 'Region_O', 'Region_P',
                     'Region_Q', 'Region_R', 'Region_S', 'Region_T', 'Region_U']
    tol_colors = ['#332288', '#117733', '#88CCEE', '#DDCC77', '#CC6677',
                  '#AA4499', '#44AA99', '#999933', '#882255', '#661100']
    region_colors = dict(zip(region_labels, tol_colors))
    unconditional_values = df[output_col].dropna().values
    if len(unconditional_values) < 2:
        return {}, None
    x_range = np.linspace(unconditional_values.min(), unconditional_values.max(), 200)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(unconditional_values.reshape(-1,1))
    density_ref = np.exp(kde.score_samples(x_range.reshape(-1,1)))
    ref_line, = ax.plot(x_range, density_ref, '--', color='black', linewidth=2.0, label='Reference')
    region_handles = {}
    for cat in df[interaction_col].unique():
        sub = df.loc[df[interaction_col]==cat, output_col].dropna().values
        if len(sub) < 2:  # skip tiny subgroups
            continue
        gender, region = cat.split(', ')
        color = region_colors.get(region, 'gray')
        linestyle = '-' if gender=='Male' else ':'
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sub.reshape(-1,1))
        density = np.exp(kde.score_samples(x_range.reshape(-1,1)))
        line, = ax.plot(x_range, density, color=color, linestyle=linestyle, linewidth=2.0,
                        label=region if region not in region_handles else '_nolegend_')
        region_handles.setdefault(region, line)
    ax.set_xlabel("Predicted claim frequency")
    ax.grid(True)
    return region_handles, ref_line

def compute_empirical_cdf(data):
    """
    This function computes the empirical CDF, taking tidied data from build_df.
    """
    sorted_data = np.sort(data)
    cdf_values = np.linspace(0, 1, len(sorted_data), endpoint=False)
    return sorted_data, cdf_values

def plot_empirical_cdf(ax, df, interaction_col, output_col):
    """
    This function plot the computed CDF using the above function.
    Also assigns plot to different subplots through ax.
    Labels and colours are assigned through interaction_col.
    """
    region_labels = ['Region_L', 'Region_M', 'Region_N', 'Region_O', 'Region_P',
                     'Region_Q', 'Region_R', 'Region_S', 'Region_T', 'Region_U']
    tol_colors = ['#332288', '#117733', '#88CCEE', '#DDCC77', '#CC6677',
                  '#AA4499', '#44AA99', '#999933', '#882255', '#661100']
    region_colors = dict(zip(region_labels, tol_colors))
    x_all, cdf_all = compute_empirical_cdf(df[output_col].dropna().values)
    ref_line, = ax.step(x_all, cdf_all, '--', color='black', linewidth=2.0, where='post')
    handles = {'Reference distribution': ref_line}
    for cat in df[interaction_col].unique():
        subset = df.loc[df[interaction_col]==cat, output_col].dropna().values
        if len(subset) < 2:
            continue
        gender, region = cat.split(', ')
        color = region_colors.get(region, 'gray')
        linestyle = '-' if gender=='Male' else ':'
        x_cat, cdf_cat = compute_empirical_cdf(subset)
        line, = ax.step(x_cat, cdf_cat, color=color, linestyle=linestyle, linewidth=2.0, where='post')
        handles.setdefault(region, line)
    ax.set_xlabel("Predicted claim frequency")
    ax.grid(True)
    return handles
