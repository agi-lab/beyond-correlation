import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson
from sklearn.neighbors import KernelDensity

import dcovs

def rps_binary(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Finding the ranked probability score for a binary classification model.
    
    Args:
        predictions: Tensor of shape (N,), predicted probabilities for the positive class.
        labels: Tensor of shape (N,), true binary labels (0 or 1).
    
    Returns:
        torch.Tensor: Mean RPS score of the model.
    """
    # Ensure predictions are 2D for consistency
    predictions = predictions.unsqueeze(1)

    # Convert labels to integers before one-hot encoding
    labels = labels.long()

    # One-hot encode the labels for two classes
    labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
    
    # Compute cumulative sums
    cumulative_predictions = torch.cat([1 - predictions, predictions], dim=1).cumsum(dim=1)
    cumulative_labels = labels.cumsum(dim=1)
    
    # Compute RPS
    squared_diff = (cumulative_predictions - cumulative_labels) ** 2
    rps = torch.mean(torch.sum(squared_diff, dim=1))
    
    return rps
  
  
def classification_accuracy(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Computes classification accuracy given predicted probabilities and ground truth labels.

    Args:
        probs: Tensor of predicted probabilities (shape: [n_samples]).
        targets: Tensor of ground-truth binary labels (0 or 1).
        threshold: Threshold to convert probabilities into binary predictions. Default is 0.5.

    Returns:
        float: Classification accuracy as a float between 0 and 1.
    """
    preds = (probs > threshold).int()
    accuracy = (preds == targets).float().mean().item()
    return accuracy


def normalize_pdf(pdf):
    """Return a PDF rescaled so that its integral (sum) equals 1.

    Args:
        pdf: 1-D NumPy array of non-negative density values.

    Returns:
        A copy of pdf divided by pdf.sum().
    """
    return pdf / np.sum(pdf)

def entropy(p, n_classes=2):
    """Shannon entropy H(p) using ``log(n_classes)`` as base.

    Args:
        p: 1-D probability mass or density array. Values are clipped to
           1*e-10 to avoid log-zero.
        n_classes: Base of the logarithm – defaults to 2 (bits).

    Returns:
        Scalar entropy value.
    """
    p = np.clip(p, 1e-10, None)  # Avoid log(0) by clipping probabilities
    return -np.sum(p * np.log(p) / np.log(2))

def JSD_generalized(output_class_df, weights="proportion"):
    """Jensen–Shannon divergence between classes for continuous outputs.

    Computes the generalised JSD (Lin, 1991) where the reference density
    is the weighted average of class-specific kernel-density estimates
    (KDEs).

    Args:
        output_class_df (pd.DataFrame): Two-column frame. First column = model outputs (floats),
                                        second column = class labels (categorical or int).
        weights ({'proportion', 'equal'}, default 'proportion'): 
            'proportion' — weight each class by its empirical frequency  
            'equal' — weight each class equally (1 / k)  
        bandwidth (float, default 0.15): Gaussian KDE bandwidth used for every class.
        grid_size (int, default 1000): Number of evenly spaced x-points on which each KDE is evaluated.

    Returns:
        float: Generalised JSD (non-negative, 0 = identical distributions).

    """
    output_list = output_class_df.iloc[:, 0]
    class_list = output_class_df.iloc[:, 1]
    classes = class_list.unique()
    n_classes = len(classes)

    pi = []
    outputs = []
    kde = []
    pdf = []

    # Use 1000 evenly spaced points for PDF evaluation
    x_values = np.linspace(min(output_list), max(output_list), 1000)

    # Collect class-specific data and compute proportion-based pi
    for cls in classes:
        # Proportion of each class
        pi.append(class_list.value_counts(normalize=True).get(cls, 0))
        outputs.append(output_list[class_list == cls].to_numpy())

    # We'll store the line handles (and labels) so we can reorder them
    line_handles = []

    for idx, output in enumerate(outputs):
        # Fit the KDE model for each class
        class_kde = KernelDensity(kernel="gaussian", bandwidth=0.15).fit(output[:, None])
        kde.append(class_kde)

        # Compute log likelihood for each x_value
        class_log_pdf = class_kde.score_samples(x_values[:, None])

        # Compute the PDF (no normalization for plotting)
        class_pdf_no_norm = np.exp(class_log_pdf)

        # Normalize the PDF for JSD calculation
        class_pdf = normalize_pdf(class_pdf_no_norm)
        pdf.append(class_pdf)

    # Apply the selected weighting scheme
    if weights == "proportion":
        # pi is already proportion-based
        pdf_ref = np.sum(np.array(pi)[:, None] * pdf, axis=0)
    elif weights == "equal":
        pi = np.full(len(classes), 1 / len(classes))
        pdf_ref = np.sum(pi[:, None] * pdf, axis=0)
    else:
        raise ValueError("weights must be 'proportion' or 'equal'")

    # Step 1: Compute entropy of the reference PDF
    H_ref = entropy(pdf_ref)

    # Step 2: Compute entropy for each class-specific PDF
    H_classes = [entropy(pdfs) for pdfs in pdf]

    # Step 3: Weighted sum of class-specific entropies
    weighted_H = np.sum(np.array(pi) * np.array(H_classes))

    # Step 4: Generalized JSD
    general_JSD = H_ref - weighted_H
    
    return general_JSD

def calculate_uf(df):
    """
    Calculate the unfairness metric UF(π) from a pandas DataFrame.
    
    Args:
        df: pandas DataFrame, where the first column is the output (π(X)), 
            and the second column is the group (D) (which can be text).
    
    Returns:
        UF(π): The unfairness metric as defined by the variance of the group-wise means 
               divided by the total variance of the output.
    """
    # Ensure the output (π(X)) column is numeric and coerce non-numeric values to NaN
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # First column is output

    # Drop rows where output is NaN after conversion (i.e., non-numeric values in output)
    df_clean = df.dropna(subset=[df.columns[0]])

    # Now extract the cleaned output and group
    output_clean = df_clean.iloc[:, 0]  # Cleaned output
    group_clean = df_clean.iloc[:, 1]   # Cleaned group

    # 1. Calculate the overall variance of the output (Var(π(X)))
    var_pi_x = np.var(output_clean, ddof=1)  # Using ddof=1 for sample variance

    # 2. Calculate the group-wise means (E[π(X) | D])
    group_means = df_clean.groupby(group_clean)[df_clean.columns[0]].mean()  # Mean of π(X) for each group

    # 3. Calculate the variance of the group-wise means (Var(E[π(X) | D]))
    var_group_means = np.var(group_means, ddof=1)

    # 4. Calculate UF(π)
    uf_pi = var_group_means / var_pi_x

    return uf_pi

def dcorr_test(outputs, protected):
    """Distance-correlation χ² test for scalar × continuous independence.
  
    Implements the asymptotic test proposed by Shen et al. (2022).
    The statistic *n⋅dCor² + 1* is χ²₁ under H₀.
  
    Args:
        outputs   : 1-D tensor of model scores (shape ``(n,)``).
        protected : 2-D tensor of protected feature(s) (shape ``(n, d)``).
  
    Returns:
        stat  : unbiased distance covariance (float).
        pval  : χ² p-value (float).
    """
    n = outputs.shape[0]
    stat = dcorr_unbiased(outputs, protected)
    pval = chi2.sf(stat * n + 1, 1)
    return stat, pval

def dcorr_test(outputs, protected):
    """Distance-correlation χ² test for scalar × continuous independence.
  
    Implements the asymptotic test proposed by Shen et al. (2022).
    The statistic *n⋅dCor² + 1* is χ²₁ under H₀.
  
    Args:
        outputs   : 1-D tensor of model scores (shape ``(n,)``).
        protected : 2-D tensor of protected feature(s) (shape ``(n, d)``).
  
    Returns:
        stat  : unbiased distance covariance (float).
        pval  : χ² p-value (float).
    """
    n = outputs.shape[0]
    stat = dcorr_unbiased(outputs, protected)
    pval = chi2.sf(stat * n + 1, 1)
    return stat, pval

def permtest_indep_jdcov(*vars, n_bootstrap = 100):
    """Permutation test for joint distance covariance (JdCov).

    Args:
        *vars : Two or more tensors, can be any dimension, each shape ``(n,)``.  
                The first is the model output/response; the rest are protected vars.
        n_bootstrap: Number of permutations (default 100).

    Returns:
        observed : Observed JdCov².
        crit     : 95-th percentile of permutation distribution.
        pvalue   : Monte-Carlo p-value (float).
    """
    observed_jdcov = JdCov_sq_unbiased(*vars)
    n = vars[0].size(0)
    dist = torch.tensor([JdCov_sq_unbiased(*[var[torch.randperm(var.size(0))] for var in vars]) for _ in range(n_bootstrap)])
    crit = np.quantile(dist, 0.95)
    pvalue = (1 + torch.sum(dist >= observed_jdcov))/(n_bootstrap + 1)
    return observed_jdcov, crit, pvalue

def permtest_indep_jdcov_mem(*vars, n_bootstrap = 100):
    """Permutation test for joint distance covariance (JdCov).

    Args:
        *vars : Two or more tensors, can be any dimension, each shape ``(n,)``.  
                The first is the model output/response; the rest are protected vars.
        n_bootstrap: Number of permutations (default 100).

    Returns:
        observed : Observed JdCov².
        crit     : 95-th percentile of permutation distribution.
        pvalue   : Monte-Carlo p-value (float).
    """
    observed_jdcov = JdCov_sq_unbiased_mem(*vars)
    n = vars[0].size(0)
    dist = torch.tensor([JdCov_sq_unbiased_mem(*[var[torch.randperm(var.size(0))] for var in vars]) for _ in range(n_bootstrap)])
    crit = np.quantile(dist, 0.95)
    pvalue = (1 + torch.sum(dist >= observed_jdcov))/(n_bootstrap + 1)
    return observed_jdcov, crit, pvalue
  
# here replace the protected attributes with vectorised version of them
def permtest_indep_ccdcov(*vars, n_bootstrap = 100):
    """Permutation test for centred composite distance covariance (CC-dCov).

    Args:
        outputs        : tensors of output/response and joint vector of protected attributes, shape ``(n,)``.
        n_bootstrap    : Number of permutations.

    Returns:
        observed : Observed CC-dCov² statistic.
        crit     : 95-th percentile of permutation distribution.
        pvalue   : Monte-Carlo p-value.
    """
    observed_ccdcov = sq_dcov_unbiased(*vars)
    n = vars[0].size(0)
    dist = torch.tensor([sq_dcov_unbiased(*[var[torch.randperm(var.size(0))] for var in vars]) for _ in range(n_bootstrap)])
    crit = np.quantile(dist, 0.95)
    pvalue = (1 + torch.sum(dist >= observed_ccdcov))/(n_bootstrap + 1)
    return observed_ccdcov, crit, pvalue


def rps_poisson(predicted_rates, observed_counts, max_count=10):
    """Ranked-Probability Score for Poisson forecasts (vectorised).
    Args:
        predicted_rates : 1-D tensor, shape (N,)  
            Poisson rate (λ) predicted for each instance.
        observed_counts : 1-D tensor, shape (N,)  
            Observed non-negative integer counts.
        max_count       : Largest count threshold *K* used to build the CDF grid
            (default 10).  Should be ≥ max(observed_counts).

    Returns:
        Mean RPS of the model.
    """
    rps_values = []
    for lam, obs in zip(predicted_rates, observed_counts):
        lam_val = lam.item()
        obs_val = int(obs.item())
        # Compute predicted CDF for k in [0, max_count]
        pred_cdf = [poisson.cdf(k, lam_val) for k in range(max_count + 1)]
        pred_cdf = torch.tensor(pred_cdf)
        # Observed CDF: 0 until obs_val, then 1
        obs_cdf = torch.zeros(max_count + 1)
        obs_cdf[obs_val:] = 1
        rps_val = torch.sum((pred_cdf - obs_cdf) ** 2)
        rps_values.append(rps_val.item())
    return np.mean(rps_values)


# HELPER: dtype‑preserving pairwise L2 distance
def _pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ‖xᵢ−yⱼ‖₂ for all i,j, keeping the input dtype."""
    if x.ndim == 1: x = x[:, None]
    if y.ndim == 1: y = y[:, None]
    return torch.cdist(x, y, p=2)  # torch.cdist preserves dtype

# MEMORY‑FRIENDLY VERSIONS
def _u_centered_dist_mem(x: torch.Tensor) -> torch.Tensor:
    """Vectorised; builds *one* (n,n) matrix, default float32."""
    n   = x.shape[0]
    dis = _pairwise_l2(x, x)
    col = dis.sum(1, keepdim=True)
    row = col.T
    tot = dis.sum()
    U   = row/(n-2) + col/(n-2) - dis - tot/((n-1)*(n-2))
    U.fill_diagonal_(0)
    return U


def sq_dcov_unbiased_mem(x1: torch.Tensor,
                     x2: torch.Tensor,
                     block = None) -> torch.Tensor:
    """
    Unbiased ν²_tilde.  If block is None, keep full (n,n) matrices.
    If block is int, stream in O(block·n) memory.
    """
    n, dtype = x1.shape[0], x1.dtype

    # ---------- full‑matrix path ----------
    if block is None:
        return (_u_centered_dist_mem(x1) *
                _u_centered_dist_mem(x2)).sum() / (n*(n-3))

    # ---------- streaming path ------------
    rc = 1.0/(n-2)
    tc = 1.0/((n-1)*(n-2))

    row1 = torch.zeros(n, dtype=dtype)
    row2 = torch.zeros(n, dtype=dtype)
    tot1 = torch.zeros((), dtype=dtype)
    tot2 = torch.zeros((), dtype=dtype)

    # pass 1 – row & total sums
    for i in range(0, n, block):
        sli = slice(i, i+block)
        D1  = _pairwise_l2(x1[sli], x1)
        D2  = _pairwise_l2(x2[sli], x2)
        row1[sli] = D1.sum(1)
        row2[sli] = D2.sum(1)
        tot1     += D1.sum()
        tot2     += D2.sum()

    tot1 *= tc
    tot2 *= tc

    # pass 2 – accumulate Σ U₁·U₂
    acc = torch.zeros((), dtype=dtype)

    for i in range(0, n, block):
        sli = slice(i, i+block)
        for j in range(i, n, block):
            slj = slice(j, j+block)

            D1 = _pairwise_l2(x1[sli], x1[slj])
            D2 = _pairwise_l2(x2[sli], x2[slj])

            U1 =  row1[sli][:, None]*rc + row1[slj][None, :]*rc - D1 - tot1
            U2 =  row2[sli][:, None]*rc + row2[slj][None, :]*rc - D2 - tot2

            if i == j:
                idx = torch.arange(U1.shape[0], device=U1.device)
                U1[idx, idx] = 0.
                U2[idx, idx] = 0.

            acc += (2.0 if i != j else 1.0) * (U1 * U2).sum()

    return acc / (n * (n-3))


def JdCov_sq_unbiased_mem(*vars: torch.Tensor,
                      block = None,
                      c: float = 1.0) -> torch.Tensor:
    """
    Unbiased JdCov²_tilde for k ≥ 2 variables.
    * block=None : one (n,n) tensor resident.
    * block=int  : streaming.
    """
    n, k, dtype = vars[0].shape[0], len(vars), vars[0].dtype

    # ------------- full‑matrix path -----------------
    if block is None:
        prod = torch.ones((n, n), dtype=dtype)
        for v in vars:
            prod.mul_(_u_centered_dist_mem(v) + c)
        return prod.sum() / (n*(n-3)) - n/(n-3)

    # ------------- streaming path ------------------
    rc = 1.0/(n-2)
    tc = 1.0/((n-1)*(n-2))

    row_sums = [torch.zeros(n, dtype=dtype) for _ in vars]
    totals   = [torch.zeros((), dtype=dtype) for _ in vars]

    # pass 1 – per‑variable row & total sums
    for i in range(0, n, block):
        sli = slice(i, i+block)
        for idx, v in enumerate(vars):
            D = _pairwise_l2(v[sli], v)
            row_sums[idx][sli] = D.sum(1)
            totals[idx]       += D.sum()

    totals = [t * tc for t in totals]

    # pass 2 – accumulate Σ Π_m U^{(m)}
    acc = torch.zeros((), dtype=dtype)

    for i in range(0, n, block):
        sli = slice(i, i+block)
        for j in range(i, n, block):
            slj = slice(j, j+block)
            prod = None

            for idx, v in enumerate(vars):
                D  = _pairwise_l2(v[sli], v[slj])
                Ui = row_sums[idx][sli][:, None]*rc
                Uj = row_sums[idx][slj][None, :]*rc
                U  = Ui + Uj - D - totals[idx]

                if i == j:
                    idx_diag = torch.arange(U.shape[0], device=U.device)
                    U[idx_diag, idx_diag] = 0.

                U = U + c
                prod = U if prod is None else prod * U

            acc += (2.0 if i != j else 1.0) * prod.sum()

    return acc / (n * (n-3)) - n / (n-3)
