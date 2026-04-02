import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import torch
import random
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from sklearn.neighbors import KernelDensity

def set_seed(seed):
    """Fix all relevant RNGs for full reproducibility.

    This helper synchronises four independent generators:

    1. PyTorch CPU / CUDA RNG  (torch.manual_seed)
    2. NumPy RNG               (np.random.seed)
    3. Python’s built-in RNG   (random.seed)
    4. CuDNN / autograd kernels (deterministic flags)

    Args:
        seed : int
        The non-negative integer to set for every RNG.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)

def stratified_split_with_tolerance(data, strat_key_col, tolerance=0.1, test_size=0.2):
    """Train/test split with per-stratum jitter around the target ratio."""
    strat_counts = data[strat_key_col].value_counts(normalize=True)

    train_idx, test_idx = [], []

    for strat_value in strat_counts.index:
        stratum_indices = data.index[data[strat_key_col] == strat_value].to_numpy(copy=True)

        frac = test_size + np.random.uniform(-tolerance, tolerance)
        frac = min(max(frac, 0), 1)

        n_test = int(len(stratum_indices) * frac)
        n_test = max(1, n_test)

        stratum_indices = np.random.permutation(stratum_indices)

        test_idx.extend(stratum_indices[:n_test])
        train_idx.extend(stratum_indices[n_test:])

    return data.loc[train_idx], data.loc[test_idx]

def save_run(
    out: list,
    run_name: str = "run",
    path: str = "path"
):
    """Save object to {folder}/{run_name}.pkl.

    Args:
        obj : Any
            Picklable Python object (model state-dict, dict of tensors, etc.).
        run_name : str
            Stem of the output filename (no extension).
        folder : str or Path, default "test_results"
            Destination directory.  Created if it does not exist.
    """
    out_dir = Path(path)
    out_dir.mkdir(exist_ok=True)

    meta = out
    with open(out_dir / f"{run_name}.pkl", "wb") as f:
        pickle.dump(meta, f)

def load_run(run_name: str, path: str, convert_to_hp_dict: bool = False):
    """Load the object stored by `save_run`.

    If `convert_to_hp_dict=True`, transform `best_params` into full-format dict.

    Args:
        run_name : str
            Stem of the filename (same as used in save_run).
        path : str or Path
            Directory that holds the pickle file.
        convert_to_hp_dict : bool
            If True, reformat `best_params` like `module__hidden_dim`, etc.

    Returns:
        dict: reformatted hyperparameter dictionary or original content.
    """
    out_dir = Path(path)
    with open(out_dir / f"{run_name}.pkl", "rb") as f:
        result = pickle.load(f)

    if convert_to_hp_dict and isinstance(result, dict) and "best_params" in result:
        raw_params = result["best_params"]
        hp_dict = {
            "module__num_layers": raw_params["num_layers"],
            "module__hidden_dim": raw_params["hidden_dim"],
            "module__activation": nn.ReLU(),  # adjust if dynamic later
            "module__dropout": raw_params["dropout"],
            "iterator_train__batch_size": raw_params["batch_size"],
            "lr": raw_params["lr"],
            "optimizer__betas": eval(raw_params["betas"]),
            "optimizer__hessian_power": raw_params["hessian_power"],
            "lambda": None,
            "max_epochs": result.get("n_epochs", 100),  # fallback if missing
        }
        return hp_dict

    return result
  
  
def hp_list_to_dict(hp_list):
    """Map skopt positional output (result.x) to a named-parameter dict.
    
    gp_minimize returns the best hyper-parameters as a list in the exact
    order of the search space.  Downstream code (fit_classifier and
    friends) expects a dict whose keys follow the skorch / scikit-learn
    naming convention (module__, iterator_train__, optimizer__).

    This helper converts:

    [num_layers, hidden_dim, activation, dropout,
       batch_size, lr, betas_raw, hessian_power]

    into

    python
    {
        "module__num_layers":            …,
        "module__hidden_dim":            …,
        "module__activation":            …,
        "module__dropout":               …,
        "iterator_train__batch_size":    …,
        "lr":                            …,
        "optimizer__betas":              …,   # tuple
        "optimizer__hessian_power":      …,
        "lmbda":                         None
    }

    Args:
        hp_list : list
            Positional hyper-parameter list as returned by skopt.
    Returns:
        dict
            Dictionary ready to be passed to training / evaluation helpers.

    """
    (num_layers, hidden_dim, activation, dropout,
     batch_size, lr, betas_raw, hessian_power) = hp_list

    return {
        # model / module
        "module__num_layers": num_layers,
        "module__hidden_dim": hidden_dim,
        "module__activation": activation,
        "module__dropout":    dropout,

        # data loader
        "iterator_train__batch_size": int(batch_size),

        # optimiser
        "lr":               lr,
        "optimizer__betas": eval(betas_raw) if isinstance(betas_raw, str) else betas_raw,
        "optimizer__hessian_power": hessian_power,

        # extra slot for λ (fill later)
        "lmbda": None
    }

def load_checkpoint(
    ckpt_file,
    convert_to_hp_dict= True,):
    """
    Load a FairPoissonReg checkpoint saved with torch.save(...).

    Parameters
    ----------
    ckpt_file : str or Path
        Path to the *.pt* file (e.g. "checkpoints/best_checkpoint.pt").
    convert_to_hp_dict : bool, default True
        If True, return the hyper-parameters in the “skorch style”
        expected by BayesSearchCV (keys like 'module__hidden_dim', etc.).
        Otherwise return the raw dict read from torch.load().

    Returns
    -------
    dict
        Either the full checkpoint dictionary (convert_to_hp_dict=False)
        or the reformatted hyper-parameter dictionary.
    """
    ckpt = torch.load(Path(ckpt_file), map_location="cpu", weights_only=False)

    if not convert_to_hp_dict:
        return ckpt                         # raw content

    hp = ckpt["hyperparameters"]

    # build the skorch-style dictionary — mirror of load_run()
    hp_dict = {
        "module__num_layers":  hp["num_layers"],
        "module__hidden_dim":  hp["hidden_dim"],
        "module__activation":  "ReLU",   # adjust if you change activation handling
        "module__dropout":     hp["dropout"],
        "iterator_train__batch_size": hp["batch_size"],
        "lr":                  hp["lr"],
        "optimizer__betas":    eval(hp["betas"]) if isinstance(hp["betas"], str) else hp["betas"],
        "optimizer__hessian_power": hp["hessian_power"],
        "lambda":              hp.get("lambda_reg", None),
        "max_epochs":          ckpt.get("total_epochs", hp["max_epochs"]),
    }
    return hp_dict
