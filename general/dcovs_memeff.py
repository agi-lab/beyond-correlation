import torch
from typing import Tuple

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
        return (_u_centered_dist(x1) *
                _u_centered_dist(x2)).sum() / (n*(n-3))

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
            prod.mul_(_u_centered_dist(v) + c)
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
