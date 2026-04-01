import torch

def u_centered_dist(x):
    '''
    Input: n-d sample vector x; torch.Tensor
    Output: U-centered matrix; torch.Tensor
    
    This function takes the sample of a random vector, and compute
    the U-centered matrix defined in 3.2 Chakraborty and Zhang (2019).
    '''
    n = x.shape[0]
    if len(x.shape) != 1:
        aux = x.unsqueeze(1).repeat(1, n, 1)
        a = torch.norm((aux - aux.transpose(0,1)).type(torch.DoubleTensor), dim = 2, p = 2)
    else:
        aux = x.repeat(n, 1)
        a = abs(aux - aux.transpose(0,1))

    colsum = a.sum(axis = 1).repeat(n,1) # sum(a_kj)
    rowsum = colsum.transpose(0,1) # sum(a_il)
    totalsum = a.sum() # sum(a_kl)
    
    U_tilde = rowsum/(n-2) + colsum/(n-2) - a - totalsum/((n-1)*(n-2)) 
    U_tilde.fill_diagonal_(0)
    
    return (U_tilde)


def JdCov_sq_unbiased(*var: torch.Tensor) -> torch.Tensor:
    '''
    Input: 1-d sample of vectors vars; torch.Tensor
    Output: Unbiased estimate of squared joint distance covariance; torch.Tensor

    This function can take arbitrary number of variables with same sample sizes
    and calculate the unbiased estimate of JdCov^2 defined by Chakraborty and Zhang (2019).
    Note that this JdCov^2 is not scale invariant. Also using U sacifices non-negativity.
    '''
    c, n = 1, var[0].shape[0] # c = 1, n = number of samples for each variable

    u_centered_mat = torch.stack([u_centered_dist(x)+c for x in var], dim = 0) # calculate the U-centered matrices for each variables
    U_product = torch.prod(u_centered_mat, 0) # multiply all U-centered matrices together
    JdCov2_tilde = torch.sum(U_product)/(n*(n-3)) - n/(n-3)

    return JdCov2_tilde

                              
def sq_dcov_unbiased(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    This function computes the unbiased estimator of squared distance covariance.
    """
    n = x1.shape[0]
    nu2_tilde = torch.sum(torch.mul(u_centered_dist(x1),u_centered_dist(x2)))/(n*(n-3))
    
    return nu2_tilde


def dcorr_unbiased(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    This function computes the unbiased estimator of distance correlation.
    """
    return sq_dcov_unbiased(x1, x2)/torch.sqrt(sq_dcov_unbiased(x1, x1)*sq_dcov_unbiased(x2, x2))
