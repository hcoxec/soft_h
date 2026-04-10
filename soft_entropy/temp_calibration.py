import numpy as np
from scipy.stats import vonmises_fisher
from scipy.special import iv, gamma
from scipy.integrate import quad, nquad

def G(alpha, beta, kappa):
    """
    Amos ratio bound function: G_{alpha,beta}(kappa) = kappa / (alpha + sqrt(kappa^2 + beta^2))
    """
    return kappa / (alpha + np.sqrt(kappa**2 + beta**2))

def Phi_minus(eps, d):
    """
    Lower envelope for log partition function
    """
    term1 = np.sqrt(eps**(-2) + (d/2)**2)
    term2 = d/2
    term3 = (d/2) * np.log(d / (d/2 + np.sqrt(eps**(-2) + (d/2)**2)))
    return term1 - term2 + term3

def Phi_plus(eps, d):
    """
    Upper envelope for log partition function
    """
    term1 = np.sqrt(eps**(-2) + (d/2 - 1)**2)
    term2 = d/2 - 1
    term3 = (d/2 - 1) * np.log((d - 2) / ((d/2 - 1) + np.sqrt(eps**(-2) + (d/2 - 1)**2)))
    return term1 - term2 + term3

def KL_lower_bound(eps, d):
    """
    Lower bound on KL(vMF || uniform)
    """
    kappa = 1 / eps
    G_term = G(d/2, d/2, kappa)
    return G_term / eps - Phi_plus(eps, d)

def KL_upper_bound(eps, d):
    """
    Upper bound on KL(vMF || uniform)
    """
    kappa = 1 / eps
    nu = d/2 - 1
    G_term = G(nu, nu, kappa)
    return G_term / eps - Phi_minus(eps, d)


def find_eps(m, d, eps_range = [1e-5, 1e0]):
    for eps in np.geomspace(eps_range[0], eps_range[1], 100):
        if KL_lower_bound(eps, d) <= np.log(m):
            return eps
