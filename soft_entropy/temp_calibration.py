"""Temperature calibration for spherical smoothing (vMF).

Solves D_KL(V_{eps, north} || U) = log(m) for eps on S^{d-1},
with U the uniform *probability* measure 
(not scipy's surface-area reference). Identity:

        (Amos ratio / eps) - (log vMF partition fn)
    D_KL = A_nu(kappa)/eps - log Z_d(eps),  
    
            nu = d/2 - 1,  kappa = 1/eps

Public names mirror hcoxec/soft_h/soft_entropy/temp_calibration.py:
G, Phi_minus, Phi_plus, KL_lower_bound, KL_upper_bound,
find_eps. KL_exact and sphere_temp_calibration are additions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, ive


def _log_iv(nu: float, kappa: float) -> float:
    """log-Bessel function"""
    return np.log(ive(nu, kappa)) + kappa


def _A_nu(nu: float, kappa: float) -> float:
    """exact Amos ratio."""
    return ive(nu + 1.0, kappa) / ive(nu, kappa)


def _log_Z_d(eps: float, d: int) -> float:
    """exact vMF log partition function"""
    nu = d / 2.0 - 1.0
    return (d / 2.0 - 1.0) * np.log(2.0 * eps) + gammaln(d / 2.0) + _log_iv(nu, 1.0 / eps)


def G(alpha: float, beta: float, kappa: float) -> float:
    """Amos bound kappa / (alpha + sqrt(kappa^2 + beta^2))."""
    return kappa / (alpha + np.sqrt(kappa * kappa + beta * beta))


def Phi_minus(eps: float, d: int) -> float:
    """lower Amos envelope for log Z_d(eps)."""
    inv = 1.0 / eps
    half = d / 2.0
    root = np.sqrt(inv * inv + half * half)
    return root - half + half * np.log(d / (half + root))


def Phi_plus(eps: float, d: int) -> float:
    """upper Amos envelope for log Z_d(eps). Singular at d = 2."""
    nu = d / 2.0 - 1.0
    inv = 1.0 / eps
    root = np.sqrt(inv * inv + nu * nu)
    return root - nu + nu * np.log((d - 2.0) / (nu + root))


def KL_lower_bound(eps: float, d: int) -> float:
    """lower bound on D_KL(V_eps || U)."""
    return G(d / 2.0, d / 2.0, 1.0 / eps) / eps - Phi_plus(eps, d)


def KL_upper_bound(eps: float, d: int) -> float:
    """upper bound on D_KL(V_eps || U)."""
    nu = d / 2.0 - 1.0
    return G(nu, nu, 1.0 / eps) / eps - Phi_minus(eps, d)


def KL_exact(eps: float, d: int) -> float:
    """Direct stable D_KL via Bessel (ive). Under/overflows once d
    is several hundred at eps ~ 1e-2 — use bounds beyond that."""
    nu = d / 2.0 - 1.0
    return _A_nu(nu, 1.0 / eps) / eps - _log_Z_d(eps, d)


def _solve_monotone(fn, target: float, bracket: tuple[float, float]) -> float:
    """Brentq for monotone-decreasing fn on bracket."""
    lo, hi = bracket
    f_lo = fn(lo) - target
    f_hi = fn(hi) - target
    if f_lo < 0:
        raise ValueError(
            f"Bracket low eps={lo} gives KL={f_lo + target:.4g} < target "
            f"{target:.4g}; widen eps_bracket downward."
        )
    if f_hi > 0:
        raise ValueError(
            f"Bracket high eps={hi} gives KL={f_hi + target:.4g} > target "
            f"{target:.4g}; widen eps_bracket upward."
        )
    return brentq(lambda e: fn(e) - target, lo, hi, xtol=1e-12, rtol=1e-10)


def find_eps(m: int, d: int, eps_range: tuple[float, float] | list = (1e-5, 1e0)) -> float:
    """Upstream-compatible wrapper. Returns *eps* solving
    Psi^-(eps, d) = log(m) via Brentq (true KL >= log(m), conservative).
    Requires d > 2."""
    return sphere_temp_calibration(
        m_bins=m, d_dim=d, mode="bounds", eps_bracket=tuple(eps_range),
    )


def sphere_temp_calibration(
    m_bins: int,
    d_dim: int,
    mode: str = "auto",
    d_threshold: int = 100,
    eps_bracket: tuple[float, float] = (1e-6, 1.0),
) -> float:
    """Solve D_KL(V_{eps, north} || U) = log(m_bins) for eps.

    mode: "exact"  -- direct Bessel, accurate at low d,
          "bounds" -- Brentq on Psi^-, conservative; needs d > 2,
          "auto"   -- exact if d < d_threshold, else bounds.
    eps_bracket: D_KL is decreasing in eps; low must over-target, 
                 high must under-target.
    """
    if d_dim < 2:
        raise ValueError(f"d_dim must be >= 2, got {d_dim}")
    if m_bins < 2:
        raise ValueError(f"m_bins must be >= 2, got {m_bins}")

    if mode == "auto":
        mode = "exact" if d_dim < d_threshold else "bounds"
    if mode not in {"exact", "bounds"}:
        raise ValueError(f"mode must be 'exact', 'bounds', or 'auto'; got {mode!r}")
    if mode == "bounds" and d_dim <= 2:
        raise ValueError("'bounds' mode requires d_dim > 2 (Psi^+ singular at d=2)")

    target = float(np.log(m_bins))
    kl_fn = KL_exact if mode == "exact" else KL_lower_bound
    return _solve_monotone(lambda e: kl_fn(e, d_dim), target, eps_bracket)
