"""
sabr.py — SABR model: implied vol formula + smile calibration.

Modules:
    sabr_vol        : Hagan 2002 implied vol for a given strike
    sabr_smile      : compute vol smile across a strike grid
    calibrate_sabr  : fit (alpha, rho, nu) to market vol smile (beta fixed)
"""

import numpy as np
from scipy.optimize import minimize


# ── 1. Hagan 2002 SABR implied vol ───────────────────────────────────────────

def sabr_vol(F: float, K: float, T: float,
             alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan et al. (2002) approximation for SABR implied vol.

    Parameters
    ----------
    F     : float — forward rate (decimal)
    K     : float — strike (decimal)
    T     : float — option expiry in years
    alpha : float — vol level parameter
    beta  : float — elasticity (fixed, typically 0.5 for rates)
    rho   : float — correlation forward-vol (-1 < rho < 1)
    nu    : float — vol of vol (> 0)

    Returns
    -------
    float — Black implied vol (annual, decimal)
    """
    # ATM case
    if abs(F - K) < 1e-8:
        FK_mid = F ** (1 - beta)
        term1  = alpha / FK_mid
        term2  = (1 + (((1 - beta)**2 / 24) * alpha**2 / FK_mid**2
                       + (rho * beta * nu * alpha) / (4 * FK_mid)
                       + (2 - 3 * rho**2) / 24 * nu**2) * T)
        return term1 * term2

    # General case
    FK   = F * K
    FKmb = FK ** ((1 - beta) / 2)
    log_FK = np.log(F / K)

    # z and x(z)
    z    = (nu / alpha) * FKmb * log_FK
    x_z  = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    # Numerator
    numer = alpha * (1 + (((1 - beta)**2 / 24) * alpha**2 / FKmb**2
                          + (rho * beta * nu * alpha) / (4 * FKmb)
                          + (2 - 3 * rho**2) / 24 * nu**2) * T)

    # Denominator
    denom = (FKmb * (1
                     + (1 - beta)**2 / 24 * log_FK**2
                     + (1 - beta)**4 / 1920 * log_FK**4)
             * (x_z / z))

    return numer / denom


# ── 2. Vol smile across strikes ───────────────────────────────────────────────

def sabr_smile(F: float, strikes: np.ndarray, T: float,
               alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """
    Compute SABR implied vol smile across a grid of strikes.

    Parameters
    ----------
    F       : float      — forward rate
    strikes : np.ndarray — strike grid (decimal)
    T       : float      — expiry in years
    alpha, beta, rho, nu : SABR parameters

    Returns
    -------
    np.ndarray — implied vols for each strike
    """
    return np.array([
        sabr_vol(F, K, T, alpha, beta, rho, nu)
        for K in strikes
    ])


# ── 3. Calibration ────────────────────────────────────────────────────────────

def calibrate_sabr(F: float, T: float, beta: float,
                   strikes: np.ndarray,
                   market_vols: np.ndarray) -> dict:
    """
    Calibrate SABR (alpha, rho, nu) to a market vol smile with beta fixed.

    Minimizes sum of squared errors between model and market vols.

    Parameters
    ----------
    F           : float      — forward rate
    T           : float      — expiry in years
    beta        : float      — fixed elasticity (e.g. 0.5)
    strikes     : np.ndarray — market strikes
    market_vols : np.ndarray — market implied vols for each strike

    Returns
    -------
    dict with keys: alpha, beta, rho, nu, rmse
    """
    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
            return 1e6
        try:
            model_vols = sabr_smile(F, strikes, T, alpha, beta, rho, nu)
            return float(np.sum((model_vols - market_vols)**2))
        except Exception:
            return 1e6

    # Initial guess: alpha ≈ ATM vol, rho = 0, nu = 0.3
    atm_vol = float(np.interp(F, strikes, market_vols))
    x0      = [atm_vol, 0.0, 0.3]
    bounds  = [(1e-4, 2.0), (-0.999, 0.999), (1e-4, 5.0)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-12, 'gtol': 1e-8})

    alpha_opt, rho_opt, nu_opt = result.x
    model_vols = sabr_smile(F, strikes, T, alpha_opt, beta, rho_opt, nu_opt)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols)**2)))

    return {
        "alpha": alpha_opt,
        "beta":  beta,
        "rho":   rho_opt,
        "nu":    nu_opt,
        "rmse":  rmse,
    }