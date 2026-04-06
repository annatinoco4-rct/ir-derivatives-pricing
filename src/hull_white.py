"""
hull_white.py — Hull-White 1F model: calibration + Monte Carlo pricing.

Modules:
    hw_discount_factor  : analytical P(0,T) under Hull-White
    simulate_paths      : Monte Carlo simulation of short rate r(t)
    hw_swaption_mc      : price a European/Bermudan swaption via MC
    calibrate           : fit (a, sigma) to market swaption prices
"""

import numpy as np
from scipy.optimize import minimize
import sys
sys.path.insert(0, 'src')
from black76 import black76_swaption, forward_swap_rate


# ── 1. Analytical discount factor under Hull-White ───────────────────────────

def _B(a: float, t: float, T: float) -> float:
    """B(t,T) function in Hull-White: B = (1 - e^{-a(T-t)}) / a"""
    if abs(a) < 1e-8:
        return T - t
    return (1 - np.exp(-a * (T - t))) / a


def hw_discount_factor(r_t: float, a: float, sigma: float,
                       maturities: np.ndarray, discount_factors: np.ndarray,
                       t: float, T: float) -> float:
    """
    Analytical P(t, T) under Hull-White given current short rate r(t).

    P(t,T) = A(t,T) * exp(-B(t,T) * r(t))

    where:
        B(t,T) = (1 - e^{-a(T-t)}) / a
        ln A(t,T) = ln(P(0,T)/P(0,t)) + B(t,T)*f(0,t)
                    - sigma^2/(4a) * B(t,T)^2 * (1 - e^{-2at})
    """
    if abs(T - t) < 1e-8:
        return 1.0

    P0t = float(np.interp(t, maturities, discount_factors))
    P0T = float(np.interp(T, maturities, discount_factors))

    B_tT = _B(a, t, T)

    eps = 1e-4
    P0t_plus = float(np.interp(t + eps, maturities, discount_factors))
    f0t = -np.log(P0t_plus / P0t) / eps

    ln_A = (np.log(P0T / P0t)
            + B_tT * f0t
            - (sigma**2 / (4 * a)) * B_tT**2 * (1 - np.exp(-2 * a * t)))

    return np.exp(ln_A - B_tT * r_t)


# ── 2. Monte Carlo simulation ─────────────────────────────────────────────────

def simulate_paths(r0: float, a: float, sigma: float, theta: np.ndarray,
                   dt: float, n_steps: int, n_paths: int,
                   seed: int = 42) -> np.ndarray:
    """
    Simulate short rate paths under Hull-White using Euler-Maruyama.

    dr(t) = [theta(t) - a*r(t)] dt + sigma * dW(t)

    Parameters
    ----------
    r0      : float      — initial short rate
    a       : float      — mean reversion speed
    sigma   : float      — short rate vol
    theta   : np.ndarray — theta(t) on the time grid (length n_steps)
    dt      : float      — time step in years
    n_steps : int        — number of time steps
    n_paths : int        — number of Monte Carlo paths
    seed    : int        — random seed for reproducibility

    Returns
    -------
    np.ndarray — shape (n_paths, n_steps+1) — simulated short rate paths
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)

    for i in range(n_steps):
        r = paths[:, i]
        paths[:, i + 1] = (r
                           + (theta[i] - a * r) * dt
                           + sigma * dW[:, i])

    return paths


def _build_theta(a: float, sigma: float,
                 maturities: np.ndarray, discount_factors: np.ndarray,
                 dt: float, n_steps: int) -> np.ndarray:
    """
    Build theta(t) to fit the initial yield curve exactly.

    theta(t) = df(0,t)/dt + a*f(0,t) + sigma^2/(2a)*(1 - e^{-2at})
    """
    times = np.linspace(0, n_steps * dt, n_steps + 1)
    eps   = 1e-4

    # Instantaneous forward rates via finite difference on log P
    def f(t):
        if t <= 0:
            t = eps
        P1 = float(np.interp(t,       maturities, discount_factors))
        P2 = float(np.interp(t + eps, maturities, discount_factors))
        return -np.log(P2 / P1) / eps

    theta = np.array([
        (f(t + eps) - f(t)) / eps
        + a * f(t)
        + sigma**2 / (2 * a) * (1 - np.exp(-2 * a * t))
        if t > 0 else
        (f(eps) - f(0)) / eps + a * f(eps)
        for t in times[:-1]
    ])

    return theta


# ── 3. Swaption pricing via Monte Carlo ───────────────────────────────────────

def hw_swaption_mc(r0: float, a: float, sigma: float,
                   maturities: np.ndarray, discount_factors: np.ndarray,
                   T_expiry: float, T_start: float, T_end: float,
                   K: float, freq: int = 2,
                   n_paths: int = 10_000, dt: float = 1/52,
                   is_payer: bool = True,
                   seed: int = 42) -> dict:
    """
    Price a European swaption via Hull-White Monte Carlo.

    Parameters
    ----------
    r0, a, sigma      : Hull-White parameters
    maturities        : np.ndarray — market maturity grid
    discount_factors  : np.ndarray — market P(0,T)
    T_expiry          : float — option expiry (years)
    T_start           : float — swap start (years, usually = T_expiry)
    T_end             : float — swap end (years)
    K                 : float — strike rate (decimal)
    freq              : int   — swap payment frequency per year
    n_paths           : int   — number of MC paths
    dt                : float — time step (default weekly = 1/52)
    is_payer          : bool  — True = payer swaption
    seed              : int   — random seed

    Returns
    -------
    dict with keys: price, std_error, ci_lower, ci_upper
    """
    n_steps = int(T_expiry / dt)
    theta   = _build_theta(a, sigma, maturities, discount_factors, dt, n_steps)
    paths   = simulate_paths(r0, a, sigma, theta, dt, n_steps, n_paths, seed)

    r_expiry = paths[:, -1]  # short rate at expiry on each path

    tau  = 1.0 / freq
    payment_dates = np.arange(T_start + tau, T_end + 1e-9, tau)

    payoffs = np.zeros(n_paths)

    for i in range(n_paths):
        r_t = r_expiry[i]
        # Analytical bond prices P(T_expiry, Tⱼ) for each payment date
        P_payments = np.array([
            hw_discount_factor(r_t, a, sigma, maturities,
                               discount_factors, T_expiry, Tj)
            for Tj in payment_dates
        ])
        P_start = hw_discount_factor(r_t, a, sigma, maturities,
                                     discount_factors, T_expiry, T_start)
        P_end   = hw_discount_factor(r_t, a, sigma, maturities,
                                     discount_factors, T_expiry, T_end)

        annuity_i = tau * np.sum(P_payments)
        swap_val  = (P_start - P_end - K * annuity_i)

        if is_payer:
            payoffs[i] = max(swap_val, 0.0)
        else:
            payoffs[i] = max(-swap_val, 0.0)

    # Discount payoffs back to today
    P0_expiry = float(np.interp(T_expiry, maturities, discount_factors))
    discounted = payoffs * P0_expiry

    price     = np.mean(discounted)
    std_error = np.std(discounted) / np.sqrt(n_paths)

    return {
        "price":    price,
        "std_error": std_error,
        "ci_lower": price - 1.96 * std_error,
        "ci_upper": price + 1.96 * std_error,
    }