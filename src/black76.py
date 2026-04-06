"""
black76.py — Black 76 pricing for interest rate derivatives.

Modules:
    black76_caplet   : price a single caplet or floorlet
    black76_cap      : price a cap (sum of caplets)
    black76_swaption : price a European swaption
    implied_vol      : recover implied vol from a market price (numerical inversion)
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ── 1. Core: d1, d2 ──────────────────────────────────────────────────────────

def _d1_d2(F: float, K: float, sigma: float, T: float) -> tuple[float, float]:
    """
    Compute Black 76 d1 and d2.

    Parameters
    ----------
    F     : float — forward rate (decimal, e.g. 0.045)
    K     : float — strike rate (decimal)
    sigma : float — implied vol (annual, decimal)
    T     : float — option expiry in years

    Returns
    -------
    (d1, d2) : tuple[float, float]
    """
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# ── 2. Caplet / Floorlet ─────────────────────────────────────────────────────

def black76_caplet(F: float, K: float, sigma: float, T: float,
                   tau: float, discount: float,
                   is_cap: bool = True) -> dict:
    """
    Price a single caplet or floorlet using Black 76.

    Parameters
    ----------
    F        : float — forward rate for the accrual period (decimal)
    K        : float — strike rate (decimal)
    sigma    : float — implied vol (annual, decimal)
    T        : float — option expiry in years (start of accrual period)
    tau      : float — accrual period length in years (e.g. 0.25 for 3M)
    discount : float — P(0, T+tau) — discount factor to payment date
    is_cap   : bool  — True for caplet, False for floorlet

    Returns
    -------
    dict with keys: price, delta, vega
    """
    d1, d2 = _d1_d2(F, K, sigma, T)

    if is_cap:
        price = discount * tau * (F * norm.cdf(d1) - K * norm.cdf(d2))
        delta = discount * tau * norm.cdf(d1)
    else:
        price = discount * tau * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -discount * tau * norm.cdf(-d1)

    vega = discount * tau * F * norm.pdf(d1) * np.sqrt(T)

    return {"price": price, "delta": delta, "vega": vega}

# ── 3. Cap / Floor ───────────────────────────────────────────────────────────

def black76_cap(schedule: list[dict], K: float, sigma: float,
                is_cap: bool = True) -> dict:
    """
    Price a cap (or floor) as a sum of caplets (floorlets).

    Parameters
    ----------
    schedule : list of dicts, each with keys:
               - F        : forward rate for the period
               - T        : option expiry (years)
               - tau      : accrual length (years)
               - discount : P(0, T+tau)
    K        : float — strike rate (decimal)
    sigma    : float — flat implied vol (decimal)
    is_cap   : bool  — True for cap, False for floor

    Returns
    -------
    dict with keys: price, delta, vega
    """
    total_price = 0.0
    total_delta = 0.0
    total_vega  = 0.0

    for period in schedule:
        result = black76_caplet(
            F=period["F"], K=K, sigma=sigma,
            T=period["T"], tau=period["tau"],
            discount=period["discount"], is_cap=is_cap
        )
        total_price += result["price"]
        total_delta += result["delta"]
        total_vega  += result["vega"]

    return {"price": total_price, "delta": total_delta, "vega": total_vega}


# ── 4. Implied vol ───────────────────────────────────────────────────────────

def implied_vol(market_price: float, F: float, K: float,
                T: float, tau: float, discount: float,
                is_cap: bool = True,
                sigma_lo: float = 1e-4,
                sigma_hi: float = 5.0) -> float:
    """
    Recover implied vol from a market caplet/floorlet price via Brent's method.

    Parameters
    ----------
    market_price : float — observed market price
    F, K, T, tau, discount, is_cap : same as black76_caplet
    sigma_lo, sigma_hi : float — search bounds for vol

    Returns
    -------
    float — implied vol (annual, decimal)
    """
    def objective(sigma):
        return black76_caplet(F, K, sigma, T, tau, discount, is_cap)["price"] - market_price

    return brentq(objective, sigma_lo, sigma_hi, xtol=1e-8)

# ── 5. European Swaption ─────────────────────────────────────────────────────

def black76_swaption(F_swap: float, K: float, sigma: float, T: float,
                     annuity: float, is_payer: bool = True) -> dict:
    """
    Price a European swaption using Black 76.

    A payer swaption gives the right to enter a swap paying fixed K.
    A receiver swaption gives the right to receive fixed K.

    Parameters
    ----------
    F_swap  : float — forward swap rate (decimal)
    K       : float — strike (fixed rate of the underlying swap, decimal)
    sigma   : float — implied vol (annual, decimal)
    T       : float — option expiry in years
    annuity : float — swap annuity = sum of tau_i * P(0, T_i) over swap tenor
    is_payer: bool  — True = payer swaption, False = receiver swaption

    Returns
    -------
    dict with keys: price, delta, vega
    """
    d1, d2 = _d1_d2(F_swap, K, sigma, T)

    if is_payer:
        price = annuity * (F_swap * norm.cdf(d1) - K * norm.cdf(d2))
        delta = annuity * norm.cdf(d1)
    else:
        price = annuity * (K * norm.cdf(-d2) - F_swap * norm.cdf(-d1))
        delta = -annuity * norm.cdf(-d1)

    vega = annuity * F_swap * norm.pdf(d1) * np.sqrt(T)

    return {"price": price, "delta": delta, "vega": vega}


# ── 6. Forward swap rate ──────────────────────────────────────────────────────

def forward_swap_rate(discount_factors: np.ndarray,
                      maturities: np.ndarray,
                      T_start: float, T_end: float,
                      freq: int = 2) -> tuple[float, float]:
    """
    Compute the forward swap rate and annuity for a swap T_start -> T_end.

    Parameters
    ----------
    discount_factors : np.ndarray — P(0, T) on the maturity grid
    maturities       : np.ndarray — maturity grid in years
    T_start          : float — swap start in years
    T_end            : float — swap end in years
    freq             : int   — payment frequency per year (default 2 = semiannual)

    Returns
    -------
    (F_swap, annuity) : tuple[float, float]
    """
    tau = 1.0 / freq
    payment_dates = np.arange(T_start + tau, T_end + 1e-9, tau)

    P_start = float(np.interp(T_start, maturities, discount_factors))
    P_end   = float(np.interp(T_end,   maturities, discount_factors))

    annuity = sum(
        tau * float(np.interp(t, maturities, discount_factors))
        for t in payment_dates
    )

    F_swap = (P_start - P_end) / annuity
    return F_swap, annuity