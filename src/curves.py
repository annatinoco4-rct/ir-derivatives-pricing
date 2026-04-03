"""
curves.py — Yield curve construction and discount factor bootstrapping.

Modules:
    bond_price          : price a fixed-coupon bond given YTM
    bootstrap_curve     : extract discount factors P(0, T) from market instruments
    forward_rate        : compute forward rates from discount factors
    spot_rate           : convert discount factors to continuously-compounded spot rates
"""

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import CubicSpline


# ── 1. Bond pricing ──────────────────────────────────────────────────────────

def bond_price(face: float, coupon_rate: float, ytm: float,
               n_years: int, freq: int = 2) -> float:
    """
    Price a fixed-coupon bond given yield to maturity.

    Parameters
    ----------
    face        : float — face/par value (e.g. 1000)
    coupon_rate : float — annual coupon rate (e.g. 0.085 for 8.5%)
    ytm         : float — yield to maturity, annual (e.g. 0.092)
    n_years     : int   — years to maturity
    freq        : int   — coupon payments per year (default: 2 semiannual)

    Returns
    -------
    float — dirty price of the bond
    """
    n_periods = n_years * freq
    t = np.arange(1, n_periods + 1)
    c = np.full(n_periods, coupon_rate * face / freq)
    c[-1] += face
    discount_factors = 1 / (1 + ytm / freq) ** t
    return float(np.sum(c * discount_factors))

# ── 2. Bootstrapping ─────────────────────────────────────────────────────────

def bootstrap_curve(maturities: np.ndarray, yields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap a discount factor curve from par yields (e.g. DGS Treasury yields).

    Assumes instruments are zero-coupon (or par bonds) at each maturity.
    For par bonds: coupon rate = yield, so price = par = 100.

    Parameters
    ----------
    maturities : np.ndarray — maturities in years (e.g. [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields     : np.ndarray — par yields, annual, decimal (e.g. 0.045 for 4.5%)

    Returns
    -------
    maturities : np.ndarray — same input maturities
    discount_factors : np.ndarray — P(0, T) for each maturity
    """
    n = len(maturities)
    P = np.zeros(n)  # discount factors P(0, T_i)

    for i, (T, y) in enumerate(zip(maturities, yields)):
        if T <= 1.0:
            # Short end: treat as zero-coupon instrument
            P[i] = 1 / (1 + y * T)
        else:
            # Par bond: coupon = yield, price = 1 (normalized)
            # 1 = y * sum_{j: T_j <= T} P(0, T_j) * delta_j + P(0, T)
            # Solve for P(0, T) given all previous P already bootstrapped
            coupon_times = maturities[:i]  # all previous nodes
            coupon_dfs = P[:i]             # their discount factors

            # Build interpolated DFs for annual coupon dates before T
            annual_dates = np.arange(1.0, T, 1.0)
            if len(annual_dates) > 0 and len(coupon_times) >= 2:
                cs = CubicSpline(maturities[:i+1] if i > 0 else [T],
                                 P[:i+1] if i > 0 else [1.0],
                                 extrapolate=False)
                interp_dfs = np.interp(annual_dates, coupon_times, coupon_dfs)
                pv_coupons = y * np.sum(interp_dfs)
            else:
                pv_coupons = 0.0

            P[i] = (1 - pv_coupons) / (1 + y)

    return maturities, P


# ── 3. Forward rates ─────────────────────────────────────────────────────────

def forward_rate(maturities: np.ndarray, discount_factors: np.ndarray,
                 T1: float, T2: float) -> float:
    """
    Compute the simply-compounded forward rate f(T1, T2) from discount factors.

    Parameters
    ----------
    maturities       : np.ndarray — maturity grid
    discount_factors : np.ndarray — P(0, T) on the same grid
    T1               : float — start of forward period (years)
    T2               : float — end of forward period (years)

    Returns
    -------
    float — simply-compounded forward rate for [T1, T2]
    """
    P_T1 = float(np.interp(T1, maturities, discount_factors))
    P_T2 = float(np.interp(T2, maturities, discount_factors))
    tau = T2 - T1
    return (P_T1 / P_T2 - 1) / tau


# ── 4. Spot rates ────────────────────────────────────────────────────────────

def spot_rate(maturity: float, discount_factor: float) -> float:
    """
    Convert a discount factor P(0, T) to a continuously-compounded spot rate.

    Parameters
    ----------
    maturity        : float — T in years
    discount_factor : float — P(0, T)

    Returns
    -------
    float — continuously-compounded spot rate r(T) = -ln(P(0,T)) / T
    """
    return -np.log(discount_factor) / maturity

# ── 5. Market data — FRED Treasury yields ────────────────────────────────────

FRED_TICKERS = {
    "DGS1MO": 1/12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS3":   3.0,
    "DGS5":   5.0,
    "DGS7":   7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}


def fetch_treasury_yields() -> tuple[np.ndarray, np.ndarray]:
    """
    Download the most recent on-the-run US Treasury par yields from FRED.

    Returns
    -------
    maturities : np.ndarray — tenor in years
    yields     : np.ndarray — par yields in decimal (e.g. 0.045 for 4.5%)
    """
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    maturities, yields = [], []

    for ticker, maturity in FRED_TICKERS.items():
        url = base + ticker
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        # Last non-missing row
        for line in reversed(lines[1:]):
            date, val = line.split(",")
            if val.strip() != ".":
                maturities.append(maturity)
                yields.append(float(val) / 100)
                break

    idx = np.argsort(maturities)
    return np.array(maturities)[idx], np.array(yields)[idx]