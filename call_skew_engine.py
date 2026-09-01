"""Call skew, ORDC-style scarcity, and full reval for a packed ERCOT option book."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

Cp = Literal["call", "put"]

VOL_FLOOR = 0.04
VOL_CAP = 4.0
DAYS_PER_YEAR = 365.0
SHIFT = 50.0  # shifted Black so F can print negative
HCAP_PRE = 9_000.0  # through 31 May 2021
HCAP_POST = 5_000.0  # from 1 Jun 2021
URI_CUTOVER = pd.Timestamp("2021-06-01")


@dataclass(frozen=True)
class SmileParams:
    """Quadratic smile in log-moneyness k = ln((K+s)/(F+s)). β>0 is call/spike skew."""

    sigma_atm: float = 1.15
    skew: float = 0.42  # call skew
    smile: float = 0.28
    vol_floor: float = VOL_FLOOR
    vol_cap: float = VOL_CAP
    shift: float = SHIFT


POWER_SMILE = SmileParams()
EQUITY_SMILE = SmileParams(sigma_atm=0.18, skew=-0.28, smile=0.18, shift=0.0)


@dataclass
class OptionLeg:
    cp: Cp
    strike: float
    days_to_expiry: float
    quantity: float
    multiplier: float = 1.0
    label: str = ""

    def tenor(self) -> float:
        return max(float(self.days_to_expiry) / DAYS_PER_YEAR, 1.0 / DAYS_PER_YEAR)


@dataclass
class LegReval:
    label: str
    cp: str
    strike: float
    F0: float
    F1: float
    T0: float
    T1: float
    sigma0: float
    sigma1: float
    value0: float
    value1: float
    pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    unexplained: float


def offer_cap(ts: pd.Timestamp | None = None) -> float:
    if ts is None:
        return HCAP_POST
    t = pd.Timestamp(ts)
    return HCAP_PRE if t < URI_CUTOVER else HCAP_POST


def clip_vol(sigma, floor: float = VOL_FLOOR, cap: float = VOL_CAP):
    return np.clip(sigma, floor, cap)


def _shifted(x: float | np.ndarray, shift: float) -> np.ndarray:
    return np.asarray(x, dtype=float) + float(shift)


def smile_vol(K, F: float, p: SmileParams):
    """σ(K; F) = σ_ATM + β k + χ k², k = ln((K+s)/(F+s))."""
    Ks = np.maximum(_shifted(K, p.shift), 1e-8)
    Fs = max(float(F) + p.shift, 1e-8)
    k = np.log(Ks / Fs)
    raw = p.sigma_atm + p.skew * k + p.smile * (k**2)
    return clip_vol(raw, p.vol_floor, p.vol_cap)


def _d1_d2(F: float, K: float, T: float, sigma: float, shift: float) -> tuple[float, float]:
    Fs = max(float(F) + shift, 1e-8)
    Ks = max(float(K) + shift, 1e-8)
    T = max(float(T), 1e-12)
    sigma = max(float(sigma), 1e-8)
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(Fs / Ks) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return float(d1), float(d2)


def black76_price(F: float, K: float, T: float, sigma: float, cp: Cp, shift: float = SHIFT) -> float:
    T = max(float(T), 0.0)
    if T <= 1e-8:
        return max(F - K, 0.0) if cp == "call" else max(K - F, 0.0)
    d1, d2 = _d1_d2(F, K, T, sigma, shift)
    Fs = float(F) + shift
    Ks = float(K) + shift
    if cp == "call":
        return Fs * norm.cdf(d1) - Ks * norm.cdf(d2)
    return Ks * norm.cdf(-d2) - Fs * norm.cdf(-d1)


def black76_greeks(
    F: float, K: float, T: float, sigma: float, cp: Cp, shift: float = SHIFT
) -> dict[str, float]:
    """Per-unit Greeks on the unshifted price. Vega is dV/dσ (σ decimal)."""
    T = max(float(T), 1e-12)
    sigma = max(float(sigma), 1e-8)
    d1, _ = _d1_d2(F, K, T, sigma, shift)
    Fs = float(F) + shift
    n_d1 = float(norm.pdf(d1))
    sqrt_t = np.sqrt(T)
    gamma = n_d1 / (max(Fs, 1e-8) * sigma * sqrt_t)
    vega = Fs * n_d1 * sqrt_t
    d_price_dT = Fs * n_d1 * sigma / (2.0 * sqrt_t)
    theta = -d_price_dT / DAYS_PER_YEAR
    if cp == "call":
        delta = float(norm.cdf(d1))
    else:
        delta = float(norm.cdf(d1)) - 1.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "d1": d1}


def risk_reversal(F: float, p: SmileParams, wing: float = 0.25) -> dict[str, float]:
    """25Δ-ish wings via log-moneyness ≈ ±wing.

    RR  = σ_call − σ_put   (sign = skew direction; >0 is call skew)
    Fly = ½(σ_call + σ_put) − σ_ATM  (curvature / fat-tail richness)
    """
    k_call = float(np.exp(wing) * (F + p.shift) - p.shift)
    k_put = float(np.exp(-wing) * (F + p.shift) - p.shift)
    sig_c = float(smile_vol(k_call, F, p))
    sig_p = float(smile_vol(k_put, F, p))
    return {
        "K_call": k_call,
        "K_put": k_put,
        "sig_call": sig_c,
        "sig_put": sig_p,
        "sig_atm": float(p.sigma_atm),
        "rr": sig_c - sig_p,
        "fly": 0.5 * (sig_c + sig_p) - float(p.sigma_atm),
        "wing": float(wing),
    }


def butterfly(F: float, p: SmileParams, wing: float = 0.25) -> dict[str, float]:
    """Alias of the fly half of risk_reversal — curvature of the smile."""
    return risk_reversal(F, p, wing=wing)


def smile_frame(F: float, p: SmileParams, n: int = 161) -> pd.DataFrame:
    k = np.linspace(-0.85, 0.95, n)
    K = np.exp(k) * (F + p.shift) - p.shift
    K = np.maximum(K, -p.shift + 0.5)
    sig = np.asarray(smile_vol(K, F, p), dtype=float)
    return pd.DataFrame({"K": K, "k": k, "iv": sig, "moneyness": K / max(F, 1e-8)})


def ordc_adder(reserves_mw: np.ndarray, voll: float = HCAP_POST, x_mw: float = 2_000.0) -> np.ndarray:
    """Stylized ORDC: adder ramps from 0 at 2X MW toward VOLL as reserves → 0."""
    r = np.clip(np.asarray(reserves_mw, dtype=float), 0.0, None)
    x = max(float(x_mw), 1.0)
    # Two-step curve in the spirit of pre-RTC ORDC (online + offline).
    online = np.clip(1.0 - r / x, 0.0, 1.0)
    offline = np.clip(1.0 - r / (2.0 * x), 0.0, 1.0)
    # Convex scarcity: more weight as reserves collapse.
    weight = 0.65 * (online**1.4) + 0.35 * (offline**1.1)
    return float(voll) * weight


def default_pack(F: float, call_pct: float = 2.5, put_pct: float = 0.35, days: float = 30.0) -> list[OptionLeg]:
    """Short OTM call (scarcity/cap) + short OTM put (crash/floor). Quantity < 0 is short."""
    F = max(float(F), 1.0)
    k_call = round(max(F * call_pct, F + 25.0), 2)
    k_put = round(max(F * put_pct, 5.0), 2)
    return [
        OptionLeg("call", k_call, days, -1.0, 1.0, f"Short call {k_call:g}"),
        OptionLeg("put", k_put, days, -1.0, 1.0, f"Short put {k_put:g}"),
    ]


def revalue_leg(
    leg: OptionLeg,
    F0: float,
    F1: float,
    p: SmileParams,
    sigma1: float | None = None,
    dt_years: float = 0.0,
) -> LegReval:
    T0 = leg.tenor()
    T1 = max(T0 - dt_years, 1.0 / DAYS_PER_YEAR)
    K = float(leg.strike)
    sig0 = float(smile_vol(K, F0, p))
    if sigma1 is None:
        sig1 = float(smile_vol(K, F1, p))
    else:
        sig1 = float(clip_vol(sigma1, p.vol_floor, p.vol_cap))
    v0 = black76_price(F0, K, T0, sig0, leg.cp, p.shift)
    v1 = black76_price(F1, K, T1, sig1, leg.cp, p.shift)
    g = black76_greeks(F0, K, T0, sig0, leg.cp, p.shift)
    scale = float(leg.quantity) * float(leg.multiplier)
    dF = float(F1) - float(F0)
    dsig = sig1 - sig0
    delta_pnl = g["delta"] * dF * scale
    gamma_pnl = 0.5 * g["gamma"] * (dF**2) * scale
    vega_pnl = g["vega"] * dsig * scale
    theta_pnl = g["theta"] * scale * (dt_years * DAYS_PER_YEAR)
    pnl = (v1 - v0) * scale
    unexplained = pnl - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)
    return LegReval(
        label=leg.label or f"{leg.cp} {K:g}",
        cp=leg.cp,
        strike=K,
        F0=float(F0),
        F1=float(F1),
        T0=T0,
        T1=T1,
        sigma0=sig0,
        sigma1=sig1,
        value0=v0 * scale,
        value1=v1 * scale,
        pnl=pnl,
        delta=g["delta"] * scale,
        gamma=g["gamma"] * scale,
        vega=g["vega"] * scale,
        theta=g["theta"] * scale,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        unexplained=unexplained,
    )


def revalue_book(
    legs: list[OptionLeg],
    F0: float,
    F1: float,
    p: SmileParams,
    sigma_bump: float = 0.0,
    dt_years: float = 0.0,
) -> tuple[list[LegReval], pd.DataFrame]:
    rows: list[LegReval] = []
    for leg in legs:
        sig1 = float(smile_vol(leg.strike, F1, p)) + float(sigma_bump)
        rows.append(revalue_leg(leg, F0, F1, p, sigma1=sig1, dt_years=dt_years))
    table = pd.DataFrame(
        [
            {
                "Label": r.label,
                "Type": r.cp,
                "Strike": r.strike,
                "F0": r.F0,
                "F1": r.F1,
                "IV t": r.sigma0,
                "IV t+1": r.sigma1,
                "Value t": r.value0,
                "Value t+1": r.value1,
                "P&L (full reval)": r.pnl,
                "Delta P&L": r.delta_pnl,
                "Gamma P&L": r.gamma_pnl,
                "Vega P&L": r.vega_pnl,
                "Theta P&L": r.theta_pnl,
                "Unexplained": r.unexplained,
                "Delta": r.delta,
                "Gamma": r.gamma,
                "Vega": r.vega,
            }
            for r in rows
        ]
    )
    return rows, table


def pnl_vs_spot(
    legs: list[OptionLeg],
    F0: float,
    p: SmileParams,
    spots: np.ndarray,
    sigma_bump: float = 0.0,
    dt_years: float = 0.0,
) -> pd.DataFrame:
    records = []
    for F1 in np.asarray(spots, dtype=float):
        rows, _ = revalue_book(legs, F0, float(F1), p, sigma_bump=sigma_bump, dt_years=dt_years)
        rec = {"F1": float(F1), "total": sum(r.pnl for r in rows)}
        for r in rows:
            rec[r.label] = r.pnl
        rec["delta"] = sum(r.delta_pnl for r in rows)
        rec["gamma"] = sum(r.gamma_pnl for r in rows)
        rec["vega"] = sum(r.vega_pnl for r in rows)
        records.append(rec)
    return pd.DataFrame(records)


def commentary(rows: list[LegReval], F0: float, F1: float, sigma_bump: float) -> list[str]:
    total = sum(r.pnl for r in rows)
    g = sum(r.gamma_pnl for r in rows)
    v = sum(r.vega_pnl for r in rows)
    d = sum(r.delta_pnl for r in rows)
    dF = F1 - F0
    bits = [
        f"Full reval P&L on the pack is **{total:,.0f}** USD/MWh-unit "
        f"(spot {F0:.1f} → {F1:.1f}, vol bump {100 * sigma_bump:+.0f} pp)."
    ]
    if dF > 0 and sigma_bump > 0:
        bits.append(
            "Short **call** is the scarcity path: ORDC/adder through the offer cap. "
            f"Gamma P&L {g:,.0f} and vega P&L {v:,.0f} **compound** with delta {d:,.0f} — "
            "a first-order hedge misses the spike."
        )
    elif dF < 0:
        bits.append(
            "Short **put** is the crash/freeze demand path: negative or collapsing RT LMP. "
            f"Delta {d:,.0f}, gamma {g:,.0f}. Packing both means this wing is live too."
        )
    if any(r.cp == "call" and r.delta < 0 for r in rows) and any(
        r.cp == "put" and r.delta > 0 for r in rows
    ):
        bits.append(
            "Pack both: short call + short put is a short strangle. Full reval, not a greek "
            "sum, is the mark after a jump because the smile is steep and the cap is close."
        )
    return bits
