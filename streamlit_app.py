"""Streamlit: call skew, ERCOT ORDC/scarcity, packed short strangle, full reval."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from call_skew_engine import (
    DAYS_PER_YEAR,
    EQUITY_SMILE,
    HCAP_POST,
    HCAP_PRE,
    POWER_SMILE,
    SHIFT,
    OptionLeg,
    SmileParams,
    commentary,
    default_pack,
    offer_cap,
    ordc_adder,
    pnl_vs_spot,
    revalue_book,
    risk_reversal,
    smile_frame,
)
from market_data import (
    HISTORY_START,
    HUB,
    SCARCITY_CUT,
    TIGHT_CUT,
    daily_from_hourly,
    event_windows,
    fetch_hb_north_rt,
)

st.set_page_config(
    page_title="Call skew — RR, butterfly, ERCOT scarcity",
    page_icon="📉",
    layout="wide",
)

COLOR_T = "#8a93a3"
COLOR_CALL = "#e07a3d"
COLOR_PUT = "#4c8bf5"
COLOR_PACK = "#c9a227"
COLOR_EQ = "#7ddea5"
COLOR_CAP = "#e05d5d"
COLOR_ORDC = "#c9842a"
COLOR_RT = "#d8dde6"
COLOR_NEG = "#5b8def"

PLOTLY_LEGEND_BELOW = dict(
    orientation="h",
    yanchor="top",
    y=-0.18,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(0,0,0,0)",
)


def _enable_plotly_mathjax() -> None:
    st.components.v1.html(
        """
        <script>
        (function () {
          var doc = window.parent.document;
          if (doc.getElementById("plotly-mathjax")) return;
          window.parent.MathJax = {
            tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
            svg: { fontCache: 'global' }
          };
          var s = doc.createElement('script');
          s.id = 'plotly-mathjax';
          s.async = true;
          s.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG';
          doc.head.appendChild(s);
        })();
        </script>
        """,
        height=0,
    )


_enable_plotly_mathjax()


def _default_pack_df(F: float) -> pd.DataFrame:
    """Default short strangle for the pack tab (editable in the table)."""
    legs = default_pack(float(F), call_pct=2.5, put_pct=0.35, days=30.0)
    return pd.DataFrame(
        [
            {
                "Label": lg.label,
                "Type": lg.cp,
                "Strike": lg.strike,
                "Days": lg.days_to_expiry,
                "Quantity": lg.quantity,
            }
            for lg in legs
        ]
    )


def _init_state() -> None:
    defaults = {
        "hourly": None,
        "status": None,
        "fetch_error": None,
        "applied": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _formulas_expander() -> None:
    with st.expander("Formulas", expanded=False):
        st.markdown(
            r"""
**Call skew.** Implied vol as a function of log-moneyness $k=\ln\bigl((K+s)/(F+s)\bigr)$
(shifted Black, $s=50$ so $F$ can print negative):

$$
\sigma(K)=\sigma_{\mathrm{ATM}}+\beta\,k+\chi\,k^{2}
$$

Equity puts have $\beta<0$ (crash skew). ERCOT power has $\beta>0$ (spike / cap skew):
OTM **calls** are rich because the right tail is ORDC through the offer cap.

**Risk reversal and butterfly** (wings at log-moneyness $\pm w$, default $w=0.25$):

$$
\mathrm{RR}=\sigma_{\mathrm{call}}-\sigma_{\mathrm{put}}
\qquad
\mathrm{Fly}=\tfrac12\bigl(\sigma_{\mathrm{call}}+\sigma_{\mathrm{put}}\bigr)-\sigma_{\mathrm{ATM}}
$$

RR is *directional skew* (sign flips with the crash side). Fly is *curvature* —
how rich both wings are versus ATM, independent of which wing wins.

**ORDC (stylized).** The operating-reserve demand curve adds a scarcity premium to
real-time LMP as online reserves $R$ collapse toward zero, up to VOLL / HCAP $C$:

$$
A(R)=C\cdot\bigl(0.65\,(1-R/X)_{+}^{1.4}+0.35\,(1-R/2X)_{+}^{1.1}\bigr)
$$

The observable in this study is RT LMP itself: hours with LMP ≥ 1,000 USD/MWh and prints at
HCAP are treated as ORDC fully engaged. HCAP was 9,000 USD/MWh through 31 May 2021
(Uri) and 5,000 USD/MWh after.

**Short call.** Selling the cap is selling ORDC/scarcity through $C$. On a spike,
delta, gamma, and vega all hurt the short:

$$
\mathrm{d}V\approx \Delta\,\mathrm{d}F+\tfrac12\Gamma\,(\mathrm{d}F)^{2}+\nu\,\mathrm{d}\sigma+\Theta\,\mathrm{d}t
$$

**Short put.** Crash / freeze on the demand side — negative RT, shoulder oversupply,
industrial freeze-off. Same greek identity, opposite spot jump.

**Pack both; full reval.** Short OTM call + short OTM put is a short strangle.
Mark-to-model is a full reval, not a greek sum, because the smile is steep and the
cap is close:

$$
\mathrm{P\&L}=q\cdot m\cdot\bigl[V(F_{1},K,T-\Delta t,\sigma_{1})-V(F_{0},K,T,\sigma_{0})\bigr]
$$

Stress the short call with a **price spike and a vol spike together**. That is the
compounding the first-order hedge misses.

Shifted Black-76 (rate = 0) on $F+s$, $K+s$. One MWh, one unit.
            """
        )


@st.cache_data(ttl=6 * 3600, show_spinner=True)
def _load_rt() -> tuple[pd.DataFrame, str]:
    return fetch_hb_north_rt()


def _ensure_rt() -> tuple[pd.DataFrame, str] | None:
    if st.session_state["hourly"] is None:
        try:
            hourly, status = _load_rt()
            st.session_state["hourly"] = hourly
            st.session_state["status"] = status
            st.session_state["fetch_error"] = None
        except Exception as exc:  # noqa: BLE001
            st.session_state["fetch_error"] = str(exc)
            return None
    return st.session_state["hourly"], st.session_state["status"]


def _smile_from_widgets(prefix: str) -> SmileParams:
    c1, c2, c3 = st.columns(3)
    atm = c1.number_input(
        "ATM vol",
        min_value=0.20,
        max_value=3.0,
        value=float(POWER_SMILE.sigma_atm),
        step=0.05,
        key=f"{prefix}_atm",
    )
    skew = c2.number_input(
        "Skew β (put < 0 < call)",
        min_value=-0.8,
        max_value=0.9,
        value=float(POWER_SMILE.skew),
        step=0.02,
        key=f"{prefix}_skew",
    )
    smile = c3.number_input(
        "Smile χ",
        min_value=0.0,
        max_value=1.0,
        value=float(POWER_SMILE.smile),
        step=0.02,
        key=f"{prefix}_smile",
    )
    return SmileParams(sigma_atm=float(atm), skew=float(skew), smile=float(smile), shift=SHIFT)


def _fig_smile(
    F: float,
    p_power: SmileParams,
    p_eq: SmileParams,
    rr: dict[str, float] | None = None,
) -> go.Figure:
    pw = smile_frame(F, p_power)
    eq = smile_frame(F, p_eq)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pw["K"],
            y=100.0 * pw["iv"],
            name="ERCOT power (call skew)",
            line=dict(color=COLOR_CALL, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eq["K"],
            y=100.0 * eq["iv"],
            name="Equity-style put skew",
            line=dict(color=COLOR_EQ, width=2.5, dash="dash"),
        )
    )
    fig.add_vline(x=F, line=dict(color=COLOR_RT, width=1, dash="dot"))
    fig.add_annotation(
        x=F,
        y=100.0 * float(p_power.sigma_atm),
        text="F (ATM)",
        showarrow=False,
        yshift=18,
        font=dict(color=COLOR_RT),
    )
    if rr is not None:
        fig.add_trace(
            go.Scatter(
                x=[rr["K_put"], rr["K_call"]],
                y=[100.0 * rr["sig_put"], 100.0 * rr["sig_call"]],
                mode="markers+text",
                name="RR wings (±w)",
                marker=dict(size=11, color=COLOR_PACK, symbol="diamond"),
                text=["Put wing", "Call wing"],
                textposition="top center",
                textfont=dict(size=11, color=COLOR_PACK),
            )
        )
        # ATM marker for fly reading.
        fig.add_trace(
            go.Scatter(
                x=[F],
                y=[100.0 * rr["sig_atm"]],
                mode="markers",
                name="ATM (fly belly)",
                marker=dict(size=10, color=COLOR_RT, symbol="circle"),
                showlegend=True,
            )
        )
    fig.update_layout(
        height=480,
        margin=dict(t=56, b=110, l=56, r=16),
        title=dict(
            text="Same ATM, opposite wings — power calls are the crash puts of ERCOT",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title="Strike K (USD/MWh)",
        yaxis_title="Implied vol (%)",
        legend=PLOTLY_LEGEND_BELOW,
        hovermode="x unified",
    )
    return fig


def _fig_rr_fly_compare(power_rr: dict[str, float], eq_rr: dict[str, float]) -> go.Figure:
    """Side-by-side RR and fly for power vs equity — separate figure, own legend."""
    labels = ["Risk reversal<br>(call − put)", "Butterfly<br>(½ wings − ATM)"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[100.0 * power_rr["rr"], 100.0 * power_rr["fly"]],
            name="ERCOT power",
            marker_color=COLOR_CALL,
            text=[f"{100 * power_rr['rr']:+.1f} pp", f"{100 * power_rr['fly']:+.1f} pp"],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[100.0 * eq_rr["rr"], 100.0 * eq_rr["fly"]],
            name="Equity-style",
            marker_color=COLOR_EQ,
            text=[f"{100 * eq_rr['rr']:+.1f} pp", f"{100 * eq_rr['fly']:+.1f} pp"],
            textposition="outside",
        )
    )
    fig.add_hline(y=0.0, line=dict(color=COLOR_T, width=1, dash="dot"))
    fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(t=56, b=110, l=56, r=16),
        title=dict(
            text="RR flips sign with the crash side; fly stays positive when both wings are rich",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        yaxis_title="Vol points (pp)",
        legend=PLOTLY_LEGEND_BELOW,
    )
    return fig


def _fig_ordc(voll: float) -> go.Figure:
    r = np.linspace(0.0, 6_000.0, 241)
    a = ordc_adder(r, voll=voll, x_mw=2_000.0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r,
            y=a,
            name="Stylized ORDC adder",
            line=dict(color=COLOR_ORDC, width=3),
            fill="tozeroy",
            fillcolor="rgba(201,132,42,0.18)",
        )
    )
    fig.add_hline(y=voll, line=dict(color=COLOR_CAP, width=1.4, dash="dash"))
    fig.add_annotation(
        x=4_800,
        y=voll,
        text=f"HCAP / VOLL {voll:,.0f}",
        showarrow=False,
        yshift=14,
        font=dict(color=COLOR_CAP, size=12),
    )
    fig.update_layout(
        height=400,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="ORDC: reserves fall, the adder walks the call wing to the cap",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title="Online reserves R (MW)",
        yaxis_title="Scarcity adder (USD/MWh)",
        legend=PLOTLY_LEGEND_BELOW,
    )
    return fig


def _fig_history(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["high"],
            name="Daily high RT",
            line=dict(color=COLOR_CALL, width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["mean"],
            name="Daily mean RT",
            line=dict(color=COLOR_RT, width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["low"],
            name="Daily low RT",
            line=dict(color=COLOR_PUT, width=1.2),
        )
    )
    fig.add_hline(y=HCAP_POST, line=dict(color=COLOR_CAP, width=1, dash="dash"))
    fig.add_hline(y=0.0, line=dict(color=COLOR_PUT, width=1, dash="dot"))
    fig.add_annotation(
        x=daily["date"].iloc[len(daily) // 2],
        y=HCAP_POST,
        text="HCAP $5,000 (from Jun 2021)",
        showarrow=False,
        yshift=12,
        font=dict(color=COLOR_CAP, size=11),
    )
    fig.update_layout(
        height=480,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text=f"{HUB} real-time LMP — daily high / mean / low",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title=None,
        yaxis_title="USD/MWh",
        legend=PLOTLY_LEGEND_BELOW,
        hovermode="x unified",
    )
    return fig


def _fig_histogram(hourly: pd.DataFrame) -> go.Figure:
    px = hourly["price"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=px,
            name="Hourly RT LMP",
            marker_color=COLOR_T,
            nbinsx=80,
        )
    )
    fig.add_vline(x=0.0, line=dict(color=COLOR_PUT, width=1.4, dash="dot"))
    fig.add_vline(x=SCARCITY_CUT, line=dict(color=COLOR_CALL, width=1.4, dash="dash"))
    fig.add_vline(x=HCAP_POST, line=dict(color=COLOR_CAP, width=1.4, dash="dash"))
    fig.update_layout(
        height=400,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="Left tail is the short put; right tail is ORDC through the cap",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title="Hourly RT LMP (USD/MWh)",
        yaxis_title="Hours",
        yaxis_type="log",
        legend=PLOTLY_LEGEND_BELOW,
    )
    return fig


def _fig_regime_hours(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["n_scarcity"],
            name="Hours ≥ $1,000 (short call / ORDC)",
            marker_color=COLOR_CALL,
        )
    )
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=-daily["n_neg"],
            name="Hours < $0 (short put / crash)",
            marker_color=COLOR_NEG,
        )
    )
    fig.update_layout(
        barmode="relative",
        height=400,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="Scarcity hours vs negative hours — both wings of the pack",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title=None,
        yaxis_title="Hours (negative = crash)",
        legend=PLOTLY_LEGEND_BELOW,
    )
    return fig


def _fig_pack_pnl(path: pd.DataFrame, F0: float, k_call: float, k_put: float) -> go.Figure:
    fig = go.Figure()
    labels = [c for c in path.columns if c not in {"F1", "total", "delta", "gamma", "vega"}]
    colors = {0: COLOR_CALL, 1: COLOR_PUT}
    for i, lab in enumerate(labels):
        fig.add_trace(
            go.Scatter(
                x=path["F1"],
                y=path[lab],
                name=lab,
                line=dict(color=colors.get(i, COLOR_T), width=2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=path["F1"],
            y=path["total"],
            name="Pack (full reval)",
            line=dict(color=COLOR_PACK, width=3),
        )
    )
    fig.add_vline(x=F0, line=dict(color=COLOR_RT, width=1, dash="dot"))
    fig.add_vline(x=k_call, line=dict(color=COLOR_CALL, width=1, dash="dash"))
    fig.add_vline(x=k_put, line=dict(color=COLOR_PUT, width=1, dash="dash"))
    fig.add_hline(y=0.0, line=dict(color=COLOR_T, width=1, dash="dot"))
    fig.update_layout(
        height=460,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="Full reval P&L vs new spot — pack both wings",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title="Fₜ₊₁ (USD/MWh)",
        yaxis_title="P&L (USD per MWh-unit)",
        legend=PLOTLY_LEGEND_BELOW,
        hovermode="x unified",
    )
    return fig


def _fig_greeks(path: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path["F1"], y=path["delta"], name="Delta P&L", line=dict(color=COLOR_T, width=2)))
    fig.add_trace(go.Scatter(x=path["F1"], y=path["gamma"], name="Gamma P&L", line=dict(color=COLOR_CALL, width=2)))
    fig.add_trace(go.Scatter(x=path["F1"], y=path["vega"], name="Vega P&L", line=dict(color=COLOR_ORDC, width=2)))
    fig.add_trace(
        go.Scatter(x=path["F1"], y=path["total"], name="Full reval", line=dict(color=COLOR_PACK, width=3))
    )
    fig.add_hline(y=0.0, line=dict(color=COLOR_T, width=1, dash="dot"))
    fig.update_layout(
        height=420,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="Greek decomposition vs full reval — gamma and vega compound on the spike",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title="Fₜ₊₁ (USD/MWh)",
        yaxis_title="P&L (USD per MWh-unit)",
        legend=PLOTLY_LEGEND_BELOW,
        hovermode="x unified",
    )
    return fig


def _fig_stress_bars(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=table["Label"], y=table["P&L (full reval)"], name="Full reval", marker_color=COLOR_PACK)
    )
    fig.add_trace(go.Bar(x=table["Label"], y=table["Delta P&L"], name="Delta", marker_color=COLOR_T))
    fig.add_trace(go.Bar(x=table["Label"], y=table["Gamma P&L"], name="Gamma", marker_color=COLOR_CALL))
    fig.add_trace(go.Bar(x=table["Label"], y=table["Vega P&L"], name="Vega", marker_color=COLOR_ORDC))
    fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="Stress the short call: price spike + vol spike",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        yaxis_title="P&L",
        legend=PLOTLY_LEGEND_BELOW,
    )
    return fig


# —— Header ——
st.markdown(
    "<h1 style='text-align: center;'>Call skew</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
- Equity smiles pay you to sell calls. **Power smiles do not.**
  - ERCOT's right tail is ORDC through the offer cap — that is call skew.
- **Short call** — you sold scarcity.
  - When reserves collapse, the adder walks LMP to HCAP.
  - Gamma and vega **compound** with the spot spike.
- **Short put** — crash / freeze on the demand side.
  - Negative RT, shoulder oversupply, industrial freeze-off.
  - **Pack both**; full reval; stress call with vol **and** price.
    """
)
_formulas_expander()

loaded = _ensure_rt()
if loaded is None:
    st.error(f"Could not load ERCOT RT LMP: {st.session_state['fetch_error']}")
    st.stop()

hourly, status = loaded
hourly = hourly.copy()
hourly["ts"] = pd.to_datetime(hourly["ts"])
hourly = hourly.dropna(subset=["price"]).sort_values("ts")
daily = daily_from_hourly(hourly)
last_px = float(hourly["price"].iloc[-1])
last_ts = pd.Timestamp(hourly["ts"].iloc[-1])
F0_live = float(daily["mean"].iloc[-1]) if len(daily) else last_px
if not np.isfinite(F0_live) or abs(F0_live) < 0.5:
    F0_live = 40.0

feed_note = {
    "live": "ERCOT MIS (NP6-785-ER yearly RT hub/zone + NP6-905-CD recent 15-min SPP)",
    "cache": "local parquet cache of a prior ERCOT download",
    "fallback": "deterministic demo path (ERCOT feed blocked) — Uri, summer scarcity, negative dumps",
}.get(status, status)

st.caption(
    f"{HUB} RT LMP from {HISTORY_START.date()} through {last_ts.date()}  ·  "
    f"last hour {last_px:,.2f} USD/MWh  ·  feed: {feed_note}"
)

tab_skew, tab_rt, tab_pack, tab_stress = st.tabs(
    [
        "1. Theory",
        "2. ERCOT RT LMP — scarcity vs crash",
        "3. Pack both · full reval",
        "4. Stress the short call",
    ]
)

# ─────────────────────────────────────────────
# Tab 1 — call skew / RR / fly
# ─────────────────────────────────────────────
with tab_skew:
    st.subheader("Why ERCOT options are call-skewed")
    st.markdown(
        """
**Equity vs ERCOT**

- In equities the crash is down: OTM puts are rich, β < 0.
- In ERCOT the crash is **up**:
  - Generation trips; reserves fall.
  - ORDC adds scarcity dollars until LMP sits on the offer cap.
  - OTM **calls** (caps) are the expensive wing.
- Selling that wing is selling ORDC through HCAP.
        """
    )
    F0 = st.number_input(
        "Forward Fₜ (USD/MWh)",
        min_value=1.0,
        max_value=500.0,
        value=float(round(max(F0_live, 5.0), 2)),
        step=1.0,
        help="Auto-filled from the latest daily-mean HB_NORTH RT print.",
    )
    wing_w = st.slider(
        "Wing log-moneyness w (RR / fly strikes at ±w)",
        min_value=0.10,
        max_value=0.50,
        value=0.25,
        step=0.05,
        help="Larger w = deeper OTM wings. Classic 25Δ-ish quote uses ~0.25.",
    )
    with st.expander("Edit power smile", expanded=False):
        p = _smile_from_widgets("skew")
    rr = risk_reversal(F0, p, wing=float(wing_w))
    eq_rr = risk_reversal(F0, EQUITY_SMILE, wing=float(wing_w))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ATM vol", f"{100 * p.sigma_atm:.0f}%")
    m2.metric("Call-wing vol", f"{100 * rr['sig_call']:.0f}%")
    m3.metric("Put-wing vol", f"{100 * rr['sig_put']:.0f}%")
    m4.metric("Risk reversal", f"{100 * rr['rr']:+.1f} pp", help="σ_call − σ_put. Positive = call skew.")
    m5.metric("Butterfly", f"{100 * rr['fly']:+.1f} pp", help="½(σ_call + σ_put) − σ_ATM. Curvature / fat tails.")

    st.markdown(
        f"""
- Equity-style at the same F: RR {100 * eq_rr['rr']:+.1f} pp, fly {100 * eq_rr['fly']:+.1f} pp.
  - The RR **sign flip** is the product.
  - Fly stays positive when both wings are rich versus ATM.
        """
    )
    st.plotly_chart(_fig_smile(F0, p, EQUITY_SMILE, rr=rr), use_container_width=True)

    st.markdown("##### Risk reversal vs butterfly")
    st.markdown(
        """
- **Risk reversal (RR)** = call-wing IV − put-wing IV — *which* crash the market prices.
  - Equity RR is usually **negative** (puts rich).
  - ERCOT power RR is **positive** (calls / caps rich).
  - Same ATM can hide opposite tails.
- **Butterfly (fly)** = average of the two wing IVs minus ATM — curvature, not direction.
  - High fly with a small |RR|: both wings expensive without a one-sided skew story.
  - Large |RR| with modest fly: almost all the premium sits in one wing.
- In energy:
  - A summer scarcity scare lifts **RR** (and often fly).
  - A shoulder oversupply dump can compress RR toward zero while **fly** stays elevated.
        """
    )
    st.plotly_chart(_fig_rr_fly_compare(rr, eq_rr), use_container_width=True)

    cap_now = offer_cap(last_ts)
    st.markdown("##### ORDC is the call wing")
    st.markdown(
        f"""
- The operating-reserve demand curve is not a separate product you can unsee — it **is**
  the right tail of RT LMP.
- As *R* falls, the adder *A(R)* walks toward HCAP ({cap_now:,.0f} USD/MWh today).
- A short call struck through that region is:
  - Short the adder.
  - Short the cap.
  - Short the vol that prints when both jump.
- That is why power RR prints **positive**.
        """
    )
    st.plotly_chart(_fig_ordc(cap_now), use_container_width=True)

    st.info(
        """
- **Short call** — ORDC / scarcity through the cap; positive RR.
  - Gamma and vega **compound** on a spike.
- **Short put** — crash / freeze on the demand side.
- **Pack both**; full reval.
  - Stress the short call with a vol spike **and** a price spike together.
        """
    )

# ─────────────────────────────────────────────
# Tab 2 — RT history
# ─────────────────────────────────────────────
with tab_rt:
    st.subheader("Five years of real-time LMP")
    st.markdown(
        f"""
**Data**

- **{HUB}** real-time settlement point price.
  - 15-minute SCED LMP, aggregated to hours.
  - Historical: ERCOT RTM hub/zone files
    ([NP6-785-ER](https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER)).
  - Recent gap-fill:
    ([NP6-905-CD](https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD)).

**What the tails mean**

- **Right tail** — short-call / ORDC story.
  - Uri at HCAP 9,000 USD/MWh.
  - Later scarcity at 5,000 USD/MWh.
- **Left tail** — short-put story.
  - Negative RT.
  - Shoulder oversupply, demand freeze-off.
        """
    )
    if st.button("Refresh RT LMP", type="secondary"):
        _load_rt.clear()
        st.session_state["hourly"] = None
        st.rerun()

    n_hours = len(hourly)
    n_neg = int((hourly["price"] < 0).sum())
    n_tight = int((hourly["price"] >= TIGHT_CUT).sum())
    n_scarce = int((hourly["price"] >= SCARCITY_CUT).sum())
    n_cap = int((hourly["price"] >= 0.90 * HCAP_POST).sum())
    n_uri_cap = int((hourly["price"] >= 0.90 * HCAP_PRE).sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Hours in sample", f"{n_hours:,}")
    k2.metric("Hours < $0 (put)", f"{n_neg:,}", help="Crash / oversupply / demand-side freeze-off.")
    k3.metric("Hours ≥ $250 (tight)", f"{n_tight:,}")
    k4.metric("Hours ≥ $1,000 (ORDC)", f"{n_scarce:,}", help="Scarcity prints — short-call territory.")
    k5.metric("Hours near HCAP", f"{n_cap + n_uri_cap:,}", help="≥ 90% of $5,000 or $9,000.")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Min RT", f"{hourly['price'].min():,.1f}")
    p2.metric("Median RT", f"{hourly['price'].median():,.1f}")
    p3.metric("P99 RT", f"{hourly['price'].quantile(0.99):,.0f}")
    p4.metric("Max RT", f"{hourly['price'].max():,.0f}")

    st.plotly_chart(_fig_history(daily), use_container_width=True)
    st.plotly_chart(_fig_regime_hours(daily), use_container_width=True)
    st.plotly_chart(_fig_histogram(hourly), use_container_width=True)

    st.markdown("##### Named windows — call side vs put side")
    st.caption("Read-only table — named scarcity and crash windows in the HB_NORTH sample.")
    ev = event_windows(hourly)
    if ev.empty:
        st.caption("No named windows overlapped this sample.")
    else:
        st.dataframe(
            ev.style.format(
                {
                    "Max RT": "{:,.1f}",
                    "Min RT": "{:,.1f}",
                    "Mean RT": "{:,.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown(
        """
- Uri is a **supply** freeze — generation trips, ORDC to HCAP.
  - Sits on the short **call**.
- A demand-side freeze-off or shoulder dump sits on the short **put**.
        """
    )

# ─────────────────────────────────────────────
# Tab 3 — pack both
# ─────────────────────────────────────────────
with tab_pack:
    st.subheader("Pack both wings, then full-reval")
    st.markdown(
        """
**What this tab does**

- Builds a **short strangle** — short OTM call (ORDC/cap) + short OTM put (crash/floor).
  - Set strikes, days, and quantity in the **editable table** (use **Reset to default pack** to reload defaults for Fₜ).
- Applies a **joint spot + vol shock** on the call-skewed smile.
  - Scarcity preset: rally toward HCAP with a vol spike (short-call path).
  - Crash preset: negative RT with a smaller vol bump (short-put path).
- Marks the book with a **full reval** — reprices every leg on the smile at Fₜ₊₁, not a greek sum.

**What you'll learn**

- **Delta alone is not the P&L** after a jump — gamma and vega on the call wing compound on the scarcity path.
- **Both wings are live** — the put collects premium on a rally, but it does not hedge ORDC; the call is still the event.
- **The smile matters** — same spot move marks differently under call skew than under a flat or equity-style smile.
- **P&L vs spot** — the full-reval curve shows where the pack bleeds (through the call strike, toward HCAP, into negative RT).
        """
    )
    F_pack = st.number_input(
        "Spot Fₜ (USD/MWh)",
        min_value=1.0,
        max_value=500.0,
        value=float(round(max(F0_live, 5.0), 2)),
        step=1.0,
        key="pack_F0",
    )
    with st.expander("Edit power smile", expanded=False):
        p_pack = _smile_from_widgets("pack")

    reset_col, _ = st.columns([1, 3])
    with reset_col:
        if st.button("Reset to default pack", key="pack_reset", help="Reload default short strangle for current Fₜ."):
            st.session_state.pop("pack_editor", None)
            st.session_state["pack_book_seed"] = _default_pack_df(F_pack)
            st.rerun()

    if "pack_book_seed" not in st.session_state:
        st.session_state["pack_book_seed"] = _default_pack_df(F_pack)

    book_df = st.session_state["pack_book_seed"]
    st.markdown("##### Packed book (short strangle)")
    st.caption(
        "**Editable** — set strike, days, and quantity here (default: short call ~2.5× F, "
        "short put ~0.35× F, 30 DTE). Use **+** to add legs or delete a row. "
        "Press **Submit full reval** after edits."
    )
    edited = st.data_editor(
        book_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Label": st.column_config.TextColumn(required=True),
            "Type": st.column_config.SelectboxColumn(options=["call", "put"], required=True),
            "Strike": st.column_config.NumberColumn(min_value=-50.0, format="%.2f"),
            "Days": st.column_config.NumberColumn(min_value=1, max_value=730, step=1),
            "Quantity": st.column_config.NumberColumn(step=1.0, format="%.2f"),
        },
        key="pack_editor",
    )

    shock_mode = st.radio(
        "Spot shock",
        options=["scarcity", "crash", "custom"],
        format_func=lambda x: {
            "scarcity": "Scarcity / ORDC (short-call path)",
            "crash": "Crash / freeze demand (short-put path)",
            "custom": "Custom Fₜ₊₁",
        }[x],
        horizontal=True,
    )
    if shock_mode == "scarcity":
        F1 = float(min(HCAP_POST, max(F_pack * 40.0, 2_000.0)))
        vol_bump = 0.80
        st.caption(f"Preset: F → {F1:,.0f} (toward HCAP) and ATM-style vol bump +80 pp.")
    elif shock_mode == "crash":
        F1 = float(min(-15.0, -0.4 * F_pack))
        vol_bump = 0.40
        st.caption(f"Preset: F → {F1:,.1f} (negative RT) and vol bump +40 pp.")
    else:
        cA, cB = st.columns(2)
        F1 = float(
            cA.number_input(
                "Fₜ₊₁ (USD/MWh)",
                min_value=-50.0,
                max_value=9_000.0,
                value=float(round(min(HCAP_POST, F_pack * 5.0), 1)),
                step=10.0,
            )
        )
        vol_bump = (
            cB.number_input("Vol bump (pp)", min_value=0.0, max_value=250.0, value=80.0, step=5.0) / 100.0
        )

    theta_col, submit_col, _ = st.columns([1, 1, 2])
    include_theta = theta_col.checkbox(
        "One day of theta",
        value=True,
        key="pack_theta",
        help=(
            "When on, calendar time advances one day — so time *to expiry* shortens "
            "(T → T − 1 day) and P&L includes time decay. When off, the shock is "
            "instantaneous at the same expiry; only spot and vol move."
        ),
    )
    submitted = submit_col.button("Submit full reval", type="primary")
    if submitted:
        try:
            legs_ed = []
            for _, row in edited.iterrows():
                legs_ed.append(
                    OptionLeg(
                        cp=str(row["Type"]),
                        strike=float(row["Strike"]),
                        days_to_expiry=float(row["Days"]),
                        quantity=float(row["Quantity"]),
                        multiplier=1.0,
                        label=str(row["Label"]),
                    )
                )
            if not legs_ed:
                raise ValueError("Add at least one option leg.")
            dt = (1.0 / DAYS_PER_YEAR) if include_theta else 0.0
            rows, table = revalue_book(legs_ed, F_pack, F1, p_pack, sigma_bump=vol_bump, dt_years=dt)
            st.session_state["applied"] = {
                "rows": rows,
                "table": table,
                "legs": legs_ed,
                "F0": F_pack,
                "F1": F1,
                "p": p_pack,
                "vol_bump": vol_bump,
                "dt": dt,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    applied = st.session_state.get("applied")
    if applied is None:
        st.info(
            """
- Set the shock and edit the pack if you like.
- Press **Submit full reval**.
            """
        )
    else:
        table = applied["table"]
        rows = applied["rows"]
        total = float(table["P&L (full reval)"].sum())
        greeks = {
            "delta": float(table["Delta P&L"].sum()),
            "gamma": float(table["Gamma P&L"].sum()),
            "vega": float(table["Vega P&L"].sum()),
        }
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fₜ → Fₜ₊₁", f"{applied['F0']:.1f} → {applied['F1']:.1f}")
        r2.metric("P&L full reval", f"{total:,.0f}")
        r3.metric("Gamma + vega", f"{greeks['gamma'] + greeks['vega']:,.0f}")
        r4.metric("Delta only (misses the rest)", f"{greeks['delta']:,.0f}")

        st.subheader("Commentary")
        for bullet in commentary(rows, applied["F0"], applied["F1"], applied["vol_bump"]):
            st.markdown(f"- {bullet}")

        spots = np.linspace(-30.0, min(HCAP_POST, max(applied["F1"] * 1.15, 1_200.0)), 81)
        path = pnl_vs_spot(
            applied["legs"],
            applied["F0"],
            applied["p"],
            spots,
            sigma_bump=applied["vol_bump"],
            dt_years=applied["dt"],
        )
        k_c = next((lg.strike for lg in applied["legs"] if lg.cp == "call"), 100.0)
        k_p = next((lg.strike for lg in applied["legs"] if lg.cp == "put"), 10.0)
        st.plotly_chart(_fig_pack_pnl(path, applied["F0"], k_c, k_p), use_container_width=True)
        st.plotly_chart(_fig_greeks(path), use_container_width=True)

        st.markdown("##### Leg detail")
        st.caption("Read-only — full-reval output for each leg after **Submit full reval**.")
        st.dataframe(
            table.style.format(
                {
                    "Strike": "{:,.2f}",
                    "F0": "{:,.2f}",
                    "F1": "{:,.2f}",
                    "IV t": "{:.1%}",
                    "IV t+1": "{:.1%}",
                    "Value t": "{:,.2f}",
                    "Value t+1": "{:,.2f}",
                    "P&L (full reval)": "{:,.2f}",
                    "Delta P&L": "{:,.2f}",
                    "Gamma P&L": "{:,.2f}",
                    "Vega P&L": "{:,.2f}",
                    "Theta P&L": "{:,.2f}",
                    "Unexplained": "{:,.2f}",
                    "Delta": "{:.3f}",
                    "Gamma": "{:.4f}",
                    "Vega": "{:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────────────────────────
# Tab 4 — stress short call
# ─────────────────────────────────────────────
with tab_stress:
    st.subheader("Stress the short call with vol + price")
    st.markdown(
        """
**What this tab does**

- Holds the **same packed short strangle** as tab 3, but zooms in on the **short-call / ORDC** stress.
  - Default: short call ~2× F, short put ~0.35× F (fixed book — adjust Fₜ and days to expiry below).
- Applies a **joint price + vol spike** to the short call — the scarcity path.
  - Drag **Stressed Fₜ₊₁** through the call strike (ORDC onset) or up to **HCAP**.
  - Lift wing IV with the vol-spike slider at the same time.
- **Decomposes** the short-call P&L into price-only, vol-only, and **interaction** (full reval minus the sum of one-risk stresses).
  - Waterfall and P&L-vs-spot charts update live as you move the slider.

**What you'll learn**

- **Do not add the stresses** — price spike + vol spike together mark worse than price-only + vol-only; the gap is **compounding** (gamma × vega on a jump).
- **Through the strike** — vega rises with spot; interaction is largest where ORDC turns on.
- **Through HCAP** — the call is intrinsic; vega **dies** and the vol bump barely moves the mark.
- **Delta alone is not P&L** — compare the interaction bar to delta-only in the packed-book table below.
- **Put premium is not a hedge** — on a rally the put helps a little, but the call leg still drives the loss.
        """
    )
    F_s = st.number_input(
        "Spot Fₜ (USD/MWh)",
        min_value=1.0,
        max_value=500.0,
        value=float(round(max(F0_live, 5.0), 2)),
        step=1.0,
        key="stress_F0",
    )
    with st.expander("Edit power smile", expanded=False):
        p_s = _smile_from_widgets("stress")

    s2, s3 = st.columns(2)
    vol_pp = s2.slider("Vol spike (pp added to IV)", min_value=0.0, max_value=200.0, value=90.0, step=5.0)
    days_s = s3.number_input("Days to expiry", min_value=1, max_value=365, value=21, step=1, key="stress_days")

    legs_s = default_pack(F_s, call_pct=2.0, put_pct=0.35, days=float(days_s))
    k_call_s = legs_s[0].strike
    onset = float(min(HCAP_POST, round(k_call_s * 1.25, 0)))
    if "stress_F1" not in st.session_state:
        st.session_state["stress_F1"] = onset
    pA, pB, pC = st.columns([1, 1, 2])
    if pA.button("Preset: through the strike", use_container_width=True):
        st.session_state["stress_F1"] = onset
        st.rerun()
    if pB.button("Preset: through HCAP", use_container_width=True):
        st.session_state["stress_F1"] = float(HCAP_POST)
        st.rerun()
    F1_s = pC.slider(
        "Stressed Fₜ₊₁ (USD/MWh)",
        min_value=0.0,
        max_value=float(HCAP_POST),
        step=25.0,
        key="stress_F1",
        help=f"Short-call strike is {k_call_s:g}. Through the strike, vega rises; through HCAP, vega dies.",
    )
    st.markdown(
        f"""
- Short call K = {k_call_s:g} USD/MWh.
  - Near that strike, a vol spike **adds** to gamma.
  - At HCAP the option is intrinsic; the vol bump barely matters.
        """
    )
    dt_s = 1.0 / DAYS_PER_YEAR
    vol_bump_s = vol_pp / 100.0

    rows_s, table_s = revalue_book(legs_s, F_s, F1_s, p_s, sigma_bump=vol_bump_s, dt_years=dt_s)
    call_row = next(r for r in rows_s if r.cp == "call")
    put_row = next(r for r in rows_s if r.cp == "put")

    # Call-only stresses for the waterfall.
    only_px, _ = revalue_book([legs_s[0]], F_s, F1_s, p_s, sigma_bump=0.0, dt_years=dt_s)
    only_vol, _ = revalue_book([legs_s[0]], F_s, F_s, p_s, sigma_bump=vol_bump_s, dt_years=dt_s)
    both, _ = revalue_book([legs_s[0]], F_s, F1_s, p_s, sigma_bump=vol_bump_s, dt_years=dt_s)
    px_pnl = only_px[0].pnl
    vol_pnl = only_vol[0].pnl
    both_pnl = both[0].pnl
    interaction = both_pnl - px_pnl - vol_pnl

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Short-call full reval", f"{call_row.pnl:,.0f}")
    c2.metric("Price spike only", f"{px_pnl:,.0f}")
    c3.metric("Vol spike only", f"{vol_pnl:,.0f}")
    c4.metric(
        "Interaction (compounding)",
        f"{interaction:,.0f}",
        help="Full (price+vol) minus price-only minus vol-only. Gamma × vega on a jump is not additive.",
    )

    st.markdown(
        f"""
- Price-only + vol-only = **{px_pnl + vol_pnl:,.0f}**.
- Together they mark **{both_pnl:,.0f}**.
- The extra **{interaction:,.0f}** is the compound:
  - Call is further in-the-money.
  - Wing vol has jumped — vega hits a fatter option.
  - That is the ORDC stress.
        """
    )

    st.plotly_chart(_fig_stress_bars(table_s), use_container_width=True)

    water = pd.DataFrame(
        {
            "Step": [
                "Price spike only",
                "Vol spike only",
                "Sum of one-risk stresses",
                "Price + vol together",
                "Interaction",
                "Packed book (call + put)",
            ],
            "P&L": [
                px_pnl,
                vol_pnl,
                px_pnl + vol_pnl,
                both_pnl,
                interaction,
                float(table_s["P&L (full reval)"].sum()),
            ],
        }
    )
    fig_w = go.Figure(
        go.Bar(
            x=water["Step"],
            y=water["P&L"],
            marker_color=[COLOR_CALL, COLOR_ORDC, COLOR_T, COLOR_PACK, COLOR_CAP, COLOR_PUT],
            text=[f"{v:,.0f}" for v in water["P&L"]],
            textposition="outside",
        )
    )
    fig_w.update_layout(
        height=420,
        margin=dict(t=56, b=120, l=56, r=16),
        title=dict(
            text="Do not add the stresses — revalue the short call under both at once",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        yaxis_title="P&L",
        showlegend=False,
    )
    st.plotly_chart(fig_w, use_container_width=True)

    spots_s = np.linspace(max(F_s * 0.2, 1.0), HCAP_POST, 61)
    path_flat = pnl_vs_spot([legs_s[0]], F_s, p_s, spots_s, sigma_bump=0.0, dt_years=dt_s)
    path_vol = pnl_vs_spot([legs_s[0]], F_s, p_s, spots_s, sigma_bump=vol_bump_s, dt_years=dt_s)
    fig_s = go.Figure()
    fig_s.add_trace(
        go.Scatter(
            x=path_flat["F1"],
            y=path_flat["total"],
            name="Short call — price only",
            line=dict(color=COLOR_T, width=2.5),
        )
    )
    fig_s.add_trace(
        go.Scatter(
            x=path_vol["F1"],
            y=path_vol["total"],
            name="Short call — price + vol spike",
            line=dict(color=COLOR_CALL, width=3),
        )
    )
    fig_s.add_vline(x=F_s, line=dict(color=COLOR_RT, width=1, dash="dot"))
    fig_s.add_vline(x=F1_s, line=dict(color=COLOR_CAP, width=1.4, dash="dash"))
    fig_s.add_hline(y=0.0, line=dict(color=COLOR_T, width=1, dash="dot"))
    fig_s.update_layout(
        height=440,
        margin=dict(t=56, b=96, l=56, r=16),
        title=dict(
            text="Short-call P&L along the ORDC path — vol spike opens a second hole",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        xaxis_title="Fₜ₊₁ (USD/MWh)",
        yaxis_title="P&L",
        legend=PLOTLY_LEGEND_BELOW,
        hovermode="x unified",
    )
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("##### Packed book under this stress")
    st.caption("Read-only — leg-level P&L and greeks for the preset stress (slider above).")
    st.markdown(
        f"""
- Put strike {put_row.strike:g} barely moves on a rally to {F1_s:,.0f}.
- Call at {call_row.strike:g} is the event.
- Packing both still requires this call stress — put premium is not a hedge for ORDC.
        """
    )
    st.dataframe(
        table_s.style.format(
            {
                "Strike": "{:,.2f}",
                "F0": "{:,.2f}",
                "F1": "{:,.2f}",
                "IV t": "{:.1%}",
                "IV t+1": "{:.1%}",
                "Value t": "{:,.2f}",
                "Value t+1": "{:,.2f}",
                "P&L (full reval)": "{:,.2f}",
                "Delta P&L": "{:,.2f}",
                "Gamma P&L": "{:,.2f}",
                "Vega P&L": "{:,.2f}",
                "Theta P&L": "{:,.2f}",
                "Unexplained": "{:,.2f}",
                "Delta": "{:.3f}",
                "Gamma": "{:.4f}",
                "Vega": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
