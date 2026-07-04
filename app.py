"""
SpaceX (SPCX) Interactive DCF Valuation — Streamlit App
Deploy: push app.py + spacex_dcf.py + requirements.txt to GitHub,
then deploy on share.streamlit.io with main file = app.py
"""

import copy
import numpy as np
import pandas as pd
import streamlit as st

from spacex_dcf import (
    ASSUMPTIONS, fetch_data, build_dcf,
    generate_sensitivity_table, solve_reverse_dcf,
)

st.set_page_config(page_title="SpaceX DCF | Reverse Valuation",
                   page_icon="🚀", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.stMetric label { color: #999 !important; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', monospace; }
.accent { color: #E8590C; }
</style>
""", unsafe_allow_html=True)

st.title("SpaceX (NASDAQ: SPCX) — What Must You Believe at $162?")
st.caption("An interactive reverse-DCF valuation · live data via yfinance, "
           "S-1 fallbacks · not investment advice")

# ------------------------------------------------------------------ data
@st.cache_data(ttl=3600, show_spinner="Pulling live SPCX data…")
def load_data():
    return fetch_data(ASSUMPTIONS)

data, source = load_data()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Assumptions")

cfg = copy.deepcopy(ASSUMPTIONS)

cfg["forecast_years"] = st.sidebar.slider("Forecast horizon (years)", 5, 15, 10)
cfg["wacc"] = st.sidebar.slider("Discount rate / WACC (%)", 6.0, 15.0, 10.0, 0.25) / 100
cfg["terminal_growth"] = st.sidebar.slider("Terminal growth (%)", 1.0, 5.0, 3.0, 0.25) / 100

st.sidebar.subheader("Revenue growth (fades to terminal)")
g1 = st.sidebar.slider("Year 1 growth (%)", 0, 100, 50, 5) / 100
g5 = st.sidebar.slider("Year 5 growth (%)", 0, 100, 30, 5) / 100
cfg["revenue_growth"] = list(np.linspace(g1, g5, 5))

st.sidebar.subheader("Profitability & spend")
m5 = st.sidebar.slider("Steady-state EBIT margin (%)", 0, 45, 25, 1) / 100
cfg["operating_margin"] = [0.02, 0.08, 0.14, 0.20, m5]
cfg["steady_state"]["operating_margin"] = m5
capex_ss = st.sidebar.slider("Steady-state CapEx (% of revenue)", 5, 30, 10, 1) / 100
cfg["steady_state"]["capex_pct_revenue"] = capex_ss
cfg["tax_rate"] = st.sidebar.slider("Tax rate (%)", 0, 35, 21, 1) / 100

st.sidebar.subheader("Share count")
use_override = st.sidebar.checkbox(
    "Use full share count (Class A + B, ~13.15B)", value=True,
    help="Yahoo reports ~7.57B (Class A only). Full count includes "
         "Musk's super-voting Class B shares.")
cfg["total_shares_override"] = 13.15e9 if use_override else None

if cfg["terminal_growth"] >= cfg["wacc"]:
    st.error("Terminal growth must be below WACC (Gordon Growth). "
             "Lower terminal growth or raise WACC.")
    st.stop()

# ------------------------------------------------------------------ model
df, out = build_dcf(cfg, data)
price = data["current_price"]
fv = out["fair_value"]
mos = (fv - price) / price

# ------------------------------------------------------------------ verdict
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current price", f"${price:,.2f}")
c2.metric("DCF fair value", f"${fv:,.2f}", f"{mos:+.1%} vs market")
c3.metric("Enterprise value", f"${out['enterprise_value']/1e9:,.0f}B")
c4.metric("WACC / terminal g", f"{out['wacc']:.1%} / {cfg['terminal_growth']:.1%}")

if mos > 0.20:
    st.success("**UNDERVALUED** — model fair value exceeds market price by a meaningful margin.")
elif mos < -0.20:
    st.warning("**OVERVALUED (per DCF)** — the market is pricing growth and optionality "
               "(Starship, xAI, Mars) well beyond these assumptions.")
else:
    st.info("**FAIRLY VALUED / NEUTRAL** — price is within ±20% of model fair value.")

# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "🎯 Sensitivity", "🔄 Reverse DCF", "📊 Fundamentals"])

with tab1:
    st.subheader(f"{cfg['forecast_years']}-year free cash flow forecast")
    display_df = (df / 1e9).round(2)
    st.dataframe(display_df, use_container_width=True)
    st.bar_chart(display_df[["FCF", "PV of FCF"]])
    st.caption(f"PV of explicit FCFs: ${out['pv_explicit']/1e9:,.1f}B · "
               f"PV of terminal value: ${out['pv_terminal']/1e9:,.1f}B "
               f"({out['pv_terminal']/out['enterprise_value']:.0%} of EV)")

with tab2:
    st.subheader("Fair value per share: WACC × terminal growth")
    sens = generate_sensitivity_table(cfg, data).round(2)
    st.dataframe(
        sens.style.background_gradient(cmap="RdYlGn", axis=None)
            .format("${:.2f}"),
        use_container_width=True)
    st.caption("Even the most generous corner of this grid sits far below "
               "the market price — the gap is structural, not parametric.")

with tab3:
    st.subheader("What growth does $%.2f imply?" % price)
    with st.spinner("Solving…"):
        implied = solve_reverse_dcf(cfg, data)
    if implied is not None:
        yr_n_rev = data["revenue"] * (1 + implied) ** cfg["forecast_years"] / 1e9
        st.metric("Market-implied revenue growth",
                  f"{implied:.1%} per year",
                  f"→ ${yr_n_rev:,.0f}B revenue in year {cfg['forecast_years']}")
        st.write(
            f"To justify **${price:,.2f}**, SpaceX must grow revenue "
            f"**{implied:.1%} annually for {cfg['forecast_years']} years** under the "
            f"current margin, CapEx, and discount assumptions. For context, no U.S. "
            f"company has ever sustained ~50% growth for a decade at this scale.")
    else:
        st.write("No growth rate within bounds equates fair value to the market "
                 "price under these assumptions.")

with tab4:
    st.subheader("Current fundamentals")
    shares_used = cfg["total_shares_override"] or data["shares_outstanding"]
    rows = [
        ("Revenue (TTM/FY)", f"${data['revenue']/1e9:,.2f}B", source["revenue"]),
        ("EBITDA", f"${data['ebitda']/1e9:,.2f}B", source["ebitda"]),
        ("Operating income", f"${data['operating_income']/1e9:,.2f}B", source["operating_income"]),
        ("Net income", f"${data['net_income']/1e9:,.2f}B", source["net_income"]),
        ("Free cash flow", f"${data['free_cash_flow']/1e9:,.2f}B", source["free_cash_flow"]),
        ("Cash", f"${data['cash']/1e9:,.2f}B", source["cash"]),
        ("Total debt", f"${data['debt']/1e9:,.2f}B", source["debt"]),
        ("Shares outstanding (used)", f"{shares_used/1e9:,.2f}B",
         "override" if cfg["total_shares_override"] else source["shares_outstanding"]),
        ("Beta", f"{data['beta']:.2f}", source["beta"]),
        ("Hist. revenue growth", f"{data['hist_revenue_growth']:.1%}", source["hist_revenue_growth"]),
        ("EBITDA margin", f"{data['ebitda_margin']:.1%}", "derived"),
        ("Operating margin", f"{data['operating_margin']:.1%}", "derived"),
    ]
    st.table(pd.DataFrame(rows, columns=["Metric", "Value", "Source"]))
    st.caption("SPCX IPO'd June 12, 2026 — statement history on Yahoo is thin, "
               "so S-1 filing figures are used as fallbacks where live data "
               "is unavailable.")

st.divider()
st.caption("Built in Python · yfinance + pandas + Streamlit · "
           "Educational analysis, not investment advice.")
