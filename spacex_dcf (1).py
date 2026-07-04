"""
================================================================================
 SPACEX (NASDAQ: SPCX) — AUTOMATED DCF VALUATION MODEL
================================================================================
 Pulls live financial data via yfinance, falls back to S-1 / IPO-disclosed
 figures if data is unavailable (SPCX IPO'd June 12, 2026 — history is thin).

 Usage:   python spacex_dcf.py
 Deps:    pip install yfinance pandas numpy
================================================================================
"""

import sys
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ==============================================================================
# ██  USER ASSUMPTIONS — EDIT THESE  ██
# ==============================================================================
ASSUMPTIONS = {
    "ticker": "SPCX",

    # --- Revenue growth, Years 1-5 (decimal). Scalar or list of 5. ---
    # SpaceX grew ~40-50% in 2025; street est. ~$25-37B for 2026.
    "revenue_growth": [0.50, 0.45, 0.40, 0.35, 0.30],

    # --- Terminal growth rate (Gordon Growth) ---
    "terminal_growth": 0.030,

    # --- Discount rate (WACC). If None, estimated via CAPM from beta. ---
    "wacc": 0.10,

    # --- Operating (EBIT) margin. Scalar, or list of 5 to model a ramp. ---
    # SpaceX posted an operating LOSS in 2025 (~ -14%); Starlink runs ~39%.
    # Default models a ramp toward profitability as Starlink scales.
    "operating_margin": [0.02, 0.08, 0.14, 0.20, 0.25],

    # --- Effective tax rate on positive EBIT ---
    "tax_rate": 0.21,

    # --- CapEx as % of revenue. Scalar or list of 5. ---
    # Heavy Starship + xAI datacenter buildout, easing as network completes.
    "capex_pct_revenue": [0.28, 0.25, 0.22, 0.18, 0.15],

    # --- Depreciation & Amortization as % of revenue ---
    "da_pct_revenue": 0.12,

    # --- Change in net working capital as % of *incremental* revenue ---
    "nwc_pct_delta_revenue": 0.03,

    # --- CAPM inputs (used only if wacc is None) ---
    "risk_free_rate": 0.042,
    "equity_risk_premium": 0.048,
    "default_beta": 1.4,

    # --- Total shares override (for Class A + B shares). None to use yfinance/fallback ---
    "total_shares_override": 13.15e9,

    # --- Forecast Horizon (Years) ---
    "forecast_years": 10,

    # --- Steady state values for linear fading of forecast assumptions ---
    "steady_state": {
        "growth": None,               # None means fade to terminal_growth
        "operating_margin": 0.25,
        "capex_pct_revenue": 0.10,
    },

    # --- Fallback fundamentals (from S-1 / IPO disclosures, FY2025) ---
    # Used only when yfinance returns nothing for a field.
    "fallbacks": {
        "revenue": 18_674e6,          # FY2025 revenue
        "ebitda": 6_584e6,            # FY2025 adjusted EBITDA
        "operating_income": -2_570e6, # FY2025 operating loss
        "net_income": -4_900e6,       # FY2025 GAAP net loss
        "free_cash_flow": -3_000e6,   # est. (cash burn on Starship/xAI)
        "cash": 90_000e6,             # est. post-IPO (~$85.7B gross raised)
        "debt": 12_000e6,             # est. (~$2B interest expense implied)
        "shares_outstanding": 13.15e9, # implied by ~$2.13T cap / ~$162
        "current_price": 162.00,
        "beta": 1.4,
        "hist_revenue_growth": 0.40,  # FY2024 -> FY2025
    },
}
# ==============================================================================
# END OF USER ASSUMPTIONS
# ==============================================================================

FORECAST_YEARS = 5


def as_list(value, n=FORECAST_YEARS):
    """Accept a scalar or a list for per-year assumptions."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Per-year assumption needs {n} values, got {len(value)}")
        return list(value)
    return [float(value)] * n


def get_faded_forecast(value, n_years, steady_value):
    """
    Converts a scalar or list into a list of length n_years.
    If the length of the list (L) is less than n_years:
    - Years 1 to L use the provided list values.
    - Years L+1 to n_years linearly fade from the last value (at L) to steady_value (at n_years).
    If the list is longer or equal, it returns the first n_years elements.
    """
    if isinstance(value, (list, tuple)):
        vals = list(value)
    else:
        vals = [float(value)]

    L = len(vals)
    if L >= n_years:
        return vals[:n_years]

    last_val = vals[-1]
    faded_vals = list(vals)
    for j in range(L, n_years):
        t = (j - L + 1) / (n_years - L)
        interpolated = (1.0 - t) * last_val + t * steady_value
        faded_vals.append(interpolated)
    return faded_vals


def safe_get(obj, *keys):
    """Pull the first non-null value for any of `keys` from a dict-like."""
    for k in keys:
        try:
            v = obj.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        except Exception:
            continue
    return None


def first_row_value(df, *row_names):
    """Grab most recent column value for the first matching row label."""
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            series = df.loc[name].dropna()
            if len(series) > 0:
                return float(series.iloc[0])
    return None


def fetch_data(cfg):
    """Pull fundamentals from yfinance; fall back to S-1 figures as needed."""
    fb = cfg["fallbacks"]
    data, source = {}, {}

    info, fin, cf, bs = {}, None, None, None
    if YF_AVAILABLE:
        try:
            tk = yf.Ticker(cfg["ticker"])
            info = tk.info or {}
            fin = tk.financials
            cf = tk.cashflow
            bs = tk.balance_sheet
        except Exception as e:
            print(f"[warn] yfinance fetch failed ({e}); using S-1 fallback data.\n")

    def resolve(key, live_value):
        if live_value is not None:
            data[key], source[key] = float(live_value), "yfinance"
        else:
            data[key], source[key] = float(fb[key]), "S-1/fallback"

    resolve("revenue", safe_get(info, "totalRevenue")
            or first_row_value(fin, "Total Revenue"))
    resolve("ebitda", safe_get(info, "ebitda")
            or first_row_value(fin, "EBITDA", "Normalized EBITDA"))
    resolve("operating_income", first_row_value(fin, "Operating Income", "EBIT"))
    resolve("net_income", safe_get(info, "netIncomeToCommon")
            or first_row_value(fin, "Net Income"))
    resolve("free_cash_flow", safe_get(info, "freeCashflow")
            or first_row_value(cf, "Free Cash Flow"))
    resolve("cash", safe_get(info, "totalCash")
            or first_row_value(bs, "Cash And Cash Equivalents",
                               "Cash Cash Equivalents And Short Term Investments"))
    resolve("debt", safe_get(info, "totalDebt")
            or first_row_value(bs, "Total Debt"))
    yf_shares = safe_get(info, "sharesOutstanding")
    data["shares_yfinance"] = yf_shares
    resolve("shares_outstanding", yf_shares)
    resolve("current_price", safe_get(info, "currentPrice",
                                      "regularMarketPrice",
                                      "previousClose"))
    resolve("beta", safe_get(info, "beta"))

    # Historical revenue growth (needs 2+ years of statements)
    hist_growth = None
    if fin is not None and not fin.empty and "Total Revenue" in fin.index:
        rev = fin.loc["Total Revenue"].dropna()
        if len(rev) >= 2 and rev.iloc[1] != 0:
            hist_growth = float(rev.iloc[0] / rev.iloc[1] - 1)
    resolve("hist_revenue_growth", hist_growth)

    # Derived metrics
    r = data["revenue"]
    data["ebitda_margin"] = data["ebitda"] / r if r else np.nan
    data["operating_margin"] = data["operating_income"] / r if r else np.nan
    data["net_margin"] = data["net_income"] / r if r else np.nan

    shares_used = cfg.get("total_shares_override") if cfg.get("total_shares_override") is not None else data["shares_outstanding"]
    data["eps"] = data["net_income"] / shares_used

    # ROIC = NOPAT / (debt + equity_book); approximate with market data
    nopat = data["operating_income"] * (1 - cfg["tax_rate"]) \
        if data["operating_income"] > 0 else data["operating_income"]
    invested = data["debt"] + max(data["cash"], 1)  # rough proxy
    data["roic"] = nopat / invested if invested else np.nan
    equity_proxy = shares_used * data["current_price"]
    data["roe"] = data["net_income"] / equity_proxy if equity_proxy else np.nan

    return data, source


def compute_wacc(cfg, beta):
    if cfg["wacc"] is not None:
        return float(cfg["wacc"]), "user assumption"
    wacc = cfg["risk_free_rate"] + beta * cfg["equity_risk_premium"]
    return wacc, f"CAPM (rf={cfg['risk_free_rate']:.1%}, beta={beta:.2f})"


def build_dcf(cfg, data, forecast_years=None, rev_growth_override=None):
    if forecast_years is None:
        forecast_years = cfg.get("forecast_years", 10)

    ss = cfg.get("steady_state", {})
    ss_growth = ss.get("growth") if ss.get("growth") is not None else cfg["terminal_growth"]
    ss_op_margin = ss.get("operating_margin", 0.25)
    ss_capex = ss.get("capex_pct_revenue", 0.10)

    if rev_growth_override is not None:
        growth = [float(rev_growth_override)] * forecast_years
    else:
        growth = get_faded_forecast(cfg["revenue_growth"], forecast_years, ss_growth)

    op_margin = get_faded_forecast(cfg["operating_margin"], forecast_years, ss_op_margin)
    capex_pct = get_faded_forecast(cfg["capex_pct_revenue"], forecast_years, ss_capex)
    da_pct = get_faded_forecast(cfg["da_pct_revenue"], forecast_years, cfg["da_pct_revenue"])
    nwc_pct = get_faded_forecast(cfg["nwc_pct_delta_revenue"], forecast_years, cfg["nwc_pct_delta_revenue"])
    tax = cfg["tax_rate"]

    wacc, wacc_src = compute_wacc(cfg, data["beta"])
    g_term = cfg["terminal_growth"]
    if g_term >= wacc:
        raise ValueError("ERROR: terminal growth must be below WACC (Gordon Growth).")

    rows = []
    rev_prev = data["revenue"]
    for yr in range(1, forecast_years + 1):
        i = yr - 1
        rev = rev_prev * (1 + growth[i])
        ebit = rev * op_margin[i]
        taxes = max(ebit, 0) * tax          # no tax benefit modeled on losses
        nopat = ebit - taxes
        da = rev * da_pct[i]
        capex = rev * capex_pct[i]
        d_nwc = (rev - rev_prev) * nwc_pct[i]
        fcf = nopat + da - capex - d_nwc
        pv = fcf / (1 + wacc) ** yr
        rows.append({
            "Year": yr, "Revenue": rev, "EBIT": ebit, "Taxes": taxes,
            "NOPAT": nopat, "D&A": da, "CapEx": capex, "ΔNWC": d_nwc,
            "FCF": fcf, "PV of FCF": pv,
        })
        rev_prev = rev

    df = pd.DataFrame(rows).set_index("Year")

    fcf_final = df.loc[forecast_years, "FCF"]
    terminal_value = fcf_final * (1 + g_term) / (wacc - g_term)
    pv_terminal = terminal_value / (1 + wacc) ** forecast_years

    enterprise_value = df["PV of FCF"].sum() + pv_terminal
    equity_value = enterprise_value + data["cash"] - data["debt"]

    shares_used = cfg.get("total_shares_override") if cfg.get("total_shares_override") is not None else data["shares_outstanding"]
    fair_value = equity_value / shares_used

    return df, {
        "wacc": wacc, "wacc_src": wacc_src,
        "terminal_value": terminal_value, "pv_terminal": pv_terminal,
        "pv_explicit": df["PV of FCF"].sum(),
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value": fair_value,
    }


def generate_sensitivity_table(cfg, data):
    waccs = [0.08, 0.09, 0.10, 0.11, 0.12]
    terminal_growths = [0.02, 0.025, 0.03, 0.035, 0.04]

    grid = {}
    for g in terminal_growths:
        col_name = f"{g:.1%}"
        grid[col_name] = []
        for w in waccs:
            if g >= w:
                grid[col_name].append(np.nan)
            else:
                temp_cfg = cfg.copy()
                temp_cfg["wacc"] = w
                temp_cfg["terminal_growth"] = g
                try:
                    _, out = build_dcf(temp_cfg, data)
                    grid[col_name].append(out["fair_value"])
                except Exception:
                    grid[col_name].append(np.nan)

    df = pd.DataFrame(grid, index=[f"{w:.1%}" for w in waccs])
    df.index.name = "WACC \\ g"
    return df


def solve_reverse_dcf(cfg, data):
    current_price = data["current_price"]
    forecast_years = cfg.get("forecast_years", 10)

    def get_fair_value_for_growth(g_rev):
        try:
            _, out = build_dcf(cfg, data, forecast_years=forecast_years, rev_growth_override=g_rev)
            return out["fair_value"]
        except Exception:
            return -1e12

    low = -0.99
    high = 10.0

    f_low = get_fair_value_for_growth(low) - current_price
    f_high = get_fair_value_for_growth(high) - current_price

    if f_low * f_high > 0:
        if f_low > 0:
            return None
        else:
            high = 100.0
            f_high = get_fair_value_for_growth(high) - current_price
            if f_low * f_high > 0:
                return None

    for _ in range(100):
        mid = (low + high) / 2
        f_mid = get_fair_value_for_growth(mid) - current_price
        if abs(f_mid) < 1e-5:
            return mid
        if f_mid * f_low < 0:
            high = mid
        else:
            low = mid
            f_low = f_mid

    return (low + high) / 2


def fmt_b(x):
    return f"${x/1e9:,.2f}B"


def main():
    cfg = ASSUMPTIONS
    print("=" * 78)
    print(f"  DCF VALUATION — {cfg['ticker']} (Space Exploration Technologies Corp)")
    print("=" * 78)

    data, source = fetch_data(cfg)

    print("\n--- CURRENT FUNDAMENTALS " + "-" * 52)
    shares_used = cfg.get("total_shares_override") if cfg.get("total_shares_override") is not None else data["shares_outstanding"]
    shares_used_src = "override" if cfg.get("total_shares_override") is not None else source["shares_outstanding"]
    yf_shares = data.get("shares_yfinance")
    yf_shares_str = f"{yf_shares/1e9:,.2f}B" if yf_shares is not None else "N/A"

    labels = [
        ("Revenue (TTM/FY)", fmt_b(data["revenue"]), source["revenue"]),
        ("EBITDA", fmt_b(data["ebitda"]), source["ebitda"]),
        ("Operating income", fmt_b(data["operating_income"]), source["operating_income"]),
        ("Net income", fmt_b(data["net_income"]), source["net_income"]),
        ("Free cash flow", fmt_b(data["free_cash_flow"]), source["free_cash_flow"]),
        ("Cash", fmt_b(data["cash"]), source["cash"]),
        ("Total debt", fmt_b(data["debt"]), source["debt"]),
        ("Shares outstanding (yf)", yf_shares_str, "yfinance" if yf_shares is not None else "N/A"),
        ("Shares outstanding (used)", f"{shares_used/1e9:,.2f}B", shares_used_src),
        ("Current price", f"${data['current_price']:,.2f}", source["current_price"]),
        ("Beta", f"{data['beta']:.2f}", source["beta"]),
        ("Hist. revenue growth", f"{data['hist_revenue_growth']:.1%}", source["hist_revenue_growth"]),
        ("EBITDA margin", f"{data['ebitda_margin']:.1%}", "derived"),
        ("Operating margin", f"{data['operating_margin']:.1%}", "derived"),
        ("Net margin", f"{data['net_margin']:.1%}", "derived"),
        ("EPS", f"${data['eps']:.2f}", "derived"),
        ("ROIC (approx)", f"{data['roic']:.1%}", "derived"),
        ("ROE (approx)", f"{data['roe']:.1%}", "derived"),
    ]
    for name, val, src in labels:
        print(f"  {name:<25}{val:>14}   [{src}]")

    df, out = build_dcf(cfg, data)

    forecast_years = cfg.get("forecast_years", 10)
    title = f"--- {forecast_years}-YEAR FORECAST ($B) "
    print("\n" + title + "-" * (77 - len(title)))
    print((df / 1e9).round(2).to_string())

    print("\n--- VALUATION " + "-" * 63)
    print(f"  Discount rate (WACC)      {out['wacc']:>12.2%}   [{out['wacc_src']}]")
    print(f"  Terminal growth           {cfg['terminal_growth']:>12.2%}")
    print(f"  PV of explicit FCFs       {fmt_b(out['pv_explicit']):>14}")
    print(f"  Terminal value            {fmt_b(out['terminal_value']):>14}")
    print(f"  PV of terminal value      {fmt_b(out['pv_terminal']):>14}")
    print(f"  Enterprise value          {fmt_b(out['enterprise_value']):>14}")
    print(f"  (+) Cash                  {fmt_b(data['cash']):>14}")
    print(f"  (-) Debt                  {fmt_b(data['debt']):>14}")
    print(f"  Equity value              {fmt_b(out['equity_value']):>14}")

    price = data["current_price"]
    fv = out["fair_value"]
    mos = (fv - price) / price

    print("\n--- VERDICT " + "-" * 65)
    print(f"  Current stock price       ${price:>12,.2f}")
    print(f"  Intrinsic value / share   ${fv:>12,.2f}")
    print(f"  Margin of safety          {mos:>13.1%}")

    if mos > 0.20:
        sentiment = "UNDERVALUED"
        note = "Model fair value exceeds market price by a meaningful margin."
    elif mos < -0.20:
        sentiment = "OVERVALUED"
        note = ("Market price far exceeds DCF fair value — the market is "
                "pricing in growth/optionality (Starship, xAI, Mars) well "
                "beyond these assumptions.")
    else:
        sentiment = "FAIRLY VALUED / NEUTRAL"
        note = "Price is within ±20% of model fair value."

    print(f"\n  >>> {sentiment} <<<")
    print(f"  {note}")
    print("\n  NOTE: SPCX IPO'd June 12, 2026. yfinance history is thin and")
    print("  the company is GAAP-unprofitable, so this DCF is extremely")
    print("  sensitive to the margin ramp and terminal assumptions above.")

    print("\n--- SENSITIVITY ANALYSIS (Fair Value / Share) " + "-" * 32)
    sens_df = generate_sensitivity_table(cfg, data)
    print(sens_df.round(2).to_string())

    print("\n--- REVERSE DCF ANALYSIS " + "-" * 52)
    implied_growth = solve_reverse_dcf(cfg, data)
    if implied_growth is not None:
        print(f"  Market-implied revenue growth: {implied_growth:.1%} per year for {forecast_years} years")
    else:
        print(f"  Market-implied revenue growth: Could not resolve (outside bounds)")
    print("=" * 78)


if __name__ == "__main__":
    main()
