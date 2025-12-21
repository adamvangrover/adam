# app.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.core_valuation import ValuationEngine
from src.credit_risk import CreditSponsorModel

st.set_page_config(page_title="ADAM Risk Core", layout="wide")

st.title("Interactive DCF & Credit Risk Workstream")
st.markdown("### VC Sponsor Model | Regulatory Ratings | Downside Sensitivity")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Financial Inputs")
ebitda_input = st.sidebar.number_input("LTM EBITDA ($)", value=50_000_000)
growth_input = st.sidebar.slider("CAGR Growth Rate (%)", 0, 20, 5) / 100
debt_input = st.sidebar.number_input("Total Debt ($)", value=200_000_000)
interest_input = st.sidebar.number_input("Annual Interest ($)", value=18_000_000)

st.sidebar.header("2. Capital Structure")
entry_mult = st.sidebar.number_input("Entry Multiple (x)", value=10.0)
equity_pct = st.sidebar.slider("Equity % in Cap Stack", 20, 100, 40) / 100
kd_input = st.sidebar.slider("Cost of Debt (%)", 4, 15, 9) / 100

# --- CALCULATIONS ---

# 1. Valuation
val_engine = ValuationEngine(ebitda_input, 0.05, 0.02, kd_input, equity_pct)
growth_array = [growth_input] * 5 # Flat growth for simple demo
proj_df, enterprise_value, wacc = val_engine.run_dcf(growth_array)

# 2. Credit Risk
risk_model = CreditSponsorModel(enterprise_value, debt_input, ebitda_input, interest_input)
base_metrics = risk_model.calculate_metrics()
base_rating = risk_model.determine_regulatory_rating(base_metrics)
snc_status = risk_model.snc_check()

# --- MAIN DASHBOARD ---

# Row 1: KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Enterprise Value (DCF)", f"${enterprise_value:,.0f}")
col2.metric("Implied WACC", f"{wacc*100:.2f}%")
col3.metric("Base Reg Rating", base_rating)
col4.metric("SNC Status", "YES" if "REQUIRED" in snc_status else "NO", delta_color="inverse")

# Row 2: Tabs for Detail
tab1, tab2, tab3 = st.tabs(["DCF Worksheet", "Credit Sensitivity", "VC Sponsor Returns"])

with tab1:
    st.subheader("Discounted Cash Flow Projections")
    st.dataframe(proj_df.style.format("${:,.2f}"))

    # Chart
    st.bar_chart(proj_df.set_index("Year")['FCF'])

with tab2:
    st.subheader("Credit Challenge: Downside Sensitivity")
    st.write("Impact of EBITDA contraction on Leverage & Regulatory Rating")

    stress_scenarios = [0.0, 0.10, 0.20, 0.30] # 0%, 10%, 20%, 30% downside
    results = []

    for s in stress_scenarios:
        m, r = risk_model.perform_downside_stress(stress_factor=s)
        results.append({
            "EBITDA Stress": f"-{s*100:.0f}%",
            "Leverage (x)": m['Leverage (x)'],
            "FCCR (x)": m['FCCR (x)'],
            "Resulting Rating": r
        })

    st.table(pd.DataFrame(results))

    # Heatmap Logic (Dynamic)
    st.subheader("PD / LTV Sensitivity Matrix")
    # Generate dummy matrix for visualization
    x_axis = [5.0, 5.5, 6.0, 6.5, 7.0] # Leverage
    y_axis = [1.1, 1.2, 1.3, 1.4, 1.5] # FCCR
    data = [[x/y for x in x_axis] for y in y_axis]

    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="RdYlGn_r", ax=ax)
    ax.set_xlabel("Leverage (x)")
    ax.set_ylabel("FCCR (x)")
    st.pyplot(fig)

with tab3:
    st.subheader("Sponsor Returns (5Y Exit)")
    exit_mult = st.slider("Exit Multiple Assumption", 6.0, 14.0, 10.0)

    future_ebitda = proj_df.iloc[-1]['EBITDA']
    exit_ev = future_ebitda * exit_mult
    # Simple debt paydown assumption (50% of cumulative FCF)
    cum_fcf = proj_df['FCF'].sum()
    remaining_debt = max(0, debt_input - (cum_fcf * 0.5))
    exit_equity = exit_ev - remaining_debt

    initial_equity = (ebitda_input * entry_mult) * equity_pct # Based on entry mult

    moic = exit_equity / initial_equity
    irr = (moic ** (1/5)) - 1

    c1, c2 = st.columns(2)
    c1.metric("MOIC (Multiple of Money)", f"{moic:.2f}x")
    c2.metric("IRR (%)", f"{irr*100:.1f}%")
