import streamlit as st
import json
import os

# Set page config
st.set_page_config(page_title="Adam: Vertical Risk Agent", layout="wide")

st.title("Adam: Vertical Risk Agent")
st.markdown("### Autonomous Credit Risk Analysis & Due Diligence")

# Sidebar for Inputs
with st.sidebar:
    st.header("Data Room")
    ticker = st.text_input("Ticker", "AAPL")
    uploaded_file = st.file_uploader("Upload 10-K or Credit Agreement", type=["pdf", "xml", "xlsx"])

    if st.button("Run Analysis"):
        st.session_state["running"] = True
        st.session_state["result"] = None

# Main Area
if st.session_state.get("running"):
    st.info("Orchestrating Agents... (Supervisor -> Quant -> Legal -> Market)")

    # Mocking the agent run
    import time
    time.sleep(1)

    result = {
        "balance_sheet": {
            "cash": 50000000,
            "debt": 120000000,
            "ebitda": 40000000
        },
        "covenants": [
            {"name": "Net Leverage Ratio", "threshold": 4.5, "current": 3.0}
        ],
        "memo": "Based on the analysis, the company is in good standing..."
    }
    st.session_state["result"] = result
    st.session_state["running"] = False

if st.session_state.get("result"):
    res = st.session_state["result"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial Analysis (Quant Agent)")
        st.json(res["balance_sheet"])

        # Edit Loop
        new_ebitda = st.number_input("Adjusted EBITDA (Override)", value=res["balance_sheet"]["ebitda"])
        if new_ebitda != res["balance_sheet"]["ebitda"]:
            st.warning(f"Analyst override: {new_ebitda}")
            # Log for DPO
            with open("dpo_dataset.jsonl", "a") as f:
                log_entry = {
                    "original": res["balance_sheet"]["ebitda"],
                    "correction": new_ebitda,
                    "ticker": ticker
                }
                f.write(json.dumps(log_entry) + "\n")
            st.success("Correction logged for training.")

    with col2:
        st.subheader("Covenant Analysis (Legal Agent)")
        st.table(res["covenants"])

    st.subheader("Investment Memo Draft")
    st.text_area("Memo", res["memo"], height=200)
