import json
import os

import streamlit as st

# Set page config
st.set_page_config(page_title="Adam: Adaptive Financial System", layout="wide")

st.title("Adam: Adaptive Financial System")

# Tabs for Unified UI
tab_home, tab_risk = st.tabs(["🏠 Home / Showcase", "🛡️ Vertical Risk Agent"])

with tab_home:
    st.header("Market Mayhem: Daily Briefing")
    st.markdown("Automated insights from the Hyper-Dimensional Knowledge Graph (HDKG).")

    # Load latest newsletter
    try:
        import glob
        newsletter_dir = "core/libraries_and_archives/newsletters"
        list_of_files = glob.glob(os.path.join(newsletter_dir, "Market_Mayhem_*.md"))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, "r") as f:
                content = f.read()
            st.markdown("---")
            st.markdown(content)
        else:
            st.warning("No newsletters generated yet. Run 'scripts/generate_market_mayhem.py'.")
    except Exception as e:
        st.error(f"Error loading newsletter: {e}")

with tab_risk:
    st.markdown("### Autonomous Credit Risk Analysis & Due Diligence")

    # Sidebar for Inputs (only show if this tab is active? Streamlit sidebar is global)
    # We'll keep it simple.

    col_input, col_main = st.columns([1, 3])

    with col_input:
        st.header("Data Room")
        ticker = st.text_input("Ticker", "AAPL")
        uploaded_file = st.file_uploader("Upload 10-K or Credit Agreement", type=["pdf", "xml", "xlsx"])

        if st.button("Run Risk Analysis"):
            st.session_state["running"] = True
            st.session_state["result"] = None

    with col_main:
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

            c1, c2 = st.columns(2)

            with c1:
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

            with c2:
                st.subheader("Covenant Analysis (Legal Agent)")
                st.table(res["covenants"])

            st.subheader("Investment Memo Draft")
            st.text_area("Memo", res["memo"], height=200)
