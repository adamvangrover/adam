from core.system.nexus_zero_orchestrator import NexusZeroOrchestrator
import streamlit as st
import json
import os
import asyncio
import sys

# Ensure core is in path if run from subdirectory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if repo_root not in sys.path:
    sys.path.append(repo_root)


# Page Config
st.set_page_config(page_title="Odyssey: CRO Copilot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #00bcd4; /* Cyan */
    }
</style>
""", unsafe_allow_html=True)

st.title("Odyssey: Unified Financial Knowledge Graph")
st.sidebar.title("Nexus-Zero Orchestration")

# Tabs
tab_mayhem, tab_analyst, tab_odyssey = st.tabs(["🔥 Market Mayhem", "💻 Analyst OS", "🧠 Odyssey Orchestrator"])

# --- Market Mayhem ---
with tab_mayhem:
    st.header("Daily Briefing & Meme Engine")
    st.info("Integrating TMT/Healthcare seriousness with Internet Culture.")

    # Mock Meme Content
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### The 'Great Divergence'")
        st.markdown("*Asset valuations detach from macro reality.*")
        # Placeholder for meme image
        st.markdown("*(Meme: Fed Rates vs Tech Valuations)*")

    with c2:
        st.markdown("### J.Crew Detection")
        st.markdown("*When the IP goes for a walk...*")
        st.code("Detected: UnrestrictedSubsidiary transfer > $500M")

# --- Analyst OS ---
with tab_analyst:
    st.header("Privacy-First Financial Workbench")
    st.markdown("Local DCF/LBO Models connected to the OUFKG Golden Source.")

    col_dcf, col_graph = st.columns(2)

    with col_dcf:
        st.subheader("Local DCF Model")
        wacc = st.slider("WACC (%)", 5.0, 15.0, 8.5)
        growth = st.slider("Terminal Growth (%)", 1.0, 5.0, 2.5)
        fcf = st.number_input("FCF (Year 1)", 100)

        # Simple Calc
        if wacc > growth:
            value = fcf / (wacc/100 - growth/100)
            st.metric("Implied Enterprise Value", f"${value:,.2f}")
        else:
            st.error("WACC must be > Growth")

    with col_graph:
        st.subheader("Graph Connection")
        st.text("Status: Connected to OUFKG")
        st.json({
            "Source": "Golden Source Lakehouse",
            "FIBO_Mapping": "Verified",
            "Latest_Update": "2024-05-25T14:00:00Z"
        })

# --- Odyssey Orchestrator ---
with tab_odyssey:
    st.header("Risk Intelligence Core")

    query = st.text_input("Strategic Query", "Assess impact of 50bps rate hike on Tech portfolio")

    # Sample Data Payload (Mocking an upload)
    st.subheader("Context / Data Payload")
    sample_json = {
        "@id": "urn:fibo:be-le-cb:Corporation:US-NVIDIA",
        "@type": "fibo-be-le-cb:Corporation",
        "legalName": "NVIDIA Corp",
        "hasCreditFacility": [
            {
                "@id": "urn:fibo:loan:TermLoanB-NVDA",
                "facilityType": "Term Loan B",
                "hasJCrewBlocker": False,
                "interestCoverageRatio": 12.5
            }
        ]
    }
    data_payload = st.text_area("JSON-LD Payload", json.dumps(sample_json, indent=2))

    if st.button("Execute Nexus-Zero Protocol"):
        st.write("Initializing Nexus-Zero...")
        orchestrator = NexusZeroOrchestrator()

        with st.spinner("Running Agents (Sentinel -> Sentry -> Odyssey)..."):
            # Run Async Loop
            try:
                # Helper to run async in Streamlit
                async def run_async():
                    payload_dict = json.loads(data_payload)
                    return await orchestrator.run_analysis(query, payload_dict)

                result = asyncio.run(run_async())
                st.success("Analysis Complete")
                st.json(result)

                st.markdown("### Credit Decision XML")
                decision = result.get("final_decision", {}).get("decision_xml")
                if decision:
                    st.code(decision, language="xml")
            except Exception as e:
                st.error(f"Execution Error: {e}")
