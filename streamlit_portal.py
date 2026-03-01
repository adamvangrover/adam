import streamlit as st
import os
import time

# Optional imports for functionality integration
try:
    from src.core_valuation import ValuationEngine
    from src.credit_risk import CreditSponsorModel
    HAS_CORE_MODELS = True
except ImportError:
    HAS_CORE_MODELS = False

try:
    from core.api import api as core_api
    HAS_CORE_API = True
except ImportError:
    HAS_CORE_API = False

# ==========================================
# 1. Environment & Permissions Setup
# ==========================================
st.set_page_config(page_title="ADAM v26 Enterprise Portal", page_icon="🌌", layout="wide")

def check_environment():
    """Verify required environment variables exist."""
    required_vars = ["OPENAI_API_KEY"] # Add other required keys
    missing = [var for var in required_vars if not os.environ.get(var)]
    return missing

def authenticate():
    """Simple session-based authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔒 ADAM Secure Access")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                # Mock validation - replace with actual enterprise SSO/LDAP
                if username == "admin" and password == "admin": 
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                elif os.environ.get("ADMIN_PASSWORD") and password == os.environ.get("ADMIN_PASSWORD"):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return False
    return True

# ==========================================
# 2. Page & Module Definitions
# ==========================================

def page_dashboard():
    st.title("🌌 ADAM v26 Global Dashboard")
    st.markdown("Welcome to the unified interface for the Neuro-Symbolic Financial Sovereign.")
    
    missing_env = check_environment()
    if missing_env:
        st.warning(f"⚠️ Missing Environment Variables: {', '.join(missing_env)}. Some features may be disabled.")
    else:
        st.success("✅ All systems nominal. Environment verified.")

    col1, col2, col3 = st.columns(3)
    col1.metric("System 1 (Swarm) Status", "ACTIVE", "Optimal")
    col2.metric("System 2 (Graph) Status", "STANDBY", "")
    col3.metric("API Gateway", "ONLINE" if HAS_CORE_API else "OFFLINE")

def page_llm_chat():
    st.title("💬 LLM Interface (System 1 & 2 Query)")
    st.markdown("Interact directly with the ADAM Meta Orchestrator and Swarm.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask ADAM a financial question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("ADAM is thinking... (Routing via Orchestrator)"):
                # Mocked LLM response via internal APIs
                time.sleep(1)
                response = f"Simulated response to: '{prompt}'. Integrating Agent outputs..."
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

def page_credit_risk():
    st.title("📊 Credit & Valuation Engine")
    if not HAS_CORE_MODELS:
        st.error("Core models not found. Please ensure `src.core_valuation` and `src.credit_risk` are accessible.")
        return
        
    st.markdown("Interactive DCF & Credit Risk Workstream")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inputs")
        ebitda = st.number_input("LTM EBITDA", value=50000000)
        debt = st.number_input("Total Debt", value=200000000)
        interest = st.number_input("Annual Interest", value=18000000)
    with col2:
        st.subheader("Outputs")
        if st.button("Run Valuation Models"):
            with st.spinner("Calculating..."):
                engine = ValuationEngine(ebitda, 0.05, 0.02, 0.09, 0.4)
                proj_df, ev, wacc = engine.run_dcf([0.05]*5)
                
                risk_model = CreditSponsorModel(ev, debt, ebitda, interest)
                metrics = risk_model.calculate_metrics()
                rating = risk_model.determine_regulatory_rating(metrics)
                
                st.metric("Enterprise Value", f"${ev:,.0f}")
                st.metric("Implied WACC", f"{wacc*100:.2f}%")
                st.metric("Regulatory Rating", rating)
                st.dataframe(proj_df)

def page_api_manager():
    st.title("🔌 API & Integration Manager")
    st.markdown("Manage external connections, MCP servers, and data pipelines.")
    
    st.subheader("Loaded Modules Summary")
    st.write(f"- Core Valuation Module: {'✅ Loaded' if HAS_CORE_MODELS else '❌ Missing'}")
    st.write(f"- Flask API Exporter: {'✅ Loaded' if HAS_CORE_API else '❌ Missing'}")

    st.subheader("MCP (Model Context Protocol) Endpoints")
    st.code('''
    {
      "mcpServers": {
        "adam-core": {
          "command": "python",
          "args": ["-m", "server.mcp_server"]
        }
      }
    }
    ''', language="json")

# ==========================================
# 3. Main Application Routing
# ==========================================
def main():
    if not authenticate():
        return

    st.sidebar.title(f"ADAM Enterprise")
    st.sidebar.markdown(f"User: `{st.session_state.username}`")
    st.sidebar.divider()
    
    pages = {
        "Dashboard": page_dashboard,
        "LLM Chat": page_llm_chat,
        "Credit & Valuation": page_credit_risk,
        "API integrations": page_api_manager,
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Render the selected page
    pages[selection]()

if __name__ == "__main__":
    main()
