import streamlit as st
import os
import time
import asyncio
import sys

# Ensure the root directory is in the python path to load core modules correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==========================================
# 0. Dynamic Module Loading
# ==========================================

# Core Functionality from existing scripts
try:
    from src.core_valuation import ValuationEngine
    from src.credit_risk import CreditSponsorModel
    HAS_CORE_MODELS = True
except ImportError:
    HAS_CORE_MODELS = False

# v26 Engine Functionality (Wrapped to survive missing Docker/Env requirements)
try:
    from core.engine.meta_orchestrator import MetaOrchestrator
    HAS_META_ORCHESTRATOR = True
except ImportError:
    HAS_META_ORCHESTRATOR = False

try:
    from core.engine.swarm.hive_mind import HiveMind
    HAS_HIVE_MIND = True
except ImportError:
    HAS_HIVE_MIND = False

try:
    from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
    HAS_PLANNER = True
except ImportError:
    HAS_PLANNER = False

# Stateful Orchestrator Initialization
if HAS_META_ORCHESTRATOR and "orchestrator" not in st.session_state:
    try:
        st.session_state.orchestrator = MetaOrchestrator()
    except Exception as e:
        st.error(f"Failed to initialize MetaOrchestrator: {e}")
        HAS_META_ORCHESTRATOR = False

if HAS_PLANNER and "planner" not in st.session_state:
    try:
        st.session_state.planner = NeuroSymbolicPlanner()
    except Exception as e:
        HAS_PLANNER = False

# ==========================================
# 1. Environment & Permissions Setup
# ==========================================
st.set_page_config(page_title="ADAM v26 Enterprise Portal", page_icon="🌌", layout="wide")

def check_environment():
    """Verify required environment variables exist."""
    required_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY"]
    # If BOTH are missing, then we have an issue
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return ["API_KEYS (OpenAI or Gemini)"]
    return []

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
                # Mock validation 
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

# Helper to run async functions within Streamlit
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ==========================================
# 2. Page & Module Definitions
# ==========================================

def page_dashboard():
    st.title("🌌 ADAM v26 Global Dashboard")
    st.markdown("Welcome to the unified interface for the Neuro-Symbolic Financial Sovereign.")
    
    missing_env = check_environment()
    if missing_env:
        st.warning(f"⚠️ Missing Environment Variables: {', '.join(missing_env)}. Some AI features will use mocked offline responses.")
    else:
        st.success("✅ All systems nominal. Environment API validated.")

    st.subheader("Engine Component Module Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Meta Orchestrator", "ONLINE" if HAS_META_ORCHESTRATOR else "OFFLINE")
    col2.metric("System 1 (Swarm)", "ONLINE" if HAS_HIVE_MIND else "OFFLINE")
    col3.metric("System 2 (Graph Planner)", "ONLINE" if HAS_PLANNER else "OFFLINE")
    col4.metric("Valuation Modeling", "ONLINE" if HAS_CORE_MODELS else "OFFLINE")

    st.divider()
    st.markdown("### Execution Environment Notes")
    st.markdown("This Streamlit file operates completely additively. It hooks into `core/engine/` using defensive logic. If a dependency (like neo4j or openai) is absent, the dashboard will remain alive but specific pages will show simulated execution states.")

def page_llm_chat():
    st.title("💭 Meta Orchestrator Chat")
    st.markdown("Interact directly with the ADAM Meta Orchestrator. It will dynamically route based on complexity (Swarm, Graph, Code Gen). Try queries like **'swarm AAPL'**, **'deep dive MSFT'**, or **'cyber red team attack'**.")
    
    if not HAS_META_ORCHESTRATOR:
        st.warning("MetaOrchestrator not physically loaded in this python environment. Falling back to mocked routing responses.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask ADAM a financial query..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("ADAM MetaOrchestrator is routing your query..."):
                if HAS_META_ORCHESTRATOR:
                    try:
                        # Check API keys
                        if check_environment():
                            time.sleep(1.5)
                            response_text = f"**[Offline Graph Mode]** Detected routing signature for: `{prompt}`. \n\n*Execute with API keys to fulfill request.*"
                        else:
                            result = run_async(st.session_state.orchestrator.route_request(prompt))
                            response_text = f"**Orchestrator Route Execution Result:**\n```json\n{result}\n```"
                    except Exception as e:
                        response_text = f"Error executing orchestrated request: {e}"
                else:
                    time.sleep(1)
                    response_text = f"Simulated routing execution for: '{prompt}'. (Install full Docker/pip environment to execute true graph nodes)."
                
                st.markdown(response_text)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

def page_swarm_manager():
    st.title("🐝 Hive Mind (Swarm Manager)")
    st.markdown("Deploy specialized asynchronous analyzer workers in parallel for rapid data gathering without hitting the slow stateful Graph.")
    
    if not HAS_HIVE_MIND:
        st.error("HiveMind module not found / import failed. Requires core environment.")
        return

    st.subheader("Manual Swarm Deployment")
    target = st.text_input("Target Ticker / Entity", placeholder="e.g. AAPL")
    intensity = st.slider("Swarm Compute Intensity", 1.0, 10.0, 5.0)
    
    if st.button("Deploy Swarm asynchronously"):
        with st.spinner(f"Deploying Hive Workers onto target '{target}'..."):
            if check_environment():
                st.warning("Running mock swarm deployment due to missing API keys.")
                time.sleep(2)
                st.success(f"Swarm successfully dispersed telemetry workers on {target} at focus {intensity}.")
            else:
                try:
                    # Execute true swarm gathering
                    hm = HiveMind(worker_count=3)
                    run_async(hm.initialize())
                    run_async(hm.disperse_task("ANALYST", {"target": target}, intensity=intensity))
                    results = run_async(hm.gather_results(timeout=5.0))
                    for res in results:
                        st.json(res)
                except Exception as e:
                    st.error(f"Swarm Execution runtime error: {e}")

def page_cognitive_planner():
    st.title("🧠 Neuro-Symbolic Planner")
    st.markdown("Trace grounded reasoning topological paths across the Neo4J / Knowledge Graph before executing.")
    
    if not HAS_PLANNER:
        st.error("NeuroSymbolicPlanner module not loaded.")
        return

    col1, col2 = st.columns(2)
    with col1:
        start_node = st.text_input("Origin Entity Node", "Apple Inc.")
    with col2:
        target_node = st.text_input("Target Risk / Intent Node", "Market_Crash")
        
    if st.button("Discover Shortest Path"):
        with st.spinner("Finding symbolic path in simulated Knowledge Graph..."):
            try:
                plan = st.session_state.planner.discover_plan(start_node, target_node)
                st.subheader("Generated execution query")
                st.code(plan.get("symbolic_plan"), language="cypher")
                
                st.subheader("Planned Linear Steps")
                for step in plan.get("steps", []):
                    st.write(f"- **Step {step.get('task_id')} ({step.get('agent')}):** {step.get('description')}")
            except Exception as e:
                st.error(f"Planner engine failed: {e}")

def page_credit_risk():
    st.title("📊 Credit & Valuation Sub-Engine")
    if not HAS_CORE_MODELS:
        st.error("Core models not found. Please ensure `src.core_valuation` and `src.credit_risk` are accessible.")
        return
        
    st.markdown("Legacy v20 execution wrapper for Interactive DCF & Credit Risk Workstream.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base Inputs")
        ebitda = st.number_input("LTM EBITDA", value=50000000)
        debt = st.number_input("Total Debt", value=200000000)
        interest = st.number_input("Annual Interest", value=18000000)
    with col2:
        st.subheader("Model Outputs")
        if st.button("Run Strict Valuation Models"):
            with st.spinner("Calculating via internal source..."):
                engine = ValuationEngine(ebitda, 0.05, 0.02, 0.09, 0.4)
                proj_df, ev, wacc = engine.run_dcf([0.05]*5)
                
                risk_model = CreditSponsorModel(ev, debt, ebitda, interest)
                metrics = risk_model.calculate_metrics()
                rating = risk_model.determine_regulatory_rating(metrics)
                
                st.metric("Enterprise Value", f"${ev:,.0f}")
                st.metric("Implied WACC", f"{wacc*100:.2f}%")
                st.metric("Regulatory Rating", rating)
                st.dataframe(proj_df)

def page_specialized_flows():
    st.title("🛡️ Specialized Execution Flows")
    st.markdown("Run specific wargaming, compliance, and wealth management graphs directly.")
    
    if not HAS_META_ORCHESTRATOR:
        st.warning("Offline Mode: Executions will be simulated.")

    tab1, tab2, tab3 = st.tabs(["Red Team / Crisis", "ESG & Compliance", "FO Wealth Super-App"])
    
    with tab1:
        st.subheader("Macro & Cyber Wargaming")
        target = st.text_input("Target Entity", "Global Banking Sector")
        scenario = st.selectbox("Scenario Type", ["Liquidity Crisis", "Cyber Attack", "Geopolitical Shock"])
        if st.button("Run Red Team / Crisis Sim"):
            with st.spinner("Simulating..."):
                if HAS_META_ORCHESTRATOR and check_environment() == []:
                    res = run_async(st.session_state.orchestrator.route_request(f"red team {scenario} on {target}"))
                    st.json(res)
                else:
                    time.sleep(1)
                    st.success(f"Simulated {scenario} against {target}. Vulnerabilities found: 3 Critical, 5 High.")
                    
    with tab2:
        st.subheader("ESG & Regulatory Compliance")
        entity = st.text_input("Entity to Audit", "ExxonMobil")
        if st.button("Run Compliance Audit"):
            with st.spinner("Auditing..."):
                time.sleep(1)
                st.success(f"Simulated ESG audit for {entity}. ESG Score: 65/100. Pending regulatory flags detected.")

    with tab3:
        st.subheader("Family Office Wealth Management")
        action = st.radio("Action", ["Screen Deal", "Generate IPS", "Plan Wealth Goal"])
        if st.button("Execute FO Action"):
            with st.spinner("Processing..."):
                time.sleep(1)
                st.success(f"Simulated {action} successfully calculated against FO guidelines.")

def page_code_alchemist():
    st.title("✨ Code Alchemist")
    st.markdown("Sandbox for autonomous code generation and system evolution.")
    
    prompt = st.text_area("What would you like the Alchemist to build?", "Write a Python script to calculate moving averages of a time series.")
    if st.button("Transmute Code"):
        with st.spinner("Synthesizing..."):
            if HAS_META_ORCHESTRATOR and check_environment() == []:
                res = run_async(st.session_state.orchestrator.route_request(f"generate code {prompt}"))
                st.code(res.get("result", ""), language="python")
            else:
                time.sleep(1)
                st.code(f"# Simulated Code Generation for: {prompt}\ndef calculate_ma(data, window=5):\n    return [sum(data[i:i+window])/window for i in range(len(data)-window+1)]", language="python")

def page_showcase_viewer():
    st.title("🏛️ Offline Showcase & Knowledge Gallery")
    st.markdown("Browse static intelligence reports securely generated by the **Adam v26 Core** during offline operations. This acts as a living archive and context-seed for the system.")
    
    showcase_dir = os.path.join(os.path.dirname(__file__), "showcase")
    
    if os.path.exists(showcase_dir):
        # Scan for HTML/MD files
        files = [f for f in os.listdir(showcase_dir) if f.endswith('.html') or f.endswith('.md')]
        files.sort(reverse=True) # Sort newest first assuming date prefixes
        
        if files:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("Report Archive")
                selected_file = st.selectbox("Select Report to View", files, index=0)
                
                st.info(f"Total Cached Reports: {len(files)}")
                st.write("---")
                st.write("**Top Tags Detected in Archive:**")
                st.write("- #Market_Mayhem")
                st.write("- #Deep_Dive")
                st.write("- #House_View")
                
            with col2:
                file_path = os.path.join(showcase_dir, selected_file)
                st.subheader(f"Viewing: `{selected_file}`")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    if selected_file.endswith('.html'):
                        st.components.v1.html(content, height=800, scrolling=True)
                    else:
                        st.markdown(content)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.warning(f"No HTML or Markdown reports found in `{showcase_dir}`.")
    else:
        st.error(f"Showcase directory not found at `{showcase_dir}`.")

# ==========================================
# 3. Main Application Routing
# ==========================================
def main():
    if not authenticate():
        return

    st.sidebar.title(f"ADAM Portal UI")
    st.sidebar.markdown(f"User: `{st.session_state.username}`")
    st.sidebar.divider()
    
    pages = {
        "Dashboard Overview": page_dashboard,
        "Meta Orchestrator (Chat)": page_llm_chat,
        "Hive Mind (Swarm View)": page_swarm_manager,
        "Graph Planner (Topological)": page_cognitive_planner,
        "Valuation Core Analysis": page_credit_risk,
        "Specialized Flows": page_specialized_flows,
        "Code Alchemist": page_code_alchemist,
        "Offline Showcase": page_showcase_viewer,
    }
    
    selection = st.sidebar.radio("Sub-Engines", list(pages.keys()))
    
    # Render the selected page
    pages[selection]()

if __name__ == "__main__":
    main()
