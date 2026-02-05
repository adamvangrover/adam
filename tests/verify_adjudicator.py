import sys
from unittest.mock import MagicMock

# Mock missing dependencies to bypass __init__ side effects
sys.modules["yaml"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["langgraph"] = MagicMock()
sys.modules["langchain"] = MagicMock()
sys.modules["neo4j"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pandas"] = MagicMock()

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Attempt to import AdjudicatorEngine
# If core.engine.__init__ fails, we might need to load the module directly from file path
try:
    from core.engine.adjudicator_engine import AdjudicatorEngine
except ImportError:
    # If package import fails due to __init__, load directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("adjudicator_engine",
        os.path.join(os.path.dirname(__file__), '../core/engine/adjudicator_engine.py'))
    adjudicator_engine = importlib.util.module_from_spec(spec)
    sys.modules["core.engine.adjudicator_engine"] = adjudicator_engine
    spec.loader.exec_module(adjudicator_engine)
    AdjudicatorEngine = adjudicator_engine.AdjudicatorEngine

def verify_adjudicator():
    engine = AdjudicatorEngine()
    engine.reset()
    state = engine.get_state()

    print("--- ACT I: Start ---")
    # 450.0 is the default in init_double_crisis_state
    assert state['sovereign_spread'] == 450.0
    assert "INJECT_001" in engine.active_injects

    # Player chooses Option 3: Sell Bonds
    print("Player chooses: Sell Bonds")
    engine.resolve_action("INJECT_001", "A1_OPT3")

    # Verify Fire Sale Impact
    # Spread should be 450 + 100 + drift
    print(f"Spread: {state['sovereign_spread']}")
    assert state['sovereign_spread'] >= 550.0

    # Verify CET1 Hit
    # Loss = 10 * 0.05 = 0.5B
    # RWA = 300
    # Hit = 0.5/300 = 0.166%
    # CET1 = 13.5 - 0.166 = 13.333
    print(f"CET1: {state['cet1']}")
    assert state['cet1'] < 13.5

    print("--- ACT II: Repo Squeeze ---")
    # Should have triggered automatically
    assert "INJECT_002" in engine.active_injects
    # Spread > 500, so Haircut should be 15%
    print(f"Repo Haircut: {state['repo_haircut']}")
    assert state['repo_haircut'] == 15.0

    # Player chooses Option 1: Pay from Buffer
    print("Player chooses: Pay from Buffer")
    engine.resolve_action("INJECT_002", "A2_OPT1")

    # Verify Liquidity Drop
    # Gap was: 50B * (0.15 - 0.02) = 50 * 0.13 = 6.5B
    # Initial Buffer = 5B.
    # New Buffer = 5 - 6.5 = -1.5B
    print(f"Intraday Liquidity: {state['intraday_liquidity']}")
    assert state['intraday_liquidity'] < 0

    # Verify LCR Hit
    # Deficit = 6.5B (actually it's 0 - (-1.5) if we count strictly, but logic says max(0, 5000 - liquidity))
    # Wait, 5000 - (-1500) = 6500 deficit?
    # Logic: deficit = max(0, 5000 - self.state['intraday_liquidity'])
    # deficit = 5000 - (-1500) = 6500
    # lcr_hit = (6500 / 1000) * 5 = 32.5
    # LCR = 115 - 32.5 = 82.5
    print(f"LCR: {state['lcr']}")
    assert state['lcr'] < 100.0

    print("--- ACT III: Counterparty ---")
    assert "INJECT_003" in engine.active_injects

    # Player chooses Option 1: Default
    print("Player chooses: Call Default")
    engine.resolve_action("INJECT_003", "A3_OPT1")

    # CVA Charge
    assert state['counterparty_cds'] >= 10000

    print("--- ACT IV: Resolution ---")
    assert "INJECT_004" in engine.active_injects

    print("VERIFICATION SUCCESSFUL: All Acts Triggered and Logic Verified.")

if __name__ == "__main__":
    verify_adjudicator()
