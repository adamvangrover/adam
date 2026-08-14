1. **Analyze Current Sub-Brands**: Read the content of `public/adam-research/index.html`, `public/adam-governance/index.html`, `public/market-mayhem/index.html`, `public/adam-institutional/index.html`, `public/adam-os/index.html`, `public/fortress-hunt/index.html`, and `public/adam-terminals/index.html`.
2. **Retrieve Real Data**:
    - **Adam Research**: Needs eval sets (e.g., `evals/data/finance_bench.json`, output of `evals/unified_eval.py`).
    - **Adam Governance**: Needs compliance rules from `AGENTS.md` (e.g., "Enforce 0.85 minimum conviction score", "terminate if jsonLogic_version missing").
    - **Market Mayhem**: Needs current momentum data/crisis simulations (e.g., `2022_INFLATION_SHOCK` scenario from `scripts/market_mayhem_crisis_sim.py`).
    - **Adam OS**: Needs actual agent descriptions from `AGENTS.md`.
    - **Adam Institutional / Fortress-Hunt / Terminals**: Needs specific real variables from the codebase (like SOFR, 10Y UST, actual system state/memory files).
3. **Write a Hydration Script**: Create a python script `scripts/hydrate_subbrands.py` that parses the actual JSON, Markdown, and Python files to extract this representative real data, and injects it into the respective `public/<sub-brand>/index.html` files, fully expanding the dummy data with rich, dynamic layouts (while preserving the cybernetic CSS injected previously).
4. **Run the Script and Verify**: Execute the script, check the HTML outputs visually (via Playwright or `cat`) to confirm the overlays and assumptions are accurately reflected.
5. **Complete Pre-Commit Steps**.
6. **Submit**.
