import json
from core.financial_suite.schemas.workstream_context import WorkstreamContext
from core.financial_suite.engines.solver import IterativeSolver
from core.financial_suite.modules.reporting.generator import ReportGenerator
from core.financial_suite.modules.vc.waterfall import WaterfallEngine
from core.financial_suite.modules.vc.return_metrics import ReturnMetrics


class ContextManager:
    def __init__(self, context_path: str = None, context_obj: WorkstreamContext = None):
        if context_path:
            with open(context_path, 'r') as f:
                data = json.load(f)
            self.context = WorkstreamContext(**data)
        elif context_obj:
            self.context = context_obj
        else:
            raise ValueError("Must provide context_path or context_obj")

        self.results = {}

    def run_workstream(self):
        """
        Executes the full financial workstream:
        1. Solver (WACC/EV/Rating equilibrium)
        2. VC Waterfall (Exit analysis)
        3. Reporting
        """
        # 1. Solve Equilibrium
        print("Running Solver...")
        solver_res = IterativeSolver.solve_equilibrium(self.context)
        self.results['solver'] = solver_res

        # Update context with solved state
        self.context = solver_res['context']
        ev = solver_res['valuation']['enterprise_value']

        # 2. VC Waterfall
        print(f"Calculating Waterfall (Exit EV: {ev:,.2f})...")
        waterfall = WaterfallEngine.calculate_exit_waterfall(self.context, ev)
        self.results['waterfall'] = waterfall

        # Metrics
        # Assume common equity investment sum
        common_invested = sum(
            s.investment or 0 for s in self.context.capital_structure.securities if s.security_type == "COMMON")
        # Naive matching or need better ID
        common_returned = sum(v for k, v in waterfall.items() if "Common" in k or "Sponsor" in k)
        # Better: iterate context securities again to match names
        common_returned = 0.0
        for sec in self.context.capital_structure.securities:
            if sec.security_type == "COMMON":
                common_returned += waterfall.get(sec.name, 0.0)

        moic = ReturnMetrics.calculate_moic(common_invested, common_returned)
        # Assume 5 year hold for IRR default
        irr = ReturnMetrics.calculate_irr(common_invested, common_returned, 5.0)

        self.results['metrics'] = {
            "moic": moic,
            "irr": irr,
            "common_invested": common_invested,
            "common_returned": common_returned
        }

        # 3. Report
        print("Generating Report...")
        report = ReportGenerator.generate_full_report(self.context, solver_res)
        self.results['report'] = report

        return self.results

    def export_report(self, filepath: str):
        if 'report' not in self.results:
            raise RuntimeError("Run workstream first.")
        with open(filepath, 'w') as f:
            f.write(self.results['report'])
