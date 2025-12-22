from .governance import GovernanceEngine
from .wealth_manager import WealthManager
from .deal_flow import DealFlowEngine
from .portfolio import PortfolioAggregator


class FamilyOfficeService:
    """
    The Unified Family Office Service.
    Acts as the central nexus for Governance, Wealth Management, Deal Flow, and Portfolio Aggregation.
    """

    def __init__(self):
        self.governance = GovernanceEngine()
        self.wealth = WealthManager()
        self.deal_flow = DealFlowEngine()
        self.portfolio = PortfolioAggregator()

    # Wrappers for easy MCP access
    def generate_ips(self, **kwargs):
        return self.governance.generate_ips(**kwargs)

    def plan_wealth_goal(self, **kwargs):
        return self.wealth.plan_goal(**kwargs)

    def screen_deal(self, **kwargs):
        return self.deal_flow.screen_deal(**kwargs)

    def aggregate_family_risk(self, **kwargs):
        return self.portfolio.aggregate_risk(**kwargs)
