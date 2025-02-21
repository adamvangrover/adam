# core/world_simulation/wsm_v7_1.py

import random
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class MarketAgent(Agent):
    def __init__(self, unique_id, model, risk_aversion):
        super().__init__(unique_id, model)
        self.risk_aversion = risk_aversion
        self.portfolio = {'cash': 1000}  # Initial portfolio

    def step(self):
        # 1. Observe market conditions
        current_prices = self.model.stock_prices
        economic_indicators = self.model.economic_indicators

        # 2. Make investment decisions (example)
        if economic_indicators['GDP_growth'][-1] > 2.5 and current_prices['AAPL'][-1] < 180:
            # Buy AAPL if GDP growth is strong and price is below 180
            self.buy_stock('AAPL', 100)  # Buy 100 shares of AAPL

        #... (add more complex decision-making logic based on risk aversion and other factors)

    def buy_stock(self, symbol, quantity):
        #... (implement logic to buy stock, updating portfolio and market prices)
        pass  # Placeholder for actual implementation

class WorldSimulationModel(Model):
    def __init__(self, config):
        super().__init__()
        self.data_sources = config.get('data_sources', {})
        self.num_agents = config.get('num_agents', 100)
        self.schedule = RandomActivation(self)
        self.stock_prices = {'AAPL':, 'MSFT':}  # Initial stock prices
        self.economic_indicators = {'GDP_growth':, 'inflation':}  # Initial economic indicators
        #... (initialize other components, e.g., market parameters, data collectors)

        # Create agents
        for i in range(self.num_agents):
            risk_aversion = random.uniform(0, 1)  # Randomly assign risk aversion
            agent = MarketAgent(i, self, risk_aversion)
            self.schedule.add(agent)

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Stock Prices": lambda m: {k: v[-1] for k, v in m.stock_prices.items()},
                "Economic Indicators": lambda m: m.economic_indicators
            },
            agent_reporters={
                "Portfolio": lambda a: a.portfolio
            }
        )

    def step(self):
        # Update market conditions (example)
        self.stock_prices['AAPL'].append(self.stock_prices['AAPL'][-1] * (1 + random.uniform(-0.05, 0.05)))
        self.stock_prices['MSFT'].append(self.stock_prices['MSFT'][-1] * (1 + random.uniform(-0.05, 0.05)))
        self.economic_indicators['GDP_growth'].append(self.economic_indicators['GDP_growth'][-1] + random.uniform(-0.1, 0.1))
        #... (update other market parameters)

        self.schedule.step()
        self.datacollector.collect(self)
