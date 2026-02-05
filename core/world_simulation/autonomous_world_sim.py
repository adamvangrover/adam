# core/world_simulation/autonomous_world_sim.py

import random
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class MarketAgent(Agent):
    def __init__(self, unique_id, model, volatility, risk_aversion):
        super().__init__(unique_id, model)
        self.volatility = volatility
        self.risk_aversion = risk_aversion
        self.portfolio = {'cash': 1000}
        self.memory = []  # Store past actions and outcomes

    def step(self):
        current_prices = self.model.stock_prices
        economic_indicators = self.model.economic_indicators
        geopolitical_risks = self.model.geopolitical_risks

        self.memory.append({
            'prices': current_prices.copy(),
            'indicators': economic_indicators.copy(),
            'risks': geopolitical_risks.copy(),
            'portfolio': self.portfolio.copy()
        })

        if random.random() < (1 - self.risk_aversion):  # More risk-averse = less likely to trade
            for symbol in current_prices:
                price = current_prices[symbol][-1]
                if random.random() < (1 / len(current_prices)):  # Randomly choose a stock
                    quantity = int(random.uniform(-50, 50))  # Buy or sell
                    if quantity > 0 and self.portfolio['cash'] >= price * quantity:
                        self.buy_stock(symbol, quantity)
                    elif quantity < 0 and symbol in self.portfolio and self.portfolio.get(symbol, 0) >= abs(quantity):
                        self.sell_stock(symbol, abs(quantity))

        # Example of learning from memory (simplified)
        if len(self.memory) > 10:
            recent_outcomes = [m['portfolio']['cash'] for m in self.memory[-10:]]
            if recent_outcomes[-1] > recent_outcomes[0]:
                self.risk_aversion = max(0, self.risk_aversion - 0.01)  # Less risk averse if doing well.
            else:
                self.risk_aversion = min(1, self.risk_aversion + 0.01)  # More risk averse if doing poorly.

    def buy_stock(self, symbol, quantity):
        price = self.model.stock_prices[symbol][-1]
        cost = price * quantity
        self.portfolio['cash'] -= cost
        if symbol in self.portfolio:
            self.portfolio[symbol] += quantity
        else:
            self.portfolio[symbol] = quantity
        self.model.stock_prices[symbol][-1] = price * (1 + (quantity / 10000) * self.volatility)

    def sell_stock(self, symbol, quantity):
        price = self.model.stock_prices[symbol][-1]
        self.portfolio['cash'] += price * quantity
        self.portfolio[symbol] -= quantity
        self.model.stock_prices[symbol][-1] = price * (1 - (quantity / 10000) * self.volatility)


class EconomicAgent(Agent):
    def __init__(self, unique_id, model, volatility):
        super().__init__(unique_id, model)
        self.volatility = volatility

    def step(self):
        for indicator in self.model.economic_indicators:
            change = random.uniform(-0.1 * self.volatility, 0.1 * self.volatility)
            self.model.economic_indicators[indicator].append(self.model.economic_indicators[indicator][-1] + change)


class PoliticalAgent(Agent):
    def __init__(self, unique_id, model, volatility):
        super().__init__(unique_id, model)
        self.volatility = volatility

    def step(self):
        for risk in self.model.geopolitical_risks:
            change = random.uniform(-0.1 * self.volatility, 0.1 * self.volatility)
            self.model.geopolitical_risks[risk].append(max(0, min(1, self.model.geopolitical_risks[risk][-1] + change)))


class WorldSimulationModel(Model):
    def __init__(self, config):
        super().__init__()
        self.num_market_agents = config.get('num_market_agents', 100)
        self.num_economic_agents = config.get('num_economic_agents', 5)
        self.num_political_agents = config.get('num_political_agents', 3)
        self.schedule = RandomActivation(self)
        self.volatility = config.get('volatility', 0.05)
        self.stock_prices = {symbol: [random.uniform(100, 200)]
                             for symbol in config.get('stock_symbols', ['AAPL', 'MSFT', 'GOOG'])}
        self.economic_indicators = {indicator: [random.uniform(0, 10)] for indicator in config.get(
            'economic_indicators', ['GDP_growth', 'inflation', 'interest_rates'])}
        self.geopolitical_risks = {risk: [random.uniform(0, 1)] for risk in config.get(
            'geopolitical_risks', ['political_stability', 'trade_war_risk'])}

        for i in range(self.num_market_agents):
            risk_aversion = random.uniform(0, 1)
            agent = MarketAgent(i, self, self.volatility, risk_aversion)
            self.schedule.add(agent)

        for i in range(self.num_economic_agents):
            agent = EconomicAgent(self.num_market_agents + i, self, self.volatility)
            self.schedule.add(agent)

        for i in range(self.num_political_agents):
            agent = PoliticalAgent(self.num_market_agents + self.num_economic_agents + i, self, self.volatility)
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "Stock Prices": lambda m: {k: v[-1] for k, v in m.stock_prices.items()},
                "Economic Indicators": lambda m: m.economic_indicators,
                "Geopolitical Risks": lambda m: m.geopolitical_risks
            },
            agent_reporters={"Portfolio": lambda a: a.portfolio}
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def initialize_from_adam(self, adam_data):
        if 'stock_prices' in adam_data:
            self.stock_prices = {symbol: [price] for symbol, price in adam_data['stock_prices'].items()}
        if 'economic_indicators' in adam_data:
            self.economic_indicators = {indicator: [value]
                                        for indicator, value in adam_data['economic_indicators'].items()}
        if 'geopolitical_risks' in adam_data:
            self.geopolitical_risks = {risk: [value] for risk, value in adam_data['geopolitical_risks'].items()}
        # Add more data integration as needed.

# --- Standalone Execution ---


if __name__ == "__main__":
    config = {
        'num_market_agents': 100,
        'num_economic_agents': 5,
        'num_political_agents': 3,
        'volatility': 0.1,  # Adjust volatility as needed
        'stock_symbols': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'],  # Add more stock symbols
        # Add more economic indicators
        'economic_indicators': ['GDP_growth', 'inflation', 'interest_rates', 'unemployment'],
        # Add more geopolitical risks
        'geopolitical_risks': ['political_stability', 'trade_war_risk', 'regulatory_changes']
    }
    model = WorldSimulationModel(config)

    # Example standalone simulation
    for i in range(100):  # Example: Run for 100 steps
        model.step()

    # --- Data Analysis and Visualization ---
    # Example: Plot stock price trends
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    # Plot stock prices
    for stock in config['stock_symbols']:
        plt.plot(model_data['Stock Prices'][stock].values, label=stock)
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.title("Stock Price Trends")
    plt.legend()
    plt.show()

    # Analyze agent portfolio performance (example)
    final_portfolios = agent_data.xs(99, level="Step")['Portfolio']
    print("Final Portfolio Values:")
    print(final_portfolios)

    # ... Add more analysis and visualization as needed ...
