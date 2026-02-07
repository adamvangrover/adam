import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import pandas as pd

class TraderAgent(Agent):
    """
    An agent that trades stocks based on a simple strategy.
    """
    def __init__(self, unique_id, model, initial_wealth, strategy):
        super().__init__(unique_id, model)
        self.wealth = initial_wealth
        self.strategy = strategy
        self.portfolio = {}

    def step(self):
        # Determine action based on strategy
        action = self.strategy(self.model)

        if action["type"] == "buy":
            self.buy(action["ticker"], action["amount"])
        elif action["type"] == "sell":
            self.sell(action["ticker"], action["amount"])

    def buy(self, ticker, amount):
        price = self.model.get_price(ticker)
        cost = price * amount
        if self.wealth >= cost:
            self.wealth -= cost
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + amount

    def sell(self, ticker, amount):
        if self.portfolio.get(ticker, 0) >= amount:
            price = self.model.get_price(ticker)
            revenue = price * amount
            self.wealth += revenue
            self.portfolio[ticker] -= amount

class MarketModel(Model):
    """
    A model simulating a stock market with trader agents.
    """
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True
        self.prices = {"AAPL": 150.0, "GOOG": 2800.0, "TSLA": 900.0}

        # Create agents
        for i in range(self.num_agents):
            strategy = self.random_strategy
            a = TraderAgent(i, self, 10000, strategy)
            self.schedule.add(a)

            # Add agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Stock Prices": lambda m: m.prices.copy()},
            agent_reporters={"Wealth": "wealth"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.update_prices()

    def update_prices(self):
        # Simulate random price movement (Random Walk)
        for ticker in self.prices:
            change = random.uniform(-0.02, 0.02)
            self.prices[ticker] *= (1 + change)

    def get_price(self, ticker):
        return self.prices.get(ticker, 0.0)

    def random_strategy(self, model):
        # Simple random strategy
        ticker = random.choice(list(model.prices.keys()))
        action_type = random.choice(["buy", "sell", "hold"])
        amount = random.randint(1, 10)
        return {"type": action_type, "ticker": ticker, "amount": amount}

def run_simulation():
    # Create and run the model
    model = MarketModel(50, 10, 10)
    for i in range(100):
        model.step()

    # Retrieve and plot data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    print(model_data.head())

    # Plot stock prices
    plt.figure(figsize=(10, 6))
    # Extract prices for plotting - this is a bit tricky because the column contains dicts
    # We need to normalize or extract keys

    # Simple workaround for visualization if needed, assumes 'Stock Prices' column of dicts
    # In a real scenario, you'd process this data structure better

    # For demonstration, let's just print that we plotted
    print("Simulation complete. Data collected.")

    # Example plotting code if we extracted a single stock
    # prices_aapl = [d['AAPL'] for d in model_data['Stock Prices']]
    # plt.plot(prices_aapl)
    # plt.show()

    # Re-enabling the plotting logic that was causing the error, now with plt imported
    for stock in ["AAPL", "GOOG", "TSLA"]:
        # Extract series for each stock
        stock_series = model_data['Stock Prices'].apply(lambda x: x[stock])
        plt.plot(stock_series, label=stock)

    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.title("Stock Price Trends")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()
