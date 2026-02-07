import mesa
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# --- Configuration & Schemas ---

DEFAULT_CONFIG = {
    'num_traders': 50,
    'num_economic_agents': 3,
    'num_political_agents': 2,
    'volatility': 0.05, 
    'initial_cash': 10_000,
    'stock_symbols': ['AAPL', 'GOOG', 'TSLA', 'AMZN'],
    'economic_indicators': ['GDP_Growth', 'Inflation', 'Interest_Rate'],
    'geopolitical_risks': ['Political_Instability', 'Trade_Tension']
}

# --- Agents ---

class SmartTrader(Agent):
    """
    A trader that learns from history and adapts its risk aversion.
    Merges 'MarketAgent' memory with 'TraderAgent' strategy execution.
    """
    def __init__(self, unique_id: int, model: Model, initial_cash: float, risk_aversion: float):
        super().__init__(unique_id, model)
        self.wealth = initial_cash
        self.portfolio = {ticker: 0 for ticker in model.stock_symbols}
        self.risk_aversion = risk_aversion
        self.memory = []  # Stores (action, result_wealth) tuples

    def step(self):
        # 1. Update Strategy based on Performance (Learning)
        self._adapt_strategy()

        # 2. Decide Action (Buy/Sell/Hold)
        # Higher risk aversion = lower probability of trading
        if random.random() > self.risk_aversion:
            ticker = random.choice(self.model.stock_symbols)
            # Simple heuristic: Buy if recent price dip, Sell if spike (Mean Reversion)
            # In a real impl, this would be a pluggable Strategy function
            price_trend = self.model.get_price_trend(ticker)
            
            if price_trend < -0.02: # Buy the dip
                self._trade(ticker, "buy")
            elif price_trend > 0.02: # Sell the rip
                self._trade(ticker, "sell")
            else:
                # Random noise trade
                if random.random() < 0.5: self._trade(ticker, "buy")

    def _adapt_strategy(self):
        """Adjusts risk aversion based on recent portfolio performance."""
        if len(self.memory) > 5:
            start_wealth = self.memory[-5]
            current_wealth = self.wealth + self._calculate_portfolio_value()
            
            if current_wealth < start_wealth:
                # Lost money -> Become more conservative
                self.risk_aversion = min(0.95, self.risk_aversion + 0.05)
            else:
                # Made money -> Become more aggressive
                self.risk_aversion = max(0.05, self.risk_aversion - 0.02)
        
        self.memory.append(self.wealth + self._calculate_portfolio_value())

    def _calculate_portfolio_value(self):
        val = 0
        for ticker, qty in self.portfolio.items():
            val += qty * self.model.stock_prices[ticker]
        return val

    def _trade(self, ticker: str, action: str):
        price = self.model.stock_prices[ticker]
        # Trade size scaled by confidence (inverse of risk_aversion)
        max_trade_size = 50 * (1 - self.risk_aversion) 
        quantity = int(random.uniform(1, max(2, max_trade_size)))

        if action == "buy":
            cost = price * quantity
            if self.wealth >= cost:
                self.wealth -= cost
                self.portfolio[ticker] += quantity
                self.model.register_trade(ticker, quantity, "buy") # Impact Price
        
        elif action == "sell":
            if self.portfolio[ticker] >= quantity:
                revenue = price * quantity
                self.wealth += revenue
                self.portfolio[ticker] -= quantity
                self.model.register_trade(ticker, quantity, "sell") # Impact Price

class MacroAgent(Agent):
    """
    Simulates external Macroeconomic or Political shocks.
    Merges EconomicAgent and PoliticalAgent into one systematic driver.
    """
    def __init__(self, unique_id: int, model: Model, agent_type: str, volatility: float):
        super().__init__(unique_id, model)
        self.type = agent_type # 'economic' or 'political'
        self.volatility = volatility

    def step(self):
        target_dict = (self.model.economic_indicators if self.type == 'economic' 
                       else self.model.geopolitical_risks)
        
        for key in target_dict:
            # Random walk with mean reversion tendencies
            change = random.uniform(-0.1, 0.1) * self.volatility
            current = target_dict[key]
            
            # Keep percentages roughly between 0 and 10 (or 0 and 1)
            new_val = current + change
            if self.type == 'political': 
                new_val = max(0.0, min(1.0, new_val))
            
            target_dict[key] = new_val

# --- The Autonomous World ---

class AutonomousMarket(Model):
    """
    The central simulation engine.
    Manages global state, order matching impacts, and data collection.
    """
    def __init__(self, config: Dict[str, Any] = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.stock_symbols = config['stock_symbols']
        self.schedule = RandomActivation(self)
        self.running = True
        
        # Global State
        self.stock_prices = {s: random.uniform(100, 200) for s in self.stock_symbols}
        self.price_history = {s: [100.0] * 5 for s in self.stock_symbols} # Init with dummy history
        
        self.economic_indicators = {k: random.uniform(1, 5) for k in config['economic_indicators']}
        self.geopolitical_risks = {k: random.uniform(0.1, 0.3) for k in config['geopolitical_risks']}

        # Initialize Agents
        self._init_agents()

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "Prices": lambda m: m.stock_prices.copy(),
                "Economics": lambda m: m.economic_indicators.copy(),
                "Risks": lambda m: m.geopolitical_risks.copy()
            },
            agent_reporters={"Wealth": lambda a: getattr(a, "wealth", 0)}
        )

    def _init_agents(self):
        # 1. Traders
        for i in range(self.config['num_traders']):
            a = SmartTrader(i, self, self.config['initial_cash'], random.uniform(0.2, 0.8))
            self.schedule.add(a)
        
        # 2. Macro Agents
        offset = self.config['num_traders']
        for i in range(self.config['num_economic_agents']):
            self.schedule.add(MacroAgent(offset + i, self, 'economic', self.config['volatility']))
        
        for i in range(self.config['num_political_agents']):
            self.schedule.add(MacroAgent(offset + 100 + i, self, 'political', self.config['volatility']))

    def register_trade(self, ticker, quantity, side):
        """
        Price Impact Model:
        Price moves proportional to volume and market volatility.
        """
        impact_factor = (quantity / 10000) * self.config['volatility']
        
        if side == "buy":
            self.stock_prices[ticker] *= (1 + impact_factor)
        else:
            self.stock_prices[ticker] *= (1 - impact_factor)

    def get_price_trend(self, ticker):
        """Returns % change over last 3 steps."""
        hist = self.price_history[ticker]
        if len(hist) < 3: return 0.0
        return (hist[-1] - hist[-3]) / hist[-3]

    def step(self):
        # 1. Save current prices to history before modification
        for s in self.stock_symbols:
            self.price_history[s].append(self.stock_prices[s])
            if len(self.price_history[s]) > 20: self.price_history[s].pop(0)

        # 2. Agents act (Traders trade, Macro agents shift indicators)
        self.schedule.step()
        
        # 3. Collect Data
        self.datacollector.collect(self)

# --- Execution & Visualization ---

def run_simulation():
    print("Initializing Autonomous World...")
    sim = AutonomousMarket()
    
    steps = 100
    print(f"Running for {steps} steps...")
    for _ in range(steps):
        sim.step()

    # Data Extraction
    model_df = sim.datacollector.get_model_vars_dataframe()
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Stock Prices
    plt.subplot(1, 2, 1)
    # We need to unpack the dictionary column 'Prices'
    price_data = pd.DataFrame(model_df["Prices"].tolist())
    for col in price_data.columns:
        plt.plot(price_data[col], label=col)
    plt.title("Stock Price Dynamics")
    plt.xlabel("Step")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Economic Indicators
    plt.subplot(1, 2, 2)
    econ_data = pd.DataFrame(model_df["Economics"].tolist())
    for col in econ_data.columns:
        plt.plot(econ_data[col], label=col, linestyle="--")
    plt.title("Macroeconomic Indicators")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Simulation Complete.")

if __name__ == "__main__":
    run_simulation()
