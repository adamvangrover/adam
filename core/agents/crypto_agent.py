# core/agents/crypto_agent.py

import json
import datetime
import requests
import numpy as np
from web3 import Web3
from web3.middleware import geth_poa_middleware
from sklearn.linear_model import LinearRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import talib
import ccxt
from pycoingecko import CoinGeckoAPI
import time
import os
from collections import deque

# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')

class CryptoAgent:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json", web3_provider_uri=None, exchange='binance', wallet_address=None, wallet_private_key=None, price_history_length=100):
        """
        Initializes the Crypto Agent.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
            web3_provider_uri (str, optional): URI for the Web3 provider.
            exchange (str): The default exchange for trading.
            wallet_address (str, optional): Wallet address to interact with Web3.
            wallet_private_key (str, optional): Private key for wallet access.
            price_history_length (int): The length of the price history to store in deque.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()
        self.web3_provider_uri = web3_provider_uri
        self.exchange = exchange
        self.wallet_address = wallet_address
        self.wallet_private_key = wallet_private_key

        # Setup Web3 connection (supports Ethereum, Binance Smart Chain, etc.)
        self.web3 = Web3(Web3.HTTPProvider(self.web3_provider_uri))
        if self.web3.isConnected():
            print("Web3 Connected")
            if self.web3.eth.chain_id == 3:  # Ropsten testnet example
                self.web3.middleware_stack.inject(geth_poa_middleware, layer=0)
        else:
            print("Web3 not connected")
        
        # Wallet setup
        if self.wallet_address and self.wallet_private_key:
            self.account = self.web3.eth.account.privateKeyToAccount(self.wallet_private_key)
        
        # Initialize Sentiment Analysis
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Setup Exchange (for Trading)
        self.exchange_client = ccxt.binance()  # Example with Binance, extendable to other exchanges

        # Setup CoinGecko API for Coin Data
        self.cg = CoinGeckoAPI()

        # Initialize the Uniswap V3 router contract ABI for token swaps
        self.uniswap_v3_router_abi = self.get_uniswap_v3_router_abi()

        # Initialize deque for storing price history
        self.price_history = deque(maxlen=price_history_length)

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.

        Returns:
            dict: The knowledge base data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    def get_uniswap_v3_router_abi(self):
        """
        Fetches the ABI for the Uniswap V3 router. Replace with actual ABI.
        """
        return json.loads('[...]')  # Replace with actual ABI for Uniswap V3

    def analyze_crypto_market(self, symbol):
        """
        Analyzes a specific cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            dict: Analysis results, including price prediction, risk assessment,
                  on-chain analysis, and sentiment analysis.
        """
        price_prediction = self.predict_price(symbol)
        risk_assessment = self.assess_risk(symbol)
        on_chain_analysis = self.analyze_on_chain_metrics(symbol)
        sentiment_analysis = self.get_social_media_sentiment(symbol)

        analysis_results = {
            "price_prediction": price_prediction,
            "risk_assessment": risk_assessment,
            "on_chain_analysis": on_chain_analysis,
            "sentiment_analysis": sentiment_analysis
        }

        return analysis_results

    def predict_price(self, symbol):
        """
        Predicts the future price of a cryptocurrency using linear regression.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            float: The predicted price.
        """
        historical_data = self.get_historical_data(symbol)
        if not historical_data:
            return None

        # Append the latest price to the deque
        for data in historical_data:
            self.price_history.append(data["price_close"])

        X = [[datetime.datetime.strptime(d["date"], "%Y-%m-%d").timestamp()] for d in historical_data]
        y = [d["price_close"] for d in historical_data]

        model = LinearRegression()
        model.fit(X, y)

        # Predict the price for 7 days from now
        future_date = (datetime.datetime.now() + datetime.timedelta(days=7)).timestamp()
        predicted_price = model.predict([[future_date]])

        return predicted_price[0]

    def assess_risk(self, symbol):
        """
        Assesses the risk associated with a cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            str: The risk assessment (e.g., "High Risk", "Medium Risk", "Low Risk").
        """
        historical_data = self.get_historical_data(symbol)
        if not historical_data:
            return "Unknown Risk"

        # Update the deque with latest data
        prices = [d["price_close"] for d in historical_data]
        for price in prices:
            self.price_history.append(price)

        volatility = self.calculate_volatility(list(self.price_history))

        if volatility > 0.2:
            return "High Risk"
        elif volatility > 0.1:
            return "Medium Risk"
        else:
            return "Low Risk"

    def calculate_volatility(self, prices):
        """
        Calculates the volatility of a price series.

        Args:
            prices (list): A list of prices.

        Returns:
            float: The volatility.
        """
        return np.std(prices) / np.mean(prices)  # Placeholder for volatility calculation

    def get_historical_data(self, symbol):
        """
        Retrieves historical data for a cryptocurrency from the knowledge base.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            list: Historical data for the cryptocurrency.
        """
        return self.knowledge_base.get("cryptocurrencies", {}).get(symbol, {}).get("historical_data", [])

    def analyze_on_chain_metrics(self, symbol):
        """
        Analyzes on-chain metrics for a cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            dict: On-chain analysis results.
        """
        on_chain_data = self.get_on_chain_data(symbol)
        if not on_chain_data:
            return {}

        analysis_results = {
            "transaction_volume": on_chain_data.get("transaction_volume", None),
            "active_addresses": on_chain_data.get("active_addresses", None),
        }

        return analysis_results

    def get_on_chain_data(self, symbol):
        """
        Retrieves on-chain data for a cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            dict: On-chain data for the cryptocurrency.
        """
        return self.knowledge_base.get("cryptocurrencies", {}).get(symbol, {}).get("on_chain_data", {})

    def get_social_media_sentiment(self, symbol):
        """
        Analyzes social media sentiment for a cryptocurrency.

        Args:
            symbol (str): The symbol of the cryptocurrency.

        Returns:
            dict: Sentiment analysis results, including overall sentiment
                  score and sentiment breakdown (positive, negative, neutral).
        """
        news_data = self.cg.get_coin_news(id=symbol)
        sentiment_score = 0
        for article in news_data:
            text = article['title'] + " " + article['description']
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            sentiment_score += sentiment['compound']

        sentiment_results = {
            "overall_sentiment": sentiment_score / len(news_data) if news_data else 0,
            "sentiment_breakdown": {
                "positive": sentiment_score / len(news_data) if sentiment_score > 0 else 0,
                "negative": sentiment_score / len(news_data) if sentiment_score < 0 else 0,
                "neutral": 1 - (sentiment_score / len(news_data)) if not sentiment_score else 0,
            }
        }

        return sentiment_results

    def trade_decision(self, symbol, strategy='macd'):
        """
        Makes a trading decision based on chosen strategy.

        Args:
            symbol (str): The symbol of the cryptocurrency.
            strategy (str): Trading strategy to use ('macd', 'sentiment', etc.).

        Returns:
            str: The decision to 'buy', 'sell', or 'hold'.
        """
        if strategy == 'macd':
            return self.moving_average_crossover(symbol)
        elif strategy == 'sentiment':
            sentiment = self.get_social_media_sentiment(symbol)
            if sentiment["overall_sentiment"] > 0:
                return 'buy'
            else:
                return 'sell'
        elif strategy == 'volatility':
            risk = self.assess_risk(symbol)
            if risk == 'Low Risk':
                return 'buy'
            else:
                return 'sell'
        else:
            return 'hold'

    def moving_average_crossover(self, symbol, short_period=12, long_period=26):
        """
        Implement the MACD and Moving Average Crossover strategy.

        Args:
            symbol (str): The symbol of the cryptocurrency.
            short_period (int): The short period for the moving average.
            long_period (int): The long period for the moving average.

        Returns:
            str: 'buy', 'sell', or 'hold' based on the MACD strategy.
        """
        historical_prices = self.get_historical_data(symbol)
        if not historical_prices:
            return 'hold'

        prices = [d["price_close"] for d in historical_prices]
        short_term_ma = talib.SMA(np.array(prices), timeperiod=short_period)
        long_term_ma = talib.SMA(np.array(prices), timeperiod=long_period)

        if short_term_ma[-1] > long_term_ma[-1]:
            return 'buy'
        elif short_term_ma[-1] < long_term_ma[-1]:
            return 'sell'
        return 'hold'

    def execute_trade(self, symbol, action, amount):
        """
        Executes the trade on the connected exchange.

        Args:
            symbol (str): The symbol of the cryptocurrency (e.g., 'BTC/USDT').
            action (str): 'buy' or 'sell'.
            amount (float): Amount to trade.

        Returns:
            dict: The result of the trade execution.
        """
        if action == 'buy':
            order = self.exchange_client.create_market_buy_order(symbol, amount)
        elif action == 'sell':
            order = self.exchange_client.create_market_sell_order(symbol, amount)
        else:
            return {'error': 'Invalid action'}

        return order

    def create_smart_contract(self, contract_code, **kwargs):
        """
        Generates a smart contract based on a provided template and kwargs.

        Args:
            contract_code (str): Smart contract template code (Solidity, Vyper, etc.).
            **kwargs: Parameters to insert into the template.

        Returns:
            str: The finalized contract code.
        """
        return contract_code.format(**kwargs)

    def deploy_smart_contract(self, contract_code, gas_price=20000000000, gas_limit=5000000):
        """
        Deploys a smart contract to the blockchain using the connected Web3 instance.

        Args:
            contract_code (str): The smart contract's compiled bytecode.
            gas_price (int): Gas price for the deployment.
            gas_limit (int): Gas limit for the deployment.

        Returns:
            str: Transaction hash of the deployment.
        """
        if not self.wallet_address or not self.wallet_private_key:
            raise ValueError("Wallet address or private key not provided.")

        compiled_contract = self.web3.eth.contract(abi=contract_code['abi'], bytecode=contract_code['bytecode'])

        transaction = {
            'from': self.wallet_address,
            'gas': gas_limit,
            'gasPrice': gas_price,
            'data': compiled_contract.bytecode,
        }

        signed_txn = self.web3.eth.account.sign_transaction(transaction, self.wallet_private_key)
        tx_hash = self.web3.eth.sendRawTransaction(signed_txn.rawTransaction)
        
        return f"Transaction Hash: {tx_hash.hex()}"


