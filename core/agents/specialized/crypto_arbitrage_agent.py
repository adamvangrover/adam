
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available. CryptoArbitrageAgent will operate in mock mode.")

from core.agents.agent_base import AgentBase

# Pydantic Models
class ArbitrageOpportunity(BaseModel):
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_percentage: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    estimated_profit: Optional[float] = None

class ArbitrageRequest(BaseModel):
    symbols: List[str] = Field(default=["BTC/USDT", "ETH/USDT"])
    exchanges: List[str] = Field(default=["binance", "kraken"])
    min_spread: float = 0.5  # Minimum spread in percentage

class CryptoArbitrageAgent(AgentBase):
    """
    A specialized agent that monitors cryptocurrency prices across multiple exchanges
    to identify arbitrage opportunities.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.exchanges: Dict[str, Any] = {}
        self.initialized_exchanges = False

    async def _init_exchanges(self, exchange_ids: List[str]):
        """Initializes CCXT exchange instances."""
        if not CCXT_AVAILABLE:
            return

        for ex_id in exchange_ids:
            if ex_id not in self.exchanges:
                try:
                    exchange_class = getattr(ccxt, ex_id)
                    self.exchanges[ex_id] = exchange_class()
                    # await self.exchanges[ex_id].load_markets() # Optional: load markets
                except Exception as e:
                    logging.error(f"Failed to initialize exchange {ex_id}: {e}")

        self.initialized_exchanges = True

    async def _close_exchanges(self):
        """Closes exchange connections."""
        if not CCXT_AVAILABLE:
            return
        for exchange in self.exchanges.values():
            await exchange.close()
        self.exchanges.clear()

    async def _fetch_price(self, exchange_id: str, symbol: str) -> Optional[float]:
        """Fetches the latest ticker price for a symbol on an exchange."""
        if not CCXT_AVAILABLE:
            # Mock data for testing/fallback
            import random
            base_price = 50000 if "BTC" in symbol else 3000
            variance = random.uniform(-0.01, 0.01)
            return base_price * (1 + variance)

        if exchange_id not in self.exchanges:
            return None

        try:
            exchange = self.exchanges[exchange_id]
            ticker = await exchange.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            logging.warning(f"Error fetching {symbol} from {exchange_id}: {e}")
            return None

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes the arbitrage scan.

        Args:
            symbols (List[str]): List of trading pairs (e.g. ['BTC/USDT']).
            exchanges (List[str]): List of exchange IDs (e.g. ['binance', 'kraken']).
            min_spread (float): Minimum spread % to report.

        Returns:
            Dict containing found opportunities.
        """
        # Parse inputs using Pydantic
        request_data = kwargs.copy()
        # Handle case where args might pass a dict or object
        if args and isinstance(args[0], dict):
            request_data.update(args[0])

        try:
            request = ArbitrageRequest(**request_data)
        except Exception as e:
            logging.error(f"Invalid input for CryptoArbitrageAgent: {e}")
            return {"error": str(e)}

        await self._init_exchanges(request.exchanges)

        opportunities: List[ArbitrageOpportunity] = []

        tasks = []
        # Create a grid of fetch tasks: (exchange, symbol)
        for symbol in request.symbols:
            for exchange_id in request.exchanges:
                tasks.append(self._fetch_price_wrapper(exchange_id, symbol))

        results = await asyncio.gather(*tasks)

        # Organize prices: prices[symbol][exchange] = price
        prices: Dict[str, Dict[str, float]] = {s: {} for s in request.symbols}

        for (exchange_id, symbol, price) in results:
            if price is not None:
                prices[symbol][exchange_id] = price

        # Analyze for arbitrage
        for symbol, exchange_prices in prices.items():
            if len(exchange_prices) < 2:
                continue

            # Find min and max
            sorted_prices = sorted(exchange_prices.items(), key=lambda x: x[1])
            min_ex, min_price = sorted_prices[0]
            max_ex, max_price = sorted_prices[-1]

            if min_price > 0:
                spread = ((max_price - min_price) / min_price) * 100
                if spread >= request.min_spread:
                    opp = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=min_ex,
                        sell_exchange=max_ex,
                        buy_price=min_price,
                        sell_price=max_price,
                        spread_percentage=spread,
                        estimated_profit=(max_price - min_price)
                    )
                    opportunities.append(opp)

        await self._close_exchanges()

        return {
            "opportunities": [opp.model_dump() for opp in opportunities],
            "scan_timestamp": datetime.utcnow().isoformat(),
            "status": "success" if opportunities else "no_opportunities_found"
        }

    async def _fetch_price_wrapper(self, exchange_id: str, symbol: str):
        """Helper to unpack results for gather."""
        price = await self._fetch_price(exchange_id, symbol)
        return (exchange_id, symbol, price)
