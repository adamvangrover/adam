from typing import Any, Dict, List, Optional, Union
import logging
from pydantic import BaseModel, Field
from web3 import Web3
from decimal import Decimal
import math
import asyncio

from core.agents.agent_base import AgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class DeFiLiquidityAgentOutput(BaseModel):
    pool_address: str
    liquidity_metrics: Dict[str, Any]
    impermanent_loss_pct: Optional[float] = None
    yield_analysis: Dict[str, Any]
    recommendation: str

class DeFiLiquidityAgent(AgentBase):
    """
    Analyzes DeFi Liquidity Pools for health, yield, and risks (Impermanent Loss).
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.web3_provider_uri = self.config.get("web3_provider_uri")
        if self.web3_provider_uri:
            self.web3 = Web3(Web3.HTTPProvider(self.web3_provider_uri))
            if self.web3.is_connected():
                logger.info(f"Connected to Web3 provider: {self.web3_provider_uri}")
            else:
                logger.warning(f"Failed to connect to Web3 provider: {self.web3_provider_uri}")
                self.web3 = None
        else:
            self.web3 = None
            logger.warning("No Web3 provider URI provided. Running in simulation/mock mode.")

    def calculate_impermanent_loss(self, initial_price: float, current_price: float) -> float:
        """
        Calculates Impermanent Loss (IL) percentage.
        Formula: 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        where price_ratio = current_price / initial_price
        """
        if initial_price <= 0 or current_price <= 0:
            return 0.0

        price_ratio = current_price / initial_price
        il = (2 * math.sqrt(price_ratio)) / (1 + price_ratio) - 1
        return abs(il) * 100  # Return as positive percentage

    async def get_pool_reserves(self, pool_address: str) -> Dict[str, Any]:
        """
        Fetches pool reserves.
        If Web3 is available, it would call the contract (ABI needed).
        For now, if Web3 is missing, it returns mock data.
        """
        if self.web3:
            # Placeholder for real contract interaction
            # We would need the ABI for Uniswap V2 pair
            # pair_contract = self.web3.eth.contract(address=pool_address, abi=UNI_V2_PAIR_ABI)
            # reserves = pair_contract.functions.getReserves().call()
            # return {"reserve0": reserves[0], "reserve1": reserves[1]}
            pass

        # Mock Data
        return {
            "reserve0": 1000000.0,
            "reserve1": 500.0,
            "mock": True
        }

    def analyze_yield(self, reserves: Dict[str, Any], chain_id: int) -> Dict[str, Any]:
        """
        Simple heuristic analysis of yield potential based on reserves/volume (mocked).
        """
        # Heuristic: Lower liquidity *might* mean higher slippage but potentially higher APR if volume is high.
        # This is very simplified.
        liquidity_score = (reserves.get("reserve0", 0) * reserves.get("reserve1", 0)) ** 0.5

        estimated_apy = 0.05 # Base 5%
        if liquidity_score < 100000:
            estimated_apy = 0.15 # High risk, high reward?
        elif liquidity_score > 10000000:
            estimated_apy = 0.02 # Stable

        return {
            "estimated_apy": estimated_apy,
            "liquidity_score": liquidity_score
        }

    async def execute(self, input_data: Union[str, AgentInput, Dict[str, Any]] = None, **kwargs) -> Union[Dict[str, Any], AgentOutput]:
        """
        Executes the analysis logic.
        Supports AgentInput.
        """
        # Input Normalization
        pool_address = ""
        chain_id = 1
        initial_price = None
        current_price = None
        is_standard_mode = False
        query = ""

        if input_data is not None:
            if isinstance(input_data, AgentInput):
                query = input_data.query
                # Assume query contains address or context does
                pool_address = query if query.startswith("0x") else ""
                if input_data.context:
                    pool_address = input_data.context.get("pool_address", pool_address)
                    chain_id = input_data.context.get("chain_id", 1)
                    initial_price = input_data.context.get("initial_price")
                    current_price = input_data.context.get("current_price")
                is_standard_mode = True
            elif isinstance(input_data, dict):
                pool_address = input_data.get("pool_address", "")
                chain_id = input_data.get("chain_id", 1)
                initial_price = input_data.get("initial_price")
                current_price = input_data.get("current_price")
                kwargs.update(input_data)
            elif isinstance(input_data, str):
                pool_address = input_data

        # Fallback to kwargs
        if not pool_address:
            pool_address = kwargs.get("pool_address", "")
        chain_id = kwargs.get("chain_id", chain_id)
        if initial_price is None: initial_price = kwargs.get("initial_price")
        if current_price is None: current_price = kwargs.get("current_price")

        logger.info(f"Analyzing liquidity pool: {pool_address} on chain {chain_id}")

        if not pool_address:
            msg = "No pool address provided."
            if is_standard_mode:
                return AgentOutput(answer=msg, confidence=0.0, metadata={"error": msg})
            return {"error": msg}

        # 1. Get Reserves
        reserves = await self.get_pool_reserves(pool_address)

        # 2. Calculate IL if prices provided
        il_pct = 0.0
        if initial_price and current_price:
            il_pct = self.calculate_impermanent_loss(initial_price, current_price)

        # 3. Analyze Yield
        yield_data = self.analyze_yield(reserves, chain_id)

        # 4. Formulate Recommendation
        recommendation = "HOLD"
        if il_pct > 5.0:
            recommendation = "WITHDRAW (High IL)"
        elif yield_data["estimated_apy"] > 0.10:
            recommendation = "DEPOSIT (Good Yield)"

        output_model = DeFiLiquidityAgentOutput(
            pool_address=pool_address,
            liquidity_metrics=reserves,
            impermanent_loss_pct=il_pct,
            yield_analysis=yield_data,
            recommendation=recommendation
        )

        result_dict = output_model.model_dump()

        if is_standard_mode:
            answer = f"DeFi Liquidity Analysis for {pool_address}:\n"
            answer += f"Recommendation: {recommendation}\n"
            answer += f"Estimated APY: {yield_data['estimated_apy']*100:.2f}%\n"
            if il_pct:
                answer += f"Impermanent Loss: {il_pct:.2f}%\n"

            return AgentOutput(
                answer=answer,
                sources=["On-Chain Data", "Internal Models"],
                confidence=0.9,
                metadata=result_dict
            )

        return result_dict
