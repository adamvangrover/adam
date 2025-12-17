import asyncio
import logging
from core.agents.specialized.institutional_trend_agent import InstitutionalTrendAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Quarterly Trend Analysis Simulation...")

    # Mock Raw Data (Simulating 13F Summaries)
    raw_data = """
    Berkshire Hathaway: Sold 10M shares of AAPL. Bought 5M shares of OXY.
    Renaissance Technologies: Reduced position in PLTR by 20%. Increased position in GOOGL by 15%.
    Citadel: Increased Put options on SPY by 50%. Net short exposure increased.
    Baupost Group: Entered new position in gold miners.
    """

    # Initialize Agent
    config = {
        "agent_id": "trend_analyst_01",
        "description": "Quarterly 13F Trend Monitor"
    }

    # Check for Semantic Kernel (optional, for real execution)
    kernel = None
    try:
        from semantic_kernel import Kernel
        kernel = Kernel()
        # In a real scenario, we would configure the kernel with an AI service here.
    except ImportError:
        logger.warning("Semantic Kernel not installed, running in mock mode.")

    agent = InstitutionalTrendAgent(config, kernel=kernel)

    # Execute Analysis
    result = await agent.execute(raw_data=raw_data)

    # Output Result
    print("\n" + "="*40)
    print("FINAL REPORT")
    print("="*40)
    print(result["content"])
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
