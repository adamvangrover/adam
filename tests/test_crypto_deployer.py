
import asyncio
import pytest
from core.trading.crypto.deployer import CryptoDeployer, DeploymentRequest, Network, DeploymentType

@pytest.mark.asyncio
async def test_crypto_deployer():
    deployer = CryptoDeployer()

    # Test Trading Bot Deployment
    req_bot = DeploymentRequest(
        network=Network.ETHEREUM,
        type=DeploymentType.TRADING_BOT,
        config={"symbol": "BTC/USD", "initial_balance": 50000},
        deployer_id="admin"
    )

    result_bot = await deployer.deploy(req_bot)
    assert result_bot.success
    assert result_bot.deployment_id.startswith("bot-")
    assert "BTC/USD" in result_bot.message

    # Test Smart Contract Deployment
    req_contract = DeploymentRequest(
        network=Network.BASE,
        type=DeploymentType.SMART_CONTRACT,
        config={"bytecode": "0x123..."},
        deployer_id="dev"
    )

    result_contract = await deployer.deploy(req_contract)
    assert result_contract.success
    assert result_contract.tx_hash.startswith("0x")

if __name__ == "__main__":
    asyncio.run(test_crypto_deployer())
