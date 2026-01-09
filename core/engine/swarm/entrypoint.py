
import asyncio
from core.engine.swarm.hive_mind import HiveMind

async def main():
    hm = HiveMind(worker_count=20)
    await hm.initialize()
    print('HiveMind Node Active')
    try:
        while True:
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("Shutting down...")
    finally:
        await hm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
