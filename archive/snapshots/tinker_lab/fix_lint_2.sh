sed -i 's/from typing import Dict, Any//g' core/market_data/timescale/hypertable_manager.py
sed -i 's/import logging/import logging/g' core/market_data/historical_loader.py
sed -i 's/from typing import Optional//g' core/market_data/historical_loader.py
sed -i 's/from typing import Union//g' core/market_data/service.py
sed -i 's/from typing import Callable, Dict, Any, Optional/from typing import Callable, Dict, Any/g' core/mcp/universal_mcp_socket.py
