sed -i '/import json/d' core/devx/telemetry/microscopic_logger.py
sed -i '/import json/d' core/mcp/universal_mcp_socket.py
sed -i 's/from typing import Callable, Any, Dict, Optional/from typing import Callable, Optional/g' core/market_data/nats/jetstream_bus.py
sed -i 's/from typing import Dict, Any//g' core/market_data/timescale/hypertable_manager.py
sed -i 's/sql = f"""/# sql = f"""/g' core/market_data/timescale/hypertable_manager.py
sed -i '/from typing import Optional/d' core/mcp/universal_mcp_socket.py
sed -i 's/from typing import List/from typing import List/g' core/os_framework/coordination/process_scheduler.py
sed -i '/import os/d' core/os_framework/coordination/process_scheduler.py
sed -i '/from typing import List/d' core/os_framework/coordination/process_scheduler.py
sed -i 's/from typing import Dict, Optional/from typing import Optional/g' core/security/hardware/sgx_enclave.py
sed -i 's/SecurityError/RuntimeError/g' core/security/hardware/sgx_enclave.py
