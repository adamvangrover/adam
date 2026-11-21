sed -i '/import logging/d' core/market_data/historical_loader.py
sed -i '/from typing import Optional/d' core/market_data/historical_loader.py
sed -i '/from typing import Union/d' core/market_data/service.py
sed -i 's/155 > 120/155/g' core/market_data/timescale/hypertable_manager.py
