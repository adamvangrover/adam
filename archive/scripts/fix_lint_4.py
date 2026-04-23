with open("core/market_data/historical_loader.py", "r") as f:
    lines = f.readlines()
with open("core/market_data/historical_loader.py", "w") as f:
    for line in lines:
        if not ("import typing" in line or "from typing" in line):
            f.write(line)

with open("core/market_data/service.py", "r") as f:
    lines = f.readlines()
with open("core/market_data/service.py", "w") as f:
    for line in lines:
        if not "typing.Union" in line and not "from typing import Union" in line:
            f.write(line)

with open("core/market_data/timescale/hypertable_manager.py", "r") as f:
    lines = f.readlines()
with open("core/market_data/timescale/hypertable_manager.py", "w") as f:
    for line in lines:
        if "SELECT create_hypertable" in line:
            f.write("        SELECT create_hypertable('{table_name}', '{time_column}', \\\n")
            f.write("        partitioning_column => '{symbol_column}', number_partitions => 4, if_not_exists => TRUE);\n")
        elif "Created/Verified hypertable:" in line:
            f.write("        logger.info(\n")
            f.write("            f\"Created/Verified hypertable: {table_name} partitioned by {time_column} and {symbol_column}.\"\n")
            f.write("        )\n")
        else:
            f.write(line)
