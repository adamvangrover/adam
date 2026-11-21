with open("core/market_data/timescale/hypertable_manager.py", "r") as f:
    lines = f.readlines()
with open("core/market_data/timescale/hypertable_manager.py", "w") as f:
    for line in lines:
        if "        sql = f\"\"\"" in line:
            f.write("        _sql = f\"\"\"\n")
        elif "        SELECT create_hypertable" in line:
            f.write("        SELECT create_hypertable('{table_name}', '{time_column}', \\\n")
        elif "        partitioning_column =>" in line:
            f.write("            partitioning_column => '{symbol_column}', number_partitions => 4, if_not_exists => TRUE);\n")
        else:
            f.write(line)
