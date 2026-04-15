with open("src/market_mayhem/scanners.py", "r") as f:
    lines = f.readlines()
with open("src/market_mayhem/scanners.py", "w") as f:
    for line in lines:
        if "    @classmethod" in line:
            f.write("\n")
        f.write(line.rstrip() + "\n")
