import time
import pandas as pd
import numpy as np

idx = pd.date_range("2024-01-01", periods=1000)
df = pd.DataFrame(np.random.rand(1000, 5), index=idx, columns=["Open", "High", "Low", "Close", "Volume"])

start = time.time()
res1 = []
for index, row in df.iterrows():
    res1.append({
        "timestamp": index.isoformat(),
        "open": row["Open"],
        "high": row["High"],
        "low": row["Low"],
        "close": row["Close"],
        "volume": row["Volume"]
    })
t1 = time.time() - start

start = time.time()
df2 = df.copy()
df2 = df2.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
df2.index = df2.index.map(lambda x: x.isoformat())
res2 = df2.reset_index(names="timestamp")[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
t2 = time.time() - start

print(f"Iterrows: {t1:.4f}s")
print(f"Vectorized: {t2:.4f}s")
print(f"Speedup: {t1/t2:.1f}x")
print(f"Same output structure? {res1[0] == res2[0]}")
