import pandas as pd
import numpy as np
import time

# Create a large dummy dataframe
N = 100000
df = pd.DataFrame({
    'shares_prev': np.random.randint(0, 100, N),
    'shares_curr': np.random.randint(0, 100, N),
})
df['share_change'] = df['shares_curr'] - df['shares_prev']

def determine_action(row):
    if row['shares_prev'] == 0 and row['shares_curr'] > 0:
        return 'NEW'
    elif row['shares_curr'] == 0 and row['shares_prev'] > 0:
        return 'EXIT'
    elif row['share_change'] > 0:
        return 'ADD'
    elif row['share_change'] < 0:
        return 'REDUCE'
    else:
        return 'HOLD'

start = time.time()
r1 = df.apply(determine_action, axis=1)
t_apply = time.time() - start

start = time.time()
conditions = [
    (df['shares_prev'] == 0) & (df['shares_curr'] > 0),
    (df['shares_curr'] == 0) & (df['shares_prev'] > 0),
    (df['share_change'] > 0),
    (df['share_change'] < 0)
]
choices = ['NEW', 'EXIT', 'ADD', 'REDUCE']
r2 = np.select(conditions, choices, default='HOLD')
t_np = time.time() - start

print(f"Apply: {t_apply:.4f}s")
print(f"NP Select: {t_np:.4f}s")
print(f"Speedup: {t_apply / t_np:.1f}x")
