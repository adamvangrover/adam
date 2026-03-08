import time
import pandas as pd
import numpy as np

# Generate a large DataFrame
n = 10000
df = pd.DataFrame({
    'ticker_q0': ['AAPL'] * n,
    'ticker_q1': ['AAPL'] * n,
    'cusip': ['123456789'] * n,
    'value_q0': np.random.rand(n) * 1000,
    'share_type_q0': ['SH'] * n,
    'share_type_q1': ['SH'] * n,
    'issuer_q0': ['Apple'] * n,
    'issuer_q1': ['Apple'] * n,
    '_merge': ['left_only'] * (n // 2) + ['both'] * (n // 2),
    'shares_q0': np.random.randint(100, 1000, n),
    'shares_q1': np.random.randint(50, 500, n)
})

def method_iterrows():
    start = time.time()
    signals = []
    for _, row in df.iterrows():
        ticker = row.get('ticker_q0') or row.get('ticker_q1') or row['cusip']
        val_q0 = row.get('value_q0', 0)
        share_type = row.get('share_type_q0') or row.get('share_type_q1')
        issuer = row.get('issuer_q0') or row.get('issuer_q1')

        if row['_merge'] == 'left_only':
            desc = f"New Position: {issuer}"
            if share_type == 'PRN':
                desc += " (DEBT/CONVERTIBLE - HIGH CONVICTION)"
            signals.append({
                'ticker': str(ticker),
                'signal_type': "VULTURE_ENTRY",
                'change_pct': 100.0,
                'conviction_score': float(val_q0),
                'description': desc,
                'share_type': str(share_type)
            })
        elif row['_merge'] == 'both':
            shares_q0 = row['shares_q0']
            shares_q1 = row['shares_q1']
            if shares_q1 > 0:
                pct_change = ((shares_q0 - shares_q1) / shares_q1) * 100
                if pct_change > 20:
                    signals.append({
                        'ticker': str(ticker),
                        'signal_type': "ACCUMULATION",
                        'change_pct': pct_change,
                        'conviction_score': float(val_q0),
                        'description': f"Increased position by {pct_change:.1f}%",
                        'share_type': str(share_type)
                    })
    return time.time() - start

def method_vectorized():
    start = time.time()
    signals = []

    # Pre-calculate fields to avoid row-wise gets
    df['calc_ticker'] = df['ticker_q0'].combine_first(df['ticker_q1']).combine_first(df['cusip'])
    df['calc_val_q0'] = df['value_q0'].fillna(0)
    df['calc_share_type'] = df['share_type_q0'].combine_first(df['share_type_q1'])
    df['calc_issuer'] = df['issuer_q0'].combine_first(df['issuer_q1'])

    # Left only
    left_only = df[df['_merge'] == 'left_only']
    for row in left_only.to_dict('records'):
        desc = f"New Position: {row['calc_issuer']}"
        if row['calc_share_type'] == 'PRN':
            desc += " (DEBT/CONVERTIBLE - HIGH CONVICTION)"
        signals.append({
            'ticker': str(row['calc_ticker']),
            'signal_type': "VULTURE_ENTRY",
            'change_pct': 100.0,
            'conviction_score': float(row['calc_val_q0']),
            'description': desc,
            'share_type': str(row['calc_share_type'])
        })

    # Both
    both = df[(df['_merge'] == 'both') & (df['shares_q1'] > 0)].copy()
    both['pct_change'] = ((both['shares_q0'] - both['shares_q1']) / both['shares_q1']) * 100
    accum = both[both['pct_change'] > 20]
    for row in accum.to_dict('records'):
        signals.append({
            'ticker': str(row['calc_ticker']),
            'signal_type': "ACCUMULATION",
            'change_pct': row['pct_change'],
            'conviction_score': float(row['calc_val_q0']),
            'description': f"Increased position by {row['pct_change']:.1f}%",
            'share_type': str(row['calc_share_type'])
        })

    return time.time() - start

print(f"Iterrows: {method_iterrows():.4f}s")
print(f"Vectorized: {method_vectorized():.4f}s")
