import pandas as pd
from statsmodels.tsa.stattools import coint

df = pd.read_csv('COMPLETE_DATASET.csv', parse_dates=['date']).set_index('date')

pairs = [
    ('KB',  'RF'),
    ('KB',  'KEY'),
    ('SAN', 'DB'),
    ('PNC', 'SCHW'),
    ('KEY', 'HBAN'),
    ('ALLY','DB'),
    ('CMA', 'TFC'),
]

# 3) Engle-Granger-Test
for a, b in pairs:
    series_a = df[f'close_{a}']
    series_b = df[f'close_{b}']
    coint_t, p_value, crit_vals = coint(series_a, series_b)
    print(f"\nPair {a}-{b}:")
    print(f"  Engle-Granger t-Stat: {coint_t:.4f}")
    print(f"  p-Wert             : {p_value:.4f}")
    print(f"  kritische Werte    : 1% {crit_vals[0]:.4f}, 5% {crit_vals[1]:.4f}, 10% {crit_vals[2]:.4f}")
