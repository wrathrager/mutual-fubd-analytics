import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 1. Load your existing data
df = pd.read_excel('PRICES.xlsx', sheet_name='Sheet1') # Ensure this is your uploaded PRICES.xlsx - Sheet1.csv
df.columns = [str(column).strip() for column in df.columns]

date_column = next(
    (column for column in df.columns if column.strip().lower() == 'date'),
    None,
)
if date_column is None:
    raise KeyError(f"Could not find a date column. Available columns: {list(df.columns)}")

df = df.rename(columns={date_column: 'Date'})
df['Date'] = pd.to_datetime(df['Date'])

# 2. Define the anchor date (01-07-2026)
anchor_date = pd.to_datetime('2026-07-01')


def get_price_row_on_or_before(df, target_date):
    eligible = df.loc[df['Date'] <= target_date].sort_values('Date')
    if eligible.empty:
        return None
    return eligible.iloc[-1]

def calculate_lookback_returns(df, anchor):
    results = []

    anchor_row = get_price_row_on_or_before(df, anchor)
    if anchor_row is None:
        raise ValueError(f'No price data available on or before anchor date {anchor.date()}')

    actual_anchor_date = anchor_row['Date']

    # Calculate aggregated lookback returns for k=1 to 12 months back
    for k in range(1, 13):
        target_date = anchor - relativedelta(months=k)
        target_row = get_price_row_on_or_before(df, target_date)
        if target_row is None:
            continue

        lookback_label = f'{k}M'

        for col in df.columns:
            if col == 'Date':
                continue

            price_anchor = anchor_row[col]
            price_target = target_row[col]

            if pd.isna(price_anchor) or pd.isna(price_target) or price_target == 0:
                continue

            ret = ((price_anchor - price_target) / price_target) * 100

            results.append({
                'Stock': col,
                'Lookback': lookback_label,
                'Return': ret,
                'AnchorDateUsed': actual_anchor_date,
                'TargetDateUsed': target_row['Date'],
            })

    return pd.DataFrame(results)

# 3. Execution
returns_df = calculate_lookback_returns(df, anchor_date)

# 4. Ranking
returns_df['Rank'] = returns_df.groupby('Lookback')['Return'].rank(ascending=False)

# 5. Output
returns_df.to_csv('stock_performance_ranks1.csv', index=False)

# Show Top/Worst for a few timelines
for k in [1, 6, 12]:
    subset = returns_df[returns_df['Lookback'] == f'{k}M'].sort_values('Rank')
    print(f"\n--- {k} Month Lookback ---")
    print("Best 5:", subset.head(5)['Stock'].values)
    print("Worst 5:", subset.tail(5)['Stock'].values)
