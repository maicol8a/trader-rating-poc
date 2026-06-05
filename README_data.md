# Data Collection

The raw dataset (`traders_data.csv`) is not distributed in this repository
due to Binance Terms of Service restrictions on redistribution.

## Expected format

- Encoding: ISO-8859-1
- Delimiter: semicolon (`;`)
- Rows: one per trader (100 in the original study)
- Decimal separator: comma (`,`) for numeric fields
- Percentage fields: include `%` symbol

## Columns

| Column | Type | Description |
|--------|------|-------------|
| Nombre trader | str | Trader display name |
| ID | float | Trader internal ID |
| ROI | str | Return on investment (%) |
| Pnl | str | Net profit and loss (USD) |
| Sharpe Ratio | str | Risk-adjusted return ratio |
| MDD | str | Maximum drawdown (%) |
| Proporción de ganancias | str | Win rate — fraction of profitable trades (%) |
| Días con ganancias | float | Count of days with positive return |
| Pnl del copiador | str | Cumulative P&L earned by traders who copied this trader (USD) |
| Días de trading | float | Number of active trading days |

## Binance copy-trading API

Data can be collected from the Binance public copy-trading leaderboard:

```
Endpoint: https://www.binance.com/bapi/futures/v1/public/future/leaderboard/getLeaderboard
Method: POST
Payload: {"isShared": true, "periodType": "MONTHLY", "statisticsType": "ROI"}
```

See Binance documentation for current API specifications and rate limits.
The collection period for the original study was March–April 2025.

## Preprocessing pipeline

After collection, the pipeline in `src/01_preprocessing.py` applies:
1. Numeric parsing (comma → dot, % removal)
2. Missing value removal (N=100 → N=88, 12 records with missing Sharpe Ratio)
3. RobustScaler normalisation (median/IQR, justified by non-normality)
4. Mahalanobis outlier removal at p97.5 threshold (N=88 → N=85)
