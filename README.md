# Vietnam Market Intelligence Radar

A Python project for researching unusual trading activity in Vietnamese stocks, with particular relevance to VietinBank Securities (VBSE). The planned workflow collects historical market data, prepares features, detects anomalies, compares results with statistical baselines, and presents findings in an interactive dashboard.

**Current status:** Task 003 adds an offline financial feature table to the daily OHLCV pipeline. Model training and dashboards are future work.

## Planned capabilities

- Historical Vietnamese stock data through Vnstock Generation 5
- Data cleaning and feature engineering
- Isolation Forest anomaly detection and statistical baseline evaluation
- Streamlit charts, anomaly explanations, and sector filters
- Reproducible research workflows suitable for a public repository

## Setup on macOS

Use Python 3.13 (the project's target is Python 3.13.2):

```sh
python3.13 --version
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
python -c "import market_radar, vnstock; print(market_radar.__version__, vnstock.__version__)"
```

`requirements.txt` configures [Vnstock's official Generation 5 package index](https://vnstocks.com/docs/vnstock/cai-dat). The Vnstock dependency is pinned to 5.1.0; the installation page still shows 5.0.1 in examples, but the package index now marks it as yanked. A plain `pip install vnstock` or `pip install -e .` without that index may select an older PyPI release or fail. The requirements file installs this package in editable mode with development dependencies.

Vnstock 5 can be used at guest level. For future authenticated data access, copy `.env.example` to `.env`, set `VNSTOCK_API_KEY`, and load it in your shell or application configuration. `.env` is ignored by Git. No code reads this setting yet.

## Download historical data

From the repository root with the virtual environment active:

```sh
# Default: the ten stocks below plus VNINDEX, about three calendar years requested.
python -m market_radar.download_data

# Selected stocks (VNINDEX is always included):
python -m market_radar.download_data --symbols FPT SSI --start 2025-01-01 --end 2025-12-31

# Ignore matching local caches and query the provider again:
python -m market_radar.download_data --refresh
```

Symbols can also be comma separated, for example `--symbols FPT,SSI`. Dates must use `YYYY-MM-DD`; the end date cannot be later than the latest completed Vietnam weekday. The default end date is the latest weekday whose 16:00 Vietnam time cutoff has passed. Exchange holidays are handled by the provider's returned sessions. The default start is 1,095 calendar days earlier. The pipeline requests daily bars sequentially, pauses seven seconds between requests to stay below the observed guest limit of ten calls per minute, retries transient network and minute-limit errors up to three times, and exits nonzero if any asset fails or the provider explicitly truncates the requested history.

The initial stocks are FPT (Technology), HPG (Steel), MWG (Retail), VNM (Consumer goods), CTG (Banking), CTS (Securities), VCB (Banking), SSI (Securities), GAS (Energy), and VHM (Real estate). Unknown custom stock symbols have sector `Unclassified`. Invalid or unavailable symbols are reported as failures. The benchmark is `VNINDEX` (index). The universe mapping is in `src/market_radar/ingestion/universe.py`.

**Guest history limit:** On 2026-10-18, Vnstock 5.1.0 returned only 365 days of FPT history for a three-year request and marked the response `tier_cap_history_days=365`. This occurred with both documented Community sources, VCI and KBS. The [current Vnstock access limits](https://www.vnstocks.com/docs/vnstock/tham-khao/cap-quyen-va-gioi-han) list 365 days for guests and 2,920 days for registered Community users. The pipeline saves those real rows, records the cap in metadata, labels the result `PARTIAL`, and exits nonzero. It does not fill missing history or require an API key. A complete three-year equity dataset cannot currently be obtained through the unauthenticated route; this is a source limit, not a pipeline error.

### Files and schema

Provider response CSVs are stored in `data/raw/`. Validated, normalized CSVs and provenance sidecars are stored in `data/processed/`. Filenames include asset type, symbol, and requested start/end dates; rerunning the same window uses the cache unless `--refresh` is supplied. Both directories are ignored by Git.

The normalized CSV columns are `date,symbol,asset_type,sector,open,high,low,close,volume,price_unit`. Dates are Vietnam market dates (`YYYY-MM-DD`), prices are numeric, and volume is an integer. Stock prices remain in **thousands of VND**; VNINDEX prices remain in **index points**. Exact duplicate rows are removed; conflicting same-date bars, missing values, nonpositive prices, negative volume, and inconsistent OHLC values cause a clear failure. No artificial rows are made for weekends, holidays, suspended trading, or unavailable history.

Each `.meta.json` sidecar records the Vnstock version, actual source, UTC retrieval time, requested window, first and last observed dates, row counts, duplicate count, unit, adjustment convention, and any tier cap. To inspect a file:

```sh
head -n 5 data/processed/equity_FPT_*.csv
python -c "import pandas as pd; from pathlib import Path; p = sorted(Path('data/processed').glob('equity_FPT_*.csv'))[-1]; print(pd.read_csv(p).describe(include='all'))"
```

### Data provenance and limitations

The pipeline uses the [Vnstock Generation 5 Market API](https://www.vnstocks.com/docs/vnstock/thi-truong-market), with `Market().equity(symbol).ohlcv()` and `Market().index("VNINDEX").ohlcv()`. It explicitly requests the documented VCI source, available to Community and guest users, for both asset types. Vnstock is the technical connector; its metadata identifies the underlying provider. The documented equity OHLCV prices are adjusted for corporate actions, and adjustment methods can differ by source. Historical prices may change after provider revisions. Review the underlying provider's terms before redistributing market data; this repository intentionally excludes downloaded CSVs.

## Build financial features offline

The feature builder reads existing validated CSVs and makes **no network requests**:

```sh
python -m market_radar.build_features
python -m market_radar.build_features --symbols FPT SSI --output data/features/fpt_ssi_features.csv
```

The default output is `data/features/market_features.csv`, with `market_features.meta.json` alongside it. The metadata lists the selected input CSVs, their SHA-256 hashes and retrieval provenance, the feature columns, and row counts. Generated feature files are ignored by Git. If a symbol has multiple cached windows, the builder chooses the most recent last observation, then the longest observed history. It always loads VNINDEX for date matching, but the output contains stock rows only.

Each row retains `date`, `symbol`, `asset_type`, `sector`, original normalized OHLCV, `price_unit`, the features below, and a boolean `model_ready`. **The ten columns below are the complete numerical input list for Task 004.** Symbols and sectors remain identifiers, not numeric predictors.

| Feature | Definition on trading session T | Unit |
| --- | --- | --- |
| `return_1d` | `close(T) / close(T-1) - 1` | fraction |
| `return_5d` | `close(T) / close(T-5) - 1` | fraction |
| `intraday_range_pct` | `100 × (high(T) - low(T)) / open(T)` | percent |
| `price_vs_prev_ma20_pct` | `100 × (close(T) / mean(close(T-20..T-1)) - 1)` | percent |
| `volume_ratio_prev20` | `volume(T) / mean(volume(T-20..T-1))` | ratio |
| `volume_zscore_prev20` | `(volume(T) - mean(previous 20 volumes)) / sample_std(previous 20 volumes)` | standard deviations |
| `return_volatility_prev20` | Sample standard deviation of daily returns from `T-20..T-1` | fraction |
| `return_zscore_prev20` | `(return_1d(T) - mean(previous 20 daily returns)) / sample_std(previous 20 daily returns)` | standard deviations |
| `vnindex_return_1d` | VNINDEX daily return on the same market date | fraction |
| `excess_return_1d` | `return_1d(T) - vnindex_return_1d(T)` | fraction |

Returns and volatility are fractions: `0.01` means 1%. Columns ending in `_pct` are percentage values: `1.0` means 1%. Rolling windows use **prior trading observations**, never calendar-day resampling or day T in the comparison baseline. Day T's completed price and volume may appear in its own features. VNINDEX matches on the exact date; missing benchmark sessions are not forward-filled. These rules prevent future observations from affecting earlier feature rows.

Early rows lack sufficient history, and ratios or z-scores with a zero historical denominator are undefined. They remain blank (`NaN`) in the CSV. No extreme observations are removed, and no missing trading days are invented. `model_ready` is true only when all ten numerical features are present and finite; Task 004 can select those rows before fitting its own scaler or model. The guest-tier one-year equity history limits the number of ready rows. Corporate actions, provider revisions, or unadjusted price discontinuities may produce unusually large returns that need interpretation before being called anomalies.

Inspect the generated table with:

```sh
head -n 5 data/features/market_features.csv
python -c "import pandas as pd; d = pd.read_csv('data/features/market_features.csv'); print(d.groupby('symbol')['model_ready'].agg(['size', 'sum']))"
```

## Layout

- `src/market_radar/ingestion/`: Vnstock daily data access, validation, and caching
- `src/market_radar/features/`: offline financial feature calculation and cache loading
- `src/market_radar/modeling/`: future training and evaluation
- `src/market_radar/visualization/`: future Streamlit and Plotly presentation
- `src/market_radar/config.py`: project-relative paths
- `data/raw/`, `data/processed/`, `data/features/`: local provider, normalized, and feature datasets, ignored by Git
- `artifacts/`: local model outputs, ignored by Git
- `tests/`: pytest checks

## Roadmap

1. **Task 001:** package structure, dependency setup, documentation, and smoke test.
2. **Task 002:** collect and validate historical Vietnamese stock data using Vnstock 5.1.0.
3. **Task 003 (this task):** engineer financial features from cached OHLCV without future leakage.
4. Establish statistical baselines; train and evaluate Isolation Forest anomaly detection.
5. Build the interactive Streamlit dashboard.

This project is for research and portfolio demonstration, not investment advice.
