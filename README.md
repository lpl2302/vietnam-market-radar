# Vietnam Market Intelligence Radar

A Python project for researching unusual trading activity in Vietnamese stocks, with particular relevance to VietinBank Securities (VBSE). The planned workflow collects historical market data, prepares features, detects anomalies, compares results with statistical baselines, and presents findings in an interactive dashboard.

**Current status:** Task 006 adds covered-sector analysis, retrospective follow-up comparisons, model-only investigation, and Markdown research briefs to the offline dashboard.

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

Early rows lack sufficient history, and ratios or z-scores with a zero historical denominator are undefined. They remain blank (`NaN`) in the CSV. No extreme observations are removed, and no missing trading days are invented. `model_ready` is true only when all ten numerical features are present and finite; the detector selects only those rows for fitting and scoring. The guest-tier one-year equity history limits the number of ready rows. Corporate actions, provider revisions, or unadjusted price discontinuities may produce unusually large returns that need interpretation before being called anomalies.

Inspect the generated table with:

```sh
head -n 5 data/features/market_features.csv
python -c "import pandas as pd; d = pd.read_csv('data/features/market_features.csv'); print(d.groupby('symbol')['model_ready'].agg(['size', 'sum']))"
```

## Detect unusual trading sessions offline

Run the pooled Isolation Forest after building the feature table; this command makes **no network requests**:

```sh
python -m market_radar.detect_anomalies
python -m market_radar.detect_anomalies --n-estimators 300 --contamination 0.05 --random-state 42
```

The model uses exactly the ten numerical features in the table above. One `StandardScaler` and one `IsolationForest` are fitted across all stocks. Default settings are 200 trees, expected training contamination 0.05, seed 42, and `n_jobs=-1`. Stock and sector identify rows but are not predictors. The scaler and model are saved together in a scikit-learn pipeline.

The earliest 70% of **unique model-ready trading dates** form the training set; the latest 30% form a single chronological holdout. All stocks on one date stay in the same partition, even if their individual histories differ. Preprocessing and the Isolation Forest boundary are fitted on training rows only. Both partitions are scored, with `train` meaning in-sample and `holdout` meaning out-of-sample. The holdout does not choose settings or thresholds. This is one holdout assessment, not a walk-forward backtest.

`anomaly_score` is the negative of scikit-learn's `decision_function`: larger values are more unusual, and positive values cross the fitted anomaly boundary. `anomaly_flag` comes directly from `predict == -1`. Contamination is an expected training fraction; the holdout fraction can differ. Non-ready rows remain in the result with `scoring_status=insufficient_features`, `partition=not_model_ready`, and missing predictions. Flagged rows receive deterministic **heuristic context** drawn from observed return, relative performance, intraday range, and volume. These sentences describe the data; they are neither Isolation Forest feature attribution nor a causal explanation.

The output files are:

| Path | Purpose |
| --- | --- |
| `artifacts/isolation_forest.joblib` | Fitted scaler and model; load only locally trusted joblib files |
| `artifacts/isolation_forest.meta.json` | Ordered feature names, settings, dates, sample counts, versions, hashes, and creation time |
| `data/anomalies/market_anomalies.csv` | OHLCV, features, partitions, scores, flags, baselines, and explanations for future dashboard use |
| `data/anomalies/evaluation.json` | Descriptive rates, score distribution, baseline overlap, top events, and retrospective event study |
| `data/anomalies/market_anomalies.meta.json` | Result provenance and model/source hashes |

These generated files are ignored by Git. The CLI accepts `--input`, `--artifact-dir`, and `--output-dir` for offline experiments. Inference code checks the saved feature order and applies the fitted pipeline. The input CSV and model metadata carry SHA-256 hashes; reruns with the same input and seed should produce the same scores and flags, though file creation times and package version changes can differ.

The evaluation reports anomaly counts and rates overall, by stock and sector, and by training/holdout partition. It compares holdout flags to two fixed statistical baselines: volume z-score **above 3**, and absolute return z-score **above 3**. Their overlap and the number of model-only detections are diagnostic comparisons, not accuracy measures; there are no verified anomaly labels. A retrospective event study computes absolute close-to-close returns over the next five observed sessions **within each symbol** for eligible holdout rows. It excludes rows without a full five-session window, uses future prices only after scoring, and cannot establish predictive value, causality, or profitability.

Unsupervised outliers can reflect genuine market events, normal volatility regimes, trading limits, corporate actions, or data-provider adjustments. A pooled model may flag some stocks more often because their volatility differs. Analysts should inspect OHLCV history, provider metadata, and public disclosures before interpreting any event. The output is for research support, not an automated trading signal or investment recommendation.

## Explore the historical dashboard

Generate the local inputs, then launch Streamlit from the repository root:

```sh
source .venv/bin/activate
python -m market_radar.build_features
python -m market_radar.detect_anomalies
python -m market_radar.build_analysis
streamlit run app.py
```

The app loads `data/features/market_features.csv`, `data/anomalies/market_anomalies.csv`, `data/anomalies/evaluation.json`, `artifacts/isolation_forest.meta.json`, and the generated `data/analysis/` tables. It works offline, does not deserialize the model, and never retrains or calls Vnstock. Missing, stale, or malformed outputs produce a regeneration message. Existing dashboard views remain usable when optional research analysis files need regeneration. Cached loading refreshes when an input file's size or modification time changes. Prices and statistics are **historical**, with the latest available dataset session shown in the interface.

The five navigation views are:

- **Market Overview:** eligible sessions, anomaly counts and rates, stock and sector comparisons with both counts and eligible-session rates, monthly frequency, score distribution, train/holdout comparison, and a sortable ranked table.
- **Stock Explorer:** stock and date selection, filtered stock statistics, historical closing prices and volume, correctly dated anomaly markers, and selected-event details.
- **Sector Intelligence:** covered-stock summaries, a sector-by-month anomaly-rate heatmap, a count/rate comparison switch, partition-aware activity timeline, stock drilldowns, normalized price comparisons, and focused GAS/VHM source-data review.
- **Anomaly Investigation:** stock, sector, date, partition, score, and model-only filters; ranked event table; selected-event price and volume context; later observed returns; and a deterministic Markdown research brief download. Holdout events appear by default; choose a flagged session to open its details. Changing the filtered cohort clears the selection.
- **Model & Methodology:** saved configuration and feature order, chronological split, score interpretation, fixed z-score baseline overlap, and retrospective holdout event-study comparisons.

Use the navigation and filters to choose an event; hover over price and volume charts for exact values. Reset buttons restore the explorer or investigation filters. Filtered event tables can be downloaded as CSV. To capture portfolio screenshots, run the app locally, open the Streamlit URL shown in the terminal, select each view, and use your browser's screenshot tool. Generated datasets and model artifacts remain excluded from Git; screenshots should avoid publishing proprietary market data beyond its permitted use.

The dashboard's stock close values are in **thousands of VND per share**, volume is in **shares**, returns shown in cards are **percentages**, and market-relative performance is in **percentage points**. Anomaly scores are signed decision values, not probabilities. Training observations were used to fit the model; holdout observations were scored by that fitted model. Rates exclude rows without complete model features. The historical coverage is constrained by local Vnstock access and source revisions; apparent outliers require human inspection. Neither flagged events nor model-only events establish manipulation, expected returns, or investment merit.

### Reproduce the research analysis

After anomaly detection, run `python -m market_radar.build_analysis`. This reads existing validated local files and writes ignored outputs under `data/analysis/`: `sector_summary.csv`, `sector_monthly.csv`, `event_outcomes.csv`, `holdout_comparisons.json`, `quality_diagnostics.json`, and `analysis.meta.json`. The metadata records the anomaly-result and feature-source hashes, one- and five-session horizons, fixed baseline thresholds, subgroup minimum size, and creation time. Rebuild these files when anomaly results change. No source CSV, features, model scores, or flags are modified.

Sector aggregates describe **covered stocks only**. The flag rate is flagged model-ready stock-sessions divided by model-ready stock-sessions in that covered sector and filter selection. Mean daily return and historical daily volatility use the same eligible rows; volatility is the sample standard deviation of daily returns. Some sectors have one stock, so their rates should not be read as industry-wide estimates or rankings. Normalized stock charts set each stock's first observed close in the selected view to 100, preserving its own price series and dates.

The event study uses each stock's next **observed trading session** and its fifth subsequent observed session: `next_return = close(T+1) / close(T) - 1` and `next_five_return = close(T+5) / close(T) - 1`; the absolute move is `abs(next_five_return)`. No calendar-day shift or missing-price interpolation is used. Rows lacking a full future window are excluded from five-session comparisons. These later prices are computed in a separate analysis module and never become model inputs. The holdout comparisons report flagged/unflagged and model-only/baseline-overlap groups with counts, mean and median absolute moves, and signed outcome proportions. Stock or sector subgroup comparisons appear only when each side has at least five complete windows. These are descriptive, selected historical samples, not predictive validation, significance tests, or profitability estimates.

Model-only events satisfy neither fixed Task 004 rule (`volume_zscore_prev20 > 3` and `abs(return_zscore_prev20) > 3`) but were flagged by the pooled model's existing boundary. The investigation panel displays the saved joint-feature context without assigning causal importance to any feature. The brief download copies actual observed and, when available, follow-up values into a concise Markdown note; it adds no news claims or investment recommendation. GAS/VHM diagnostics show monthly flag concentrations, price and volume extremes, partition rates, OHLC range, invalid-price and invalid-volume checks, and review candidates. Investigate adjustment conventions and disclosures manually before inferring a cause.

## Layout

- `src/market_radar/ingestion/`: Vnstock daily data access, validation, and caching
- `src/market_radar/features/`: offline financial feature calculation and cache loading
- `src/market_radar/modeling/`: offline model training, evaluation, and persistence
- `src/market_radar/analysis/`: separate offline sector and retrospective research calculations
- `src/market_radar/visualization/`: offline dashboard loading, summaries, Plotly charts, and Streamlit views
- `app.py`: Streamlit entry point
- `src/market_radar/config.py`: project-relative paths
- `data/raw/`, `data/processed/`, `data/features/`, `data/anomalies/`, `data/analysis/`: local datasets and results, ignored by Git
- `artifacts/`: local model outputs, ignored by Git
- `tests/`: pytest checks

## Roadmap

1. **Task 001:** package structure, dependency setup, documentation, and smoke test.
2. **Task 002:** collect and validate historical Vietnamese stock data using Vnstock 5.1.0.
3. **Task 003:** engineer financial features from cached OHLCV without future leakage.
4. **Task 004:** train pooled Isolation Forest and compare its anomalies with fixed statistical baselines.
5. **Task 005:** explore saved historical results in the interactive Streamlit dashboard.
6. **Task 006 (current):** investigate covered-sector patterns and retrospective outcomes without changing the detector.

This project is for research and portfolio demonstration, not investment advice.
