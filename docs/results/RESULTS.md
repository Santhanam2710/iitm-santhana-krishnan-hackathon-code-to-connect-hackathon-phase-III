# TRIDENT results

Generated 2026-10-11T12:27:28.910Z by `python scripts/run_all_results.py` in 45.7 s. Every number below is produced by code in this repository.

## Provenance

| Number | Data | Status |
|---|---|---|
| NLP metrics | gold_v0: authored fictional headlines (regression/smoke benchmark) | AUTHORED SET ONLY: replace by running fetch_data.py + label_tool.py |
| Impact calibration, IC, backtest | synthetic known-answer market (data/market/prices_synthetic.csv) | KNOWN-ANSWER VALIDATION, not a performance claim |
| Latency | measured on this machine, lexicon/rules backends | REAL measurement |
| Stress tests | synthetic wholesale book (data/portfolio) under configured scenarios | model output |
| SVB / COVID replays | reconstructed public timelines + tagged synthetic posts | demo / integration test |

## Engine

* Latency per document: p50 **1.43 ms**, p95 **3.29 ms**; throughput 855 docs/s (x86_64 · Linux · Python 3.13.16).
* Module A rebalance p50 0.34 ms; Module B full revaluation p50 2.75 ms.

### Sentiment (macro-F1, 3 classes)

| Set | n | TRIDENT | VADER | naive word count |
|---|---|---|---|---|
| gold_v0 | 180 | 0.650 | 0.475 | 0.228 |

### Events (macro-F1, 9 labels)

| Set | n | Rules | Best | Model |
|---|---|---|---|---|
| gold_v0 | 180 | 0.921 | 0.921 | rules-v1 |

Entities: precision 1.000, recall 0.979, ambiguity traps 3/3. Behavioral tests: 58/58 passed.

Caution: gold_v0 was written by the same author as the taxonomy keywords and lexicon, so its event and sentiment scores are optimistic. Report the real-data rows once fetched.

### Model registry (champion/challenger)

* 2026-10-11T12:27:26.046Z sentiment_sklearn: candidate 0.4849 vs lexicon 0.65 on gold_v0 (authored) -> **rejected**
* 2026-10-11T12:27:27.742Z events_sklearn: candidate 0.8778 vs rules 0.9206 on gold_v0 (authored) -> **rejected**

### Synthetic data realism

Adversarial AUC real vs synthetic = **0.948** (target <= 0.65; passes: False). Top synthetic tells: demand, expectations, product, restores services, quarter, strong. Template data is easy to detect, which is why it is capped at 30% of training data and never used for evaluation.

## Impact calibration and Module A

* Spearman(impact, |CAR[0,+1]|) out of sample: **0.201** (123 test events). Mean |CAR| by calibrated impact (bps): {'4': 439.4, '5': 299.4, '6': 406.6, '7': 682.1}.
* IC (sentiment at close d-1 vs market-adjusted return d): mean 0.041, t 1.44, days 80.
* Walk-forward: parameters chosen on data before 2024-01-01 = {'lambda': 0.75, 'band': 0.01, 'turnover_max': 0.05}; out-of-sample 2024-01-01 to 2025-12-31: net IR **0.95**, gross IR 1.23, cost 26 bps, Sharpe 0.03 vs cap-weight 0.00, max drawdown 24.9%.
* Robustness over 5 seeds (known-answer): impact Spearman 0.23 ± 0.08, IC 0.064 ± 0.030, net IR 0.29 ± 0.62, gross IR 0.59 ± 0.64. The planted signal is recovered every time; whether it pays after costs varies by draw.

## Module B

* Credit Event, impact 9: P&L **-80.3 $m** (-2.46%), CET1 **13.00% -> 8.41%**, RWA 2738 -> 3278 $m, 99% 10-day VaR 37.0 -> 63.6 $m.
* With Banks-sector contagion: P&L -85.4 $m, CET1 8.11%.
* Reverse stress (Credit Event): CET1 breaches 7% at 1.09x the full scenario.

| Scenario | P&L $m | CET1 after | VaR after $m |
|---|---|---|---|
| Credit Event I=7 | -50.8 | 9.98% | 56.2 |
| Credit Event I=8 | -64.5 | 9.22% | 59.9 |
| Credit Event I=9 | -80.3 | 8.41% | 63.6 |
| Credit Event I=10 | -98.5 | 7.55% | 67.2 |
| Geopolitical I=7 | -37.1 | 10.81% | 53.0 |
| Geopolitical I=8 | -46.7 | 10.25% | 56.2 |
| Geopolitical I=9 | -57.4 | 9.64% | 59.4 |
| Geopolitical I=10 | -69.3 | 9.01% | 62.6 |
| Macroeconomic I=7 | -84.8 | 9.53% | 44.5 |
| Macroeconomic I=8 | -102.7 | 8.77% | 46.0 |
| Macroeconomic I=9 | -121.3 | 7.99% | 47.7 |
| Macroeconomic I=10 | -140.4 | 7.21% | 49.4 |
| COVID crash, Feb-Mar 2020 (approximate moves) | -191.9 | 4.29% | 97.4 |
| SVB / regional-bank run, Mar 2023 (approximate moves) | -15.7 | 10.93% | 52.2 |
| June 2022 CPI print and 75 bp hike (approximate moves) | -51.1 | 10.78% | 46.4 |

## SVB replay (8-19 March 2023)

40 items -> 46 signals; dropped {'no resolvable entity and not market-wide': 3, 'duplicate': 1}; entity rejections 1; held at gate 13; triggers 5.
First item 2023-03-08T21:15:00Z; first stress trigger 2023-03-08T21:40:00.000Z (2 independent source(s)).

