# Data used by TRIDENT

Guideline §9 asks for every dataset, its source and our assumptions. Everything the prototype
needs to run is in this folder; larger public datasets are downloaded into `data/raw/`
(gitignored) by `scripts/fetch_data.py`. No proprietary or confidential data is used, and no
S&P Global or Crisil client data, names or engagement details appear anywhere.

## Shipped with the repository

| Path | What | Provenance | Used for |
|---|---|---|---|
| `replay/svb_2023-03.jsonl` | 40 items, 8-19 Mar 2023 | **Reconstructed** from widely reported public events, paraphrased in our own words, approximate UTC times; social posts and traps are **synthetic** (`is_synthetic: true`). Built by `scripts/build_replay.py`. | Demo, integration tests |
| `replay/covid_2020-03.jsonl` | 24 items, 17 Feb-23 Mar 2020 | Same method as above | Demo, integration tests |
| `gold/headlines_gold_v0.jsonl` | 180 labeled headlines | **Authored, fictional** (real company names are needed to test entity resolution; the events are invented). Built by `scripts/build_gold_v0.py`. | Regression tests and a smoke-level benchmark. Optimistic: written by the same author as the keyword taxonomy. |
| `gold/headlines_gold_real.jsonl` | created by you | **Real** headlines labeled with `scripts/label_tool.py` | The evaluation set to report |
| `synthetic/synthetic_news.jsonl` | 3,000 headlines/tweets | Template generator, seed 20261011 | Training augmentation, capped at 30% and reweighted |
| `synthetic/market_news.jsonl` | ~940 dated events | Template generator | Known-answer market (drives `market/prices_synthetic.csv`) |
| `synthetic/weak_labeled.jsonl` | weak labels | Labeling functions over replay/GDELT/RSS text | Event classifier training |
| `samples/tweets_sample.jsonl` | 300 tweets | Synthetic | Social replay adapter demo |
| `market/prices_synthetic.csv` | 15 names, 2019-2025 daily | **Synthetic known-answer market**: factor model + planted event returns | Validating impact calibration, IC and backtest code |
| `portfolio/transactions_sample.csv` | 5,000 transactions | Synthetic, in the shape of the Kaggle Financial Transactions dataset | Deriving the wholesale book |
| `portfolio/wholesale_book.csv` | 16 positions, ~$3.3bn | Derived by `scripts/build_portfolio.py` (method in `BOOK_DERIVATION.md`) | Module B |

## Downloaded on your machine (`python scripts/fetch_data.py --all`, plus `--kaggle`)

| Dataset | License / terms | Lands in | Used for |
|---|---|---|---|
| Financial PhraseBank (Malo et al., 2014) via Hugging Face `takala/financial_phrasebank` or Kaggle `ankurzing/sentiment-analysis-for-financial-news` | CC BY-NC-SA 3.0 | `raw/phrasebank/` | Sentiment evaluation and fine-tuning |
| `zeroshot/twitter-financial-news-sentiment` | MIT | `raw/twitter_sentiment/` | Sentiment training/eval (social register) |
| `zeroshot/twitter-financial-news-topic` | MIT | `raw/twitter_topic/` | Event classifier training/eval (topics mapped to our taxonomy, see `training/datasets.py`) |
| GDELT 2.0 DOC API + events files | Free, attribution to the GDELT Project | `raw/gdelt/` | Real headlines for labeling, CAMEO weak labels, `build_replay.py --gdelt` |
| RSS newswire headlines | Each publisher's feed terms: we keep headline, short summary, URL only | `raw/rss/` | Live mode, labeling |
| SEC EDGAR 8-K current filings | Public; SEC fair-access policy (descriptive User-Agent, rate limits) | live only | Live mode; Item codes as gold event labels |
| NewsAPI | Free Developer plan: 100 requests/day, 24 h delay, development use only | `raw/newsapi/` | Backfill only |
| Kaggle massive stock news (2009-2020) | Kaggle dataset terms | `raw/news_hist.jsonl` | Real IC, impact calibration and backtest |
| Kaggle stock tweets (2021-2022) | Kaggle dataset terms | `raw/tweets_kaggle.csv` | Social replay |
| Kaggle Financial Transactions dataset | Kaggle dataset terms | `raw/transactions.csv` | Rebuilding the wholesale book from real transactions |
| yfinance daily prices | Yahoo terms; research use | `market/prices_sp15.csv` | Real backtest and event study |
| FRED (DGS2, DGS10, IG/HY OAS, VIX, EUR/USD) | Public domain (St. Louis Fed) | `market/fred.csv` | Calibrating stress presets (`scripts/calibrate_shocks.py`) |
| Loughran-McDonald dictionary (via `pysentiment2`) | Free for academic use | `lexicons/` | Lexicon sentiment features |

## Assumptions

1. Consumer spend mix in transaction data is a proxy for the sector mix of a bank's corporate loan book.
2. Stress presets are approximate historical moves; verify them with `scripts/calibrate_shocks.py` after fetching FRED data.
3. Replay timestamps are approximate; replay demonstrates mechanics, not historical latency.
4. Synthetic data never enters an evaluation set, and every synthetic record carries `is_synthetic: true` through the API and dashboard.
