# TRIDENT: setup, implementation and submission guide

This guide takes you from an empty folder to a submitted entry: a public repository that a
judge can clone and run, real-data results, a recorded demo under 5 minutes, and the 7-slide
deck. Follow the phases in order. Each phase ends with a check and a suggested git commit, so
your history shows the build process the guidelines ask for.

Time budget: Phases 0-3 take about 1 hour. Phase 4 (real data and labeling) takes 3-5 hours,
mostly labeling. Phase 5 (fine-tuning) is optional, 1-2 hours on a free GPU. Phases 6-9 take
about half a day including rehearsal.

---

## Contents

0. [What you are submitting](#0-what-you-are-submitting)
1. [Prerequisites](#1-prerequisites)
2. [Install and run the demo](#2-install-and-run-the-demo)
3. [Understand the codebase](#3-understand-the-codebase)
4. [Replace authored data with real data](#4-replace-authored-data-with-real-data)
5. [Optional: fine-tune transformer models](#5-optional-fine-tune-transformer-models)
6. [Regenerate results, figures, screenshots and the deck](#6-regenerate-results-figures-screenshots-and-the-deck)
7. [Live mode and optional components](#7-live-mode-and-optional-components)
8. [Record the demo video](#8-record-the-demo-video)
9. [Publish and submit](#9-publish-and-submit)
10. [Troubleshooting](#10-troubleshooting)
11. [Reference: commands, endpoints, configuration](#11-reference)

---

## 0. What you are submitting

| Deliverable | Where | Status after this guide |
|---|---|---|
| Source code + architecture | public GitHub repo `iitm-<your-name>-hackathon` | ready |
| Presentation (5-7 slides) | `docs/presentation.pdf` (+ `.pptx`), linked in README | regenerated from your results |
| Demo video | YouTube, Unlisted, link in README | you record it (script in `docs/DEMO_SCRIPT.md`) |
| All data used | `data/` + `data/README.md` | ready; real data added in Phase 4 |
| Live jury pitch | prototype running locally | rehearsed with `docs/QA_PREP.md` |

Two requirements conflict: the problem statement caps the live demo at **5 minutes**, while
the submission guidelines describe a **10-minute** recording. Record under 5 minutes; that
satisfies both.

---

## 1. Prerequisites

| Tool | Version | Why | Check |
|---|---|---|---|
| Python | 3.11, 3.12 or 3.13 | everything | `python --version` |
| Git | any recent | repository | `git --version` |
| Node.js | 18+ | rebuilding the deck only | `node --version` |
| LibreOffice | 7+ | deck to PDF (or export from PowerPoint) | `soffice --version` |
| A browser | any | dashboard | |
| Optional: Docker | 24+ | container run | `docker --version` |
| Optional: GPU or Google Colab | | fine-tuning in Phase 5 | |

Accounts (all free, all optional): Kaggle (datasets), NewsAPI (backfill key), Hugging Face
(not required for public models), YouTube (video), GitHub (repository).

On Windows, use PowerShell. Where this guide shows `export VAR=value`, PowerShell uses
`$env:VAR="value"`. If `make` is not installed, run the command listed under each target in the
`Makefile` directly.

---

## 2. Install and run the demo

### 2.1 Put the code in place

Unzip the delivered repository and rename the folder to the required convention:

```bash
mv iitm-yourname-hackathon iitm-<firstname-lastname>-hackathon
cd iitm-<firstname-lastname>-hackathon
```

### 2.2 Create a virtual environment and install

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

`pip install -e .` registers the `trident` package so `python -m trident ...` works from any
folder. If an install fails, use the exact versions that produced the shipped results:
`pip install -r requirements-lock.txt`.

### 2.3 Run the tests

```bash
python -m pytest
```

Expected: `55 passed`. The suite covers NLP behavior, the engine, Module A properties
(Hypothesis), Module B risk math, the trigger gate and the API contract.

### 2.4 Run the demo

```bash
python -m trident demo
```

Open http://localhost:8000. The header shows `REPLAY · svb_2023-03`, a simulated clock and
the engine latency. Within about two minutes the replay runs through 8-19 March 2023:

1. **Live signals:** cards stream in. Click one to see highlighted evidence, the rationale,
   lineage and the raw JSON. The gate bar shows HELD and TRIGGERED decisions.
2. **Module A:** the heatmap turns red for JPM and BAC as bank-sector sentiment falls.
3. **Module B:** the status flips to STRESS TEST TRIGGERED at 21:40 UTC on 8 March, after the
   first headline at 21:15 was held for lack of a second source. You see P&L by asset class,
   CET1 13.0% → about 7.9%, stressed VaR and the reverse stress test.
4. **Engine Lab:** paste any headline and press Run engine.
5. **System & results:** models, counters, latency histogram, dropped and rejected items.

Other useful runs:

```bash
python -m trident demo --replay covid_2020-03 --speed 1800
python -m trident replay svb_2023-03                  # same pipeline, terminal only
python -m trident analyze "Fed signals two more rate hikes as CPI surprises to upside"
python -m trident stress --event "Credit Event" --impact 9
python -m trident stress --preset covid_2020
python -m trident bench --n 5000
```

API documentation is generated at http://localhost:8000/docs.

**Check:** tests pass and the dashboard shows a triggered stress test.

**Commit:**

```bash
git init
git add .
git commit -m "chore: initial TRIDENT scaffold, engine, modules, dashboard and tests"
```

If you prefer a history that mirrors the build, commit folder by folder instead: `configs/`
and `src/trident/{config,signals,utils}` first, then `ingest/` and `pipeline/`, then `nlp/`,
then `modules/`, then `api/` and `dashboard/`, then `training/` and `eval/`, then `tests/`,
`scripts/`, `data/` and `docs/`. Make each commit only after you have read and run that part.

---

## 3. Understand the codebase

You will be questioned on design choices, so read the code in this order (about 1 hour).

| Order | File | What to understand |
|---|---|---|
| 1 | `src/trident/signals/schema.py` | The signal contract: the three required fields plus evidence, rationale, lineage, latency |
| 2 | `src/trident/pipeline/engine.py` | The five stages, batching, market/sector routing, retrospective capping |
| 3 | `src/trident/nlp/entities.py` | Cashtags, aliases, ambiguity gate, lower-case social names, fuzzy fallback |
| 4 | `src/trident/nlp/lexicon.py`, `sentiment.py` | Lexicon + LM words, negation, delta phrases, ensemble (E1), target-dependent scores |
| 5 | `src/trident/nlp/events.py` | Rules + model hybrid, abstain, credit subtypes |
| 6 | `src/trident/nlp/impact.py` | Features, default coefficients (E3), the ordinal model |
| 7 | `src/trident/pipeline/dedup.py`, `novelty.py` | MinHash LSH with a 48 h window; why cross-source repeats are kept |
| 8 | `src/trident/modules/rebalancer/` | Aggregation (E2), tilt (A1, A2), constraints and band (A3), optimizer (A4), backtest rules |
| 9 | `src/trident/modules/stress/` | Vasicek (B3), repricing (B4), IRB and CET1 (B5), VaR (B6), gate (B1), reverse stress |
| 10 | `src/trident/pipeline/runtime.py`, `api/app.py` | How everything is wired and served |
| 11 | `configs/*.yaml` | Every tunable number lives here |

Equation labels (E1, A2, B3, ...) refer to the blueprint you already have. Keep it open while
you read.

---

## 4. Replace authored data with real data

The shipped numbers for sentiment, events and entities use an **authored** set of 180
fictional headlines, and the impact, IC and backtest numbers use a **synthetic known-answer
market**. They prove the pipeline works. Before you submit, produce real-data numbers. This is
the single most important step for credibility.

### 4.1 Fetch public datasets (no account needed)

```bash
pip install "datasets<3.0" yfinance spacy      # what the fetcher needs (torch is not required here)
python scripts/fetch_data.py --all
```

This downloads, into `data/raw/` and `data/market/`:

* Financial PhraseBank (Hugging Face), twitter-financial-news sentiment and topic sets;
* yfinance daily prices for the 15 index names since 2009 (`data/market/prices_sp15.csv`);
* FRED yields, IG/HY spreads, VIX and EUR/USD (`data/market/fred.csv`);
* a few days of real GDELT headlines plus the latest GDELT events file;
* one pass of RSS headlines;
* the spaCy `en_core_web_sm` model and the Loughran-McDonald word lists.

Each step prints `[ok]` or `[skip]` with the reason. If PhraseBank fails through Hugging Face
(newer `datasets` versions dropped script loaders), use the Kaggle route in 4.2 or download
`FinancialPhraseBank-v1.0.zip` manually and place `Sentences_AllAgree.txt` in
`data/raw/phrasebank/`.

Run the RSS and GDELT steps on two or three different days to collect a varied pool for
labeling:

```bash
python scripts/fetch_data.py --rss --gdelt
```

### 4.2 Kaggle datasets (needs a free account)

1. Kaggle → Settings → API → Create New Token. Save `kaggle.json` to `~/.kaggle/`
   (Windows: `C:\Users\<you>\.kaggle\`).
2. `pip install kaggle`
3. `python scripts/fetch_data.py --kaggle`

This produces:

* `data/raw/news_hist.jsonl`: dated headlines tagged with tickers (2009-2020), used for the
  **real** IC, impact calibration and walk-forward backtest;
* `data/raw/tweets_kaggle.csv`: stock tweets for the social replay;
* `data/raw/transactions.csv`: the Financial Transactions dataset for Module B's book;
* `data/raw/phrasebank/all-data.csv`: the Kaggle copy of PhraseBank.

If a slug has been renamed, search Kaggle for the dataset name in `KAGGLE` inside
`scripts/fetch_data.py` and update the slug.

Rebuild the wholesale book from real transactions:

```bash
python scripts/build_portfolio.py
cat data/portfolio/BOOK_DERIVATION.md
```

### 4.3 Label 300 real headlines (the evaluation set to report)

```bash
python scripts/label_tool.py --target 300
```

For each headline the tool shows the engine's suggestion. Press **Enter** to accept, a digit
**1-9** to change the event class, **p/n/u** to change polarity, **s** to skip, **q** to quit
(progress is saved). Expect 60-90 minutes. Labeling rules that keep the set consistent:

* Label the event the headline **reports**, not what might follow. "Shares fall after
  earnings miss" is Earnings, not Credit Event.
* "Other" is correct for personnel news, conferences, opinion and market colour.
* Polarity is from the point of view of the named company's investors; for market-wide news,
  from the point of view of equity investors overall.
* When unsure, skip. A smaller, clean set beats a larger, noisy one.

The labels go to `data/gold/headlines_gold_real.jsonl`. They are **never** used for training:
`train_events.py`, `nlp_eval.py` and the model registry read them only for evaluation.

### 4.4 Verify the stress presets against history

```bash
python scripts/calibrate_shocks.py
```

This prints realized moves in the 10-year yield, IG/HY spreads, VIX, EUR/USD and an
equal-weighted index of the 15 names for the COVID, SVB and June 2022 windows, next to the
presets in `configs/scenarios.yaml`. If a preset is far off, edit it and say so in your notes.

### 4.5 Replay windows with real headlines (optional)

```bash
python scripts/build_replay.py --gdelt
```

This merges real GDELT headlines for the SVB and COVID windows into the replay files, tagged
`source_type: gdelt`. The demo becomes busier and more realistic. Re-run the integration test
afterwards (`python -m pytest tests/test_engine.py`).

**Commit:** `git add data/gold data/portfolio data/README.md && git commit -m "data: real labeled gold set and book from real transactions"`
(`data/raw/` is gitignored; see 9.2 for what to commit.)

---

## 5. Optional: fine-tune transformer models

The engine runs fully on the lexicon and rules. Transformer backends raise accuracy and make
a strong slide, but they are optional. Everything below also runs on a free Colab T4: upload
the repository, then run the same commands in a notebook cell prefixed with `!`.

### 5.1 Train the sklearn models (minutes, CPU)

```bash
python -m trident.training.weak_labels
python -m trident.training.train_sentiment
python -m trident.training.train_events
```

With real data present, these train on PhraseBank, tweets, weak labels and the mapped topic
set, with synthetic data capped at 30% and reweighted. Each new model is a **challenger**: it
replaces the champion only if it scores higher on the same held-out real data. Decisions are
logged in `models/registry.json`. A rejection is a good result to show; it proves the gate works.

### 5.2 Fine-tune FinBERT and export to ONNX INT8

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
python -m trident.training.finetune_finbert --epochs 3
python -m trident.training.export_onnx --model models/finbert-ft --out models/finbert-onnx
```

Outputs `models/finbert-ft/`, `models/finbert-onnx/model_quant.onnx`,
`docs/results/finetune_finbert.json` (macro-F1 before and after) and
`docs/results/onnx_export.json` (ms per item and INT8 agreement with FP32). The engine loads
`models/finbert-onnx` automatically (`sentiment_backend: auto`). To skip fine-tuning but still
use FinBERT, export the base model: `--model ProsusAI/finbert`.

### 5.3 Fine-tune DeBERTa-v3-small for events

```bash
python -m trident.training.finetune_deberta_events --zeroshot-relabel 2000 --epochs 4
```

This optionally labels up to 2,000 unlabeled headlines with `bart-large-mnli` zero-shot, then
trains DeBERTa on weak + zero-shot + mapped topic labels with class weights, and evaluates on
your real gold set. The engine loads `models/events-deberta` automatically.

### 5.4 Calibrate the impact model on real prices

`scripts/run_all_results.py` does this when real prices and headlines exist. To run it alone:

```bash
python -m trident signals data/raw/news_hist.jsonl --out data/runtime/signals_real.jsonl
python -m trident.training.train_impact --signals data/runtime/signals_real.jsonl \
       --prices data/market/prices_sp15.csv --test-from 2017-01-01
```

This writes `models/impact_model.json`, which the engine then uses instead of the default
coefficients.

### 5.5 Optional LLM rationale

Install Ollama, run `ollama pull llama3.1:8b`, then set `TRIDENT_ENGINE_RATIONALE=llm`. Any
OpenAI-compatible endpoint works via `TRIDENT_ENGINE_LLM_BASE_URL`. The LLM output is discarded
unless every quoted evidence phrase appears verbatim in the source text.

**Commit:** `git add models/registry.json docs/results && git commit -m "feat(nlp): trained models, registry decisions, ONNX benchmark"`
(model weights are gitignored; the README explains how to recreate them.)

---

## 6. Regenerate results, figures, screenshots and the deck

### 6.1 Results

```bash
python scripts/run_all_results.py --robust
```

About 2-3 minutes. It evaluates NLP on every available labeled set (real sets are preferred
automatically), runs the behavioral suite, adversarial validation and the latency benchmark,
calibrates impact, computes IC, runs the walk-forward backtest (real data if present, plus the
synthetic known-answer market), the 5-seed robustness check, the stress-scenario table, and the
SVB and COVID replays. Outputs:

* `docs/results/results.json`: machine-readable, also served at `/api/results`;
* `docs/results/RESULTS.md`: every number with its provenance;
* `docs/figures/*.png`: charts for the README and slides.

Read `RESULTS.md` and check the Provenance table. Rows that say "AUTHORED SET ONLY" or
"KNOWN-ANSWER VALIDATION" are not yet real-data numbers. Present them that way, or finish
Phase 4 first.

### 6.2 Screenshots and architecture diagram

```bash
pip install playwright && python -m playwright install chromium
python scripts/take_screenshots.py
python scripts/render_architecture.py
```

### 6.3 Deck

Edit `docs/deck/candidate.json` with your name, college and repository URL, then:

```bash
cd docs/deck
npm install
node build_deck.js
cd ../..
python scripts/deck_to_pdf.py
```

This writes `docs/presentation.pptx` and `docs/presentation.pdf`. Every number on the slides
is read from `results.json`, so the deck always matches your latest run. Speaker notes are in
the `.pptx`. If LibreOffice is missing, open the `.pptx` in PowerPoint and export to PDF.

Open the PDF and check every slide: your name, no placeholder text, numbers matching
`RESULTS.md`, and the provenance line on slide 5.

### 6.4 README

Update the five template fields at the top of `README.md` (name, college email, college,
video link, deck link) and the results table in section 5 from `RESULTS.md`.

**Commit:** `git add docs README.md && git commit -m "docs: results, figures, screenshots, deck"`

---

## 7. Live mode and optional components

```bash
cp .env.example .env            # add NEWSAPI_KEY if you have one
# set your contact in configs/settings.yaml -> ingest.user_agent (SEC requires it)
python -m trident serve --live
```

Live mode polls RSS (60 s), SEC EDGAR (120 s), GDELT (15 min), NewsAPI (30 min, if a key is
set) and Reddit (2 min). The System tab shows each source's state, last error and lag. If a
source fails three times, its circuit opens and the others carry on.

| Component | Enable with | Notes |
|---|---|---|
| Redis Streams bus | `docker compose --profile redis up` and `TRIDENT_BUS=redis` | multi-process scale-out |
| spaCy ORG check | `python -m spacy download en_core_web_sm` | used automatically when installed |
| MiniLM novelty | `TRIDENT_NOVELTY=minilm` | needs sentence-transformers |
| Optimizer rebalancer | dashboard Method = optimizer, or `rebalancer.method: optimizer` | cvxpy + Ledoit-Wolf |
| Sector contagion | on by default (`rebalancer.contagion_weight`, Module B origin sector) | moonshot feature |
| Drift monitor | `python -m trident.monitoring.psi --ref <old.jsonl> --cur data/runtime/signals.jsonl` | PSI per score |
| Docker | `docker compose up` | same replay demo |

For the live jury pitch, run the replay demo as the main path and keep live mode in a second
terminal to show it exists. Never depend on the network during a pitch.

---

## 8. Record the demo video

Full script with timings and lines: `docs/DEMO_SCRIPT.md`. Practical setup:

1. Close other apps; set the browser zoom to 110% so text is readable in the recording.
2. Restart the server: `python -m trident demo --speed 600`.
3. Recorder: OBS Studio (free) or the OS recorder, 1080p, microphone checked.
4. Record the terminal starting the app (the guidelines ask to show the README command), then
   the dashboard. Aim for **4:30-4:50**.
5. Upload to YouTube as **Unlisted**. Open the link in a private window to confirm it plays.

---

## 9. Publish and submit

### 9.1 Create the repository

On GitHub, create a **public** repository named `iitm-<your-name>-hackathon` with no README.
Then:

```bash
git remote add origin https://github.com/<you>/iitm-<your-name>-hackathon.git
git branch -M main
git push -u origin main
```

### 9.2 What to commit

Commit: `src/`, `configs/`, `scripts/`, `tests/`, `docs/` (including `presentation.pdf`,
`architecture.png`, `results/`, `figures/`, `screenshots/`), `data/` except `data/raw/` and
`data/runtime/`, `models/registry.json`, `README.md`, `LICENSE`, `requirements*.txt`,
`pyproject.toml`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.github/`.

Do not commit: `data/raw/` (licensing and size), model weights, `.env`.

The guidelines require "all data used to run your prototype". The demo runs entirely from the
committed `data/` folder, and `scripts/fetch_data.py` recreates every downloaded dataset;
`data/README.md` documents both. If you want to include a small real sample, commit
`data/gold/headlines_gold_real.jsonl` (your labels, headline text and URLs only).

### 9.3 Final check from a clean clone

```bash
cd /tmp && git clone https://github.com/<you>/iitm-<your-name>-hackathon.git t && cd t
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m pytest && python -m trident demo
```

### 9.4 Submission checklist

* [ ] Repository is public and named `iitm-<your-name>-hackathon`
* [ ] `LICENSE` (MIT) has your name
* [ ] README template fields filled; demo video and deck links work in a private window
* [ ] `docs/presentation.pdf` is 7 slides with your name and current numbers
* [ ] `docs/architecture.png` present and embedded in the README
* [ ] `data/README.md` lists every source and assumption
* [ ] Fresh clone installs and runs with the README commands
* [ ] Commit history is incremental
* [ ] No S&P Global or Crisil client data or names in any dataset
* [ ] No Google Drive or OneDrive links
* [ ] Submitted through the official form before the deadline
* [ ] For the pitch: demo running locally, `docs/QA_PREP.md` rehearsed

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: trident` | run `pip install -e . --no-deps` in the activated venv, or set `PYTHONPATH=src` |
| Port 8000 busy | `python -m trident demo --port 8010` |
| Dashboard blank | hard refresh; check the terminal for errors; `curl localhost:8000/healthz` |
| Replay too slow or fast | `--speed 300` (slower) to `--speed 6000` (faster); gaps are capped at 4 s |
| PhraseBank download fails | Kaggle route (4.2) or manual file (4.1) |
| `yfinance` returns nothing | upgrade it (`pip install -U yfinance`); Yahoo changes endpoints; retry later |
| SEC EDGAR returns 403 | set a real contact in `ingest.user_agent` |
| NewsAPI returns 426/401 | the free plan covers development only, with a 24 h delay; keep it as backfill |
| torch install is huge | use the CPU index URL in 5.2, or fine-tune on Colab |
| `datasets` script error | `pip install "datasets<3.0"` (already pinned in requirements-ml.txt) |
| Deck build: `Cannot find module 'pptxgenjs'` | `cd docs/deck && npm install` |
| PDF conversion hangs | export the `.pptx` to PDF from PowerPoint |
| Tests fail after editing configs | the tests assume shipped configs; restore them or update the test expectations knowingly |

---

## 11. Reference

### Commands

| Command | Purpose |
|---|---|
| `python -m trident demo [--replay NAME] [--speed X] [--port P]` | replay demo server |
| `python -m trident serve --live` / `--idle` | live sources / API only |
| `python -m trident analyze "TEXT" [--source social]` | single analysis, JSON |
| `python -m trident replay NAME [--out file.jsonl]` | offline replay summary |
| `python -m trident signals IN.jsonl --out OUT.jsonl` | batch engine over a file |
| `python -m trident stress --event CLASS --impact I` / `--preset NAME` | stress run |
| `python -m trident backtest --signals S --prices P` | Module A backtest |
| `python -m trident bench --n N` | latency and throughput |
| `python -m trident.eval.nlp_eval` / `checklist` / `adversarial` / `ic` | evaluations |
| `python -m trident.training.<module>` | training (see Phase 5) |

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz`, `/metrics`, `/api/status` | health, Prometheus metrics, runtime status |
| GET | `/api/signals?ticker=&since=&event=&min_impact=&compact=` | query signals |
| GET | `/api/signals/{id}` | one signal |
| POST | `/api/analyze` `{text, source_type, publish}` | Engine Lab |
| GET | `/api/rebalancer`, `/api/rebalancer/history` | Module A state |
| POST | `/api/rebalancer/params` `{lambda, w_max, band, turnover_max, method}` | Module A controls |
| GET | `/api/stress/latest`, `/api/stress/history`, `/api/stress/decisions`, `/api/stress/presets` | Module B state |
| POST | `/api/stress/run` `{event_class, impact, preset, origin_sector, custom_shock}` | run a scenario |
| GET | `/api/book` | wholesale book |
| POST | `/api/replay/start` `{name, speed}`, `/api/replay/stop` | replay control |
| POST | `/api/synthetic/inject?scenario=credit&n=10` | tagged synthetic burst |
| WS | `/ws` | live push: signal, decision, weights, stress |
| GET | `/api/stream` | Server-Sent Events signal stream |

### Configuration

All in `configs/`. Any `settings.yaml` value can be overridden by an environment variable
`TRIDENT_<SECTION>_<KEY>`, for example `TRIDENT_ENGINE_SENTIMENT_BACKEND=lexicon`.

| File | Contents |
|---|---|
| `settings.yaml` | engine backends, source reliability, aggregation, rebalancer parameters, feeds, storage |
| `universe.yaml` | 15 index names, credit watchlist, aliases, sectors, sector link matrix |
| `taxonomy.yaml` | 8 event classes, keywords, base severities, credit subtypes, abstain threshold |
| `scenarios.yaml` | trigger gate, shock vectors, historical presets, capital and VaR parameters |
