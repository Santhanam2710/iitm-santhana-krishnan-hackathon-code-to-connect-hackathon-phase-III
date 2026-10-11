# Jury Q&A preparation

Answer in two sentences, then offer to show the code or the number. Every answer below points
to where the evidence lives in the repository.

**1. Why a lexicon and FinBERT instead of an LLM for sentiment?**
Latency, determinism and measurability: the engine scores a document in about 1.4 ms on a CPU,
returns the same answer every time and is evaluated against baselines. The LLM writes only the
rationale, off the critical path, and its output is discarded unless every quoted phrase is in
the source. (`nlp/sentiment.py`, `nlp/rationale.py`)

**2. How do you know the impact score means anything?**
It is fitted to a target: deciles of realized |CAR[0,+1]| from a market-model event study. On
the known-answer market the out-of-sample Spearman is 0.20, and 0.23 ± 0.08 across five seeds.
`train_impact.py` runs the same calibration on real prices. (`training/train_impact.py`, `docs/results/robustness.json`)

**3. NewsAPI's free plan is delayed by 24 hours. How is this real-time?**
We separate engine latency (milliseconds) from source latency (minutes to a day) and show both
on every signal. The live path uses RSS and SEC EDGAR; NewsAPI is backfill. With a licensed
newswire, engine latency is the end-to-end latency.

**4. What stops a fake headline from triggering a stress test?**
Four things: the event classifier abstains when unsure; impact must exceed 7; two independent
sources must agree within 30 minutes unless impact is 9 or 10; and retrospective articles are
capped. Every decision is logged with its reason. In the SVB replay the first headline was held.
(`modules/stress/service.py: TriggerGate`)

**5. How does Module A avoid overtrading?**
Exponential decay and shrinkage toward zero when coverage is thin, a no-trade band, a turnover
cap, a minimum holding time and costs on every trade. Parameters are chosen walk-forward on
earlier years only. (`modules/rebalancer/`)

**6. Your backtest is positive. Is that real alpha?**
No claim of alpha. The positive out-of-sample IR comes from a synthetic market with a planted
post-event drift, which tests that the pipeline can find a signal that exists. Across five seeds
it is 0.29 ± 0.62: the signal is recovered every time, but whether it pays after costs varies.
Real-data results come from the same command once prices and headlines are fetched.

**7. Why the Vasicek model?**
It is the single-factor model behind the Basel IRB capital formula, so stressed PDs and the
capital change come from the framework regulators use. Sector loadings let a credit shock hit
commercial real estate harder than healthcare. (`modules/stress/credit.py`)

**8. Why does CET1 fall more than P&L alone would imply?**
Two effects: losses reduce capital, and stressed PDs raise risk-weighted assets through the IRB
formula. We pass through 35% of the PD migration to RWA, a through-the-cycle assumption stated
in `configs/scenarios.yaml`.

**9. How do you handle sentences with several companies?**
Target-dependent scoring: competitive verbs ("X beats Y") give opposite signs, and otherwise each
company gets the sentiment of its own clause. Behavioral tests check it. (`nlp/sentiment.py: score_targets`)

**10. How did you avoid look-ahead bias?**
Signals are keyed on publication time. With close-to-close returns, day d's weights may only use
news published before the close of day d−1. The sentiment book ignores observations stamped after
the evaluation time, and a test checks this. (`modules/rebalancer/backtest.py`, `tests/test_modules.py`)

**11. Doesn't synthetic data bias the models?**
It is capped at 30% of training, reweighted to real class priors and never used for evaluation.
Our own adversarial test shows the template text is easy to detect (AUC 0.95), and the registry
rejected models trained on it. That is why the next step is an LLM generator that must pass the
same test. (`eval/adversarial.py`, `models/registry.json`)

**12. How would this scale to a bank?**
Stateless engine workers behind the bus interface; swap the in-process queue for Redis Streams
(one setting) or Kafka partitioned by ticker. The signal schema already carries lineage for a
model inventory.

**13. Why should a ratings and analytics firm care?**
Credit surveillance: obligors flagged within minutes of adverse news, ranked by calibrated impact,
with credit subtypes in rating-action language (downgrade, outlook negative, default, liquidity)
and the evidence attached for the analyst.

**14. What would you build with three more months?**
Real-label fine-tuning and ONNX serving, coverage of Indian markets and filings, multilingual news,
an LLM synthetic generator that passes adversarial validation, and a learned contagion network.

**15. What was the hardest bug?**
Suggested honest answer: the known-answer generator once attached market-wide headlines to single
stocks, which silently broke impact calibration. The fix and the 5-seed robustness check came out
of that; one random draw had looked fine.
