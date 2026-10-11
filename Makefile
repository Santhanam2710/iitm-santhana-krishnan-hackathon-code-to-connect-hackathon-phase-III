# TRIDENT task runner. On Windows without make, run the commands after each target by hand.
PY ?= python
export PYTHONPATH := src

.PHONY: setup demo live test results figures screenshots deck data train bench clean all

setup:            ## install core dependencies
	$(PY) -m pip install -r requirements.txt

demo:             ## replay the SVB week on http://localhost:8000 (no keys, no network)
	$(PY) -m trident demo --replay svb_2023-03 --speed 600

live:             ## live RSS / EDGAR / GDELT / NewsAPI / Reddit
	$(PY) -m trident serve --live

test:             ## unit, property, behavioral and API tests
	$(PY) -m pytest

data:             ## rebuild bundled data (replays, gold, synthetic, portfolio)
	$(PY) scripts/build_replay.py && $(PY) scripts/build_gold_v0.py && $(PY) scripts/make_synthetic_data.py && $(PY) scripts/build_portfolio.py

fetch:            ## download REAL public datasets (needs internet)
	$(PY) scripts/fetch_data.py --all

train:            ## weak labels + sklearn models (champion/challenger gated)
	$(PY) -m trident.training.weak_labels && $(PY) -m trident.training.train_sentiment && $(PY) -m trident.training.train_events

results:          ## every number in the deck -> docs/results + docs/figures
	$(PY) scripts/run_all_results.py

bench:
	$(PY) -m trident bench --n 5000

screenshots:      ## needs playwright + chromium
	$(PY) scripts/take_screenshots.py && $(PY) scripts/render_architecture.py

deck:             ## docs/presentation.pptx + .pdf (needs node + pptxgenjs + LibreOffice)
	cd docs/deck && npm install --silent && node build_deck.js && cd ../.. && $(PY) scripts/deck_to_pdf.py

all: data train results screenshots deck

clean:
	rm -rf data/runtime .pytest_cache **/__pycache__
