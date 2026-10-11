"""Fine-tune FinBERT (ProsusAI/finbert) on Financial PhraseBank + financial tweets.

Requires requirements-ml.txt (torch, transformers, datasets, accelerate). Runs on CPU in
roughly 20-40 minutes for 3 epochs on ~12k sentences; on a free Colab T4 in a few minutes.

  python -m trident.training.finetune_finbert --epochs 3 --out models/finbert-ft
  python -m trident.training.export_onnx --model models/finbert-ft --out models/finbert-onnx

Outputs a Hugging Face model directory and docs/results/finetune_finbert.json with held-out
macro-F1. The engine picks it up automatically (sentiment_backend: auto).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trident.config import MODELS_DIR, resolve
from trident.training import datasets as ds

LABELS = ["negative", "neutral", "positive"]


def main() -> None:  # pragma: no cover - heavy dependency
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ProsusAI/finbert")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--syn-share", type=float, default=0.1)
    ap.add_argument("--out", default=str(MODELS_DIR / "finbert-ft"))
    a = ap.parse_args()

    parts = [d for d in (ds.phrasebank("AllAgree"), ds.twitter_sentiment("train")) if d is not None]
    if not parts:
        raise SystemExit("No real sentiment data found. Run: python scripts/fetch_data.py --sentiment")
    real = pd.concat(parts, ignore_index=True)
    real = real[real["label"].isin(LABELS)]
    tr, te = train_test_split(real, test_size=0.15, random_state=42, stratify=real["label"])
    if a.syn_share > 0:
        syn = ds.synthetic("sentiment").sample(int(len(tr) * a.syn_share), random_state=0)
        tr = pd.concat([tr, syn], ignore_index=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=3)
    # map our label names onto the base model's label ids so we keep its classifier head
    name_to_id = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    to_id = {l: name_to_id.get(l, i) for i, l in enumerate(LABELS)}

    def enc(df: pd.DataFrame) -> Dataset:
        d = Dataset.from_pandas(pd.DataFrame({"text": df["text"].tolist(), "label": df["label"].map(to_id).tolist()}))
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=96), batched=True)

    dtr, dte = enc(tr), enc(te)

    def metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro")}

    args = TrainingArguments(output_dir=str(resolve(a.out)) + "_runs", num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=a.batch, per_device_eval_batch_size=64, eval_strategy="epoch",
                             save_strategy="no", logging_steps=50, weight_decay=0.01, warmup_ratio=0.06,
                             fp16=torch.cuda.is_available(), report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dte, tokenizer=tok,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    base_eval = trainer.evaluate()
    trainer.train()
    final = trainer.evaluate()
    trainer.save_model(resolve(a.out))
    tok.save_pretrained(resolve(a.out))
    res = {"base_model": a.base, "heldout_n": len(te), "macro_f1_before": base_eval.get("eval_macro_f1"),
           "macro_f1_after": final.get("eval_macro_f1"), "train_rows": len(tr), "epochs": a.epochs}
    Path(resolve("docs/results")).mkdir(parents=True, exist_ok=True)
    Path(resolve("docs/results/finetune_finbert.json")).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
