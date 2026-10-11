"""Fine-tune DeBERTa-v3-small as the event classifier (distillation of weak + zero-shot labels).

Steps (all optional except the last):
  1. python -m trident.training.weak_labels                 # keyword/regex/8-K/CAMEO votes
  2. python -m trident.training.finetune_deberta_events --zeroshot-relabel 2000
     (labels up to N unlabeled headlines with bart-large-mnli and keeps confident ones)
  3. python -m trident.training.finetune_deberta_events --epochs 4

Training set = weak labels + zero-shot labels + twitter-topic (mapped) + capped synthetic,
with class weights. Evaluated on gold_real (if you labeled it) and gold_v0. The engine loads
models/events-deberta automatically when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trident.config import DATA_DIR, MODELS_DIR, resolve
from trident.signals.schema import EVENT_LABELS
from trident.training import datasets as ds


def zeroshot_relabel(n: int, min_score: float = 0.6) -> int:  # pragma: no cover - heavy
    from trident.nlp.events import ZeroShotBackend

    texts = ds.unlabeled_corpus()[:n]
    zs = ZeroShotBackend()
    rows = []
    for i in range(0, len(texts), 16):
        p = zs.predict(texts[i : i + 16])
        for t, row in zip(texts[i : i + 16], p):
            j = int(np.argmax(row))
            if row[j] >= min_score:
                rows.append({"text": t, "label": EVENT_LABELS[j], "confidence": float(row[j]), "votes": ["zeroshot"]})
    f = DATA_DIR / "synthetic" / "zeroshot_labeled.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return len(rows)


def main() -> None:  # pragma: no cover - heavy dependency
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--syn-share", type=float, default=0.3)
    ap.add_argument("--zeroshot-relabel", type=int, default=0)
    ap.add_argument("--out", default=str(MODELS_DIR / "events-deberta"))
    a = ap.parse_args()
    if a.zeroshot_relabel:
        print("zero-shot labeled:", zeroshot_relabel(a.zeroshot_relabel))

    parts = []
    for f in ("weak_labeled.jsonl", "zeroshot_labeled.jsonl"):
        p = DATA_DIR / "synthetic" / f
        if p.exists():
            parts.append(pd.read_json(p, lines=True)[["text", "label"]])
    tt = ds.twitter_topic("train")
    if tt is not None:
        parts.append(tt[["text", "label"]])
    real = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["text", "label"])
    syn = ds.synthetic("event")[["text", "label"]]
    n_syn = int(max(len(real), 1) * a.syn_share / (1 - a.syn_share)) if len(real) else len(syn)
    train = pd.concat([real, syn.sample(min(n_syn, len(syn)), random_state=0)], ignore_index=True)
    train = train[train["label"].isin(EVENT_LABELS)]
    label2id = {l: i for i, l in enumerate(EVENT_LABELS)}
    counts = train["label"].value_counts()
    class_w = torch.tensor([len(train) / (len(EVENT_LABELS) * counts.get(l, 1)) for l in EVENT_LABELS], dtype=torch.float)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=len(EVENT_LABELS), id2label=dict(enumerate(EVENT_LABELS)),
                                                               label2id=label2id)

    def enc(df):
        d = Dataset.from_pandas(pd.DataFrame({"text": df["text"].tolist(), "label": df["label"].map(label2id).tolist()}))
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=96), batched=True)

    evals = {"gold_v0": ds.gold_v0("event")}
    if ds.gold_real("event") is not None:
        evals["gold_real"] = ds.gold_real("event")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=class_w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(output_dir=str(resolve(a.out)) + "_runs", num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=32, save_strategy="no", logging_steps=50, warmup_ratio=0.06,
                             weight_decay=0.01, fp16=torch.cuda.is_available(), report_to=[])
    trainer = WeightedTrainer(model=model, args=args, train_dataset=enc(train), tokenizer=tok, data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    trainer.save_model(resolve(a.out))
    tok.save_pretrained(resolve(a.out))
    res = {"train_rows": len(train), "eval": {}}
    for name, df in evals.items():
        pred = np.argmax(trainer.predict(enc(df)).predictions, axis=1)
        res["eval"][name] = {"n": len(df), "macro_f1": float(f1_score(df["label"].map(label2id), pred, average="macro"))}
    Path(resolve("docs/results/finetune_deberta_events.json")).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
