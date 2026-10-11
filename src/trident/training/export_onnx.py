"""Export a Hugging Face sequence classifier (FinBERT) to ONNX and quantize it to INT8.

  python -m trident.training.export_onnx --model models/finbert-ft --out models/finbert-onnx
  python -m trident.training.export_onnx --model ProsusAI/finbert --out models/finbert-onnx   # no fine-tuning

Writes model.onnx, model_quant.onnx (dynamic INT8), the tokenizer and config.json, then
benchmarks both on CPU and checks that INT8 predictions agree with FP32.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trident.config import resolve

SAMPLE = ["JPMorgan downgraded two notches as commercial real estate losses mount",
          "Nvidia posts record revenue on AI chip demand", "Fed holds rates steady"] * 16


def main() -> None:  # pragma: no cover - heavy dependency
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ProsusAI/finbert")
    ap.add_argument("--out", default="models/finbert-onnx")
    a = ap.parse_args()
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).eval()
    enc = tok(["example sentence"], return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    torch.onnx.export(model, tuple(enc[k] for k in names), str(out / "model.onnx"), input_names=names, output_names=["logits"],
                      dynamic_axes={**{k: {0: "batch", 1: "seq"} for k in names}, "logits": {0: "batch"}}, opset_version=17)
    quantize_dynamic(str(out / "model.onnx"), str(out / "model_quant.onnx"), weight_type=QuantType.QInt8)
    tok.save_pretrained(out)
    model.config.save_pretrained(out)

    batch = tok(SAMPLE, return_tensors="np", padding=True, truncation=True, max_length=96)
    res = {}
    preds = {}
    for f in ("model.onnx", "model_quant.onnx"):
        sess = ort.InferenceSession(str(out / f), providers=["CPUExecutionProvider"])
        feed = {k: batch[k].astype(np.int64) for k in names}
        sess.run(None, feed)
        t = time.perf_counter()
        for _ in range(5):
            logits = sess.run(None, feed)[0]
        ms = (time.perf_counter() - t) / 5 / len(SAMPLE) * 1e3
        preds[f] = logits.argmax(1)
        res[f] = {"ms_per_item_batched": round(ms, 3), "size_mb": round((out / f).stat().st_size / 1e6, 1)}
    res["int8_agreement"] = float(np.mean(preds["model.onnx"] == preds["model_quant.onnx"]))
    Path(resolve("docs/results")).mkdir(parents=True, exist_ok=True)
    Path(resolve("docs/results/onnx_export.json")).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
