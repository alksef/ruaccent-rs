# -*- coding: utf-8 -*-
"""Замер Python-эталона RUAccent tiny на корпусе golden/corpus.txt.

Одна машина, тот же корпус, что и Rust-бенч `benches/pipeline.rs`
(ROADMAP этап 6, часть 2). Загрузка один раз; прогрев — 2 полных прохода
по корпусу; затем N=10 проходов с замером `time.perf_counter` вокруг КАЖДОГО
вызова `process_all` (одна строка корпуса за вызов, включая пустую строку).

Вывод: таблица per-phrase mean (мс), итоговые p50/p95/mean по всем сэмплам
и строка `RESULT p50=..ms p95=..ms mean=..ms`.
"""
from __future__ import annotations

import io
import math
import os
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REFERENCE = os.path.join(REPO, ".work", "ai", "2026-08-22-golden-harness", "reference")
WORKDIR = r"E:\TTS\ruaccent-spike\tiny"
CORPUS = os.path.join(REPO, "golden", "corpus.txt")
N_PASSES = 10
WARMUP_PASSES = 2

sys.path.insert(0, REFERENCE)

from ruaccent import RUAccent  # noqa: E402


def nearest_rank(sorted_values, percentile):
    """Nearest-rank перцентиль (без сторонних зависимостей)."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(percentile / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


def main() -> int:
    with io.open(CORPUS, encoding="utf-8") as f:
        corpus = [line.rstrip("\n").rstrip("\r") for line in f]

    accentor = RUAccent()
    accentor.load(
        omograph_model_size="tiny",
        use_dictionary=True,
        tiny_mode=False,
        device="CPU",
        workdir=WORKDIR,
    )

    # transformers 5.x не возвращает token_type_ids по умолчанию, а ONNX
    # nn_accent их требует; тот же workaround, что и в tools/golden_dump.py.
    # Поведение идентично оригинальному put_accent: session.run(None, inputs)
    # + softmax + render_stress, добавляется лишь явный return_token_type_ids.
    import numpy as np

    def patched_put_accent(word):
        model = accentor.accent_model
        inputs = model.tokenizer(word.lower(), return_tensors="np", return_token_type_ids=True)
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        outputs = model.session.run(None, inputs)
        output_names = {o.name: idx for idx, o in enumerate(model.session.get_outputs())}
        logits = outputs[output_names["logits"]]
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        scores = np.max(probs, axis=-1)[0]
        labels = np.argmax(logits, axis=-1)[0]
        pred = [
            {"label": model.id2label[str(l)], "score": float(sc)}
            for l, sc in zip(labels, scores)
        ]
        return model.render_stress(word, pred)

    accentor.accent_model.put_accent = patched_put_accent

    for _ in range(WARMUP_PASSES):
        for phrase in corpus:
            accentor.process_all(phrase)

    # phrase_text -> list of ms per single call
    samples: dict[str, list[float]] = {}
    for _ in range(N_PASSES):
        for phrase in corpus:
            t = time.perf_counter()
            accentor.process_all(phrase)
            samples.setdefault(phrase, []).append((time.perf_counter() - t) * 1000.0)

    print("per-phrase mean (ms):")
    for phrase in corpus:
        mean = statistics.mean(samples[phrase])
        print(f"  {mean:8.3f}  {phrase!r}")

    all_ms = [v for values in samples.values() for v in values]
    all_ms_sorted = sorted(all_ms)
    p50 = nearest_rank(all_ms_sorted, 50)
    p95 = nearest_rank(all_ms_sorted, 95)
    mean = statistics.mean(all_ms)

    n = len(all_ms)
    print(f"\nN={n} samples, per-call process_all")
    print(f"p50 = {p50:.3f} ms")
    print(f"p95 = {p95:.3f} ms")
    print(f"mean = {mean:.3f} ms")
    print(f"RESULT p50={p50:.3f}ms p95={p95:.3f}ms mean={mean:.3f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
