# -*- coding: utf-8 -*-
"""Golden-dump: снимает промежуточные артефакты Python-пайплайна ruaccent tiny.

Один запуск полностью регенерирует golden/corpus.txt (если нет) и
golden/tiny.jsonl. Эталон для сверки Rust-порта (ROADMAP этап 1).
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REFERENCE = os.path.join(REPO, ".work", "ai", "2026-08-22-golden-harness", "reference")
WORKDIR = r"E:\TTS\ruaccent-spike\tiny"
CORPUS = os.path.join(REPO, "golden", "corpus.txt")
OUT = os.path.join(REPO, "golden", "tiny.jsonl")

sys.path.insert(0, REFERENCE)

from ruaccent import RUAccent  # noqa: E402
from ruaccent.text_preprocessor import TextPreprocessor  # noqa: E402

DEFAULT_CORPUS = [
    "На двери висит замок.",
    "Старинный замок стоит на горе.",
    "Я плачу за заказ.",
    "Я плачу от горя.",
    "В лесу белки собирают орехи.",
    "В рационе важны белки.",
    "Мука уже стоит в шкафу.",
    "Уже наступило время обеда.",
    "Дверь уже открыта.",
    "Девушка заплела косу.",
    "На поле лежит коса тумана.",
    "Замок был на холме под замком.",
    "А мука в телеге вызывала муку.",
    "",            # пустая строка
    "!!! ...",     # только пунктуация
    "Он произнёс кварделямбу и ушёл.",
    "Он все понял и ушел.",
    "Все были дома.",
    "Она нем устала.",
    "Осел стоял у двора.",
]


def to_py(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    if hasattr(v, "item") and isinstance(v, (int, float)) is False and type(v).__module__ == "numpy":
        return v.item()
    return v


def wrap_session(session, store):
    """Обернуть session.run: фиксировать inputs/outputs без изменения поведения."""
    orig = session.run

    def run(output_names, inputs, *args, **kwargs):
        outputs = orig(output_names, inputs, *args, **kwargs)
        store.append(
            {
                "inputs": {k: to_py(v) for k, v in dict(inputs).items()},
                "outputs": [to_py(o) for o in outputs],
            }
        )
        return outputs

    session.run = run


def main() -> int:
    if os.path.exists(CORPUS):
        with io.open(CORPUS, encoding="utf-8") as f:
            corpus = [line.rstrip("\n").rstrip("\r") for line in f]
    else:
        corpus = list(DEFAULT_CORPUS)
        os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
        with io.open(CORPUS, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus) + "\n")

    accentor = RUAccent()
    accentor.load(
        omograph_model_size="tiny",
        use_dictionary=True,
        tiny_mode=False,
        device="CPU",
        workdir=WORKDIR,
    )

    # --- буферы на текущую фразу ---
    buf = {
        "sentences": [],
        "words_calls": [],
        "stress_usage": [],
        "stress_session": [],
        "yo_calls": [],
        "omograph_session": [],
        "omograph_resolved": [],
        "accent_session": [],
        "accent_words": [],
        "yo_session": [],
        "yo_entities": [],
    }

    orig_split_sent = TextPreprocessor.split_by_sentences
    orig_split_words = TextPreprocessor.split_by_words
    orig_predict_su = accentor.stress_usage_predictor.predict_stress_usage
    orig_yo = accentor._process_yo
    orig_omograph = accentor._process_omographs
    orig_put_accent = accentor.accent_model.put_accent
    orig_predict_yo = accentor.yo_homograph_model.predict_yo_homographs

    @staticmethod
    def patched_split_sent(string):
        res = orig_split_sent.__func__(string) if hasattr(orig_split_sent, "__func__") else orig_split_sent(string)
        buf["sentences"].append(list(res))
        return res

    @staticmethod
    def patched_split_words(string):
        res = orig_split_words.__func__(string) if hasattr(orig_split_words, "__func__") else orig_split_words(string)
        buf["words_calls"].append((list(res[0]), list(res[1])))
        return res

    def patched_predict_su(text):
        res = orig_predict_su(text)
        buf["stress_usage"].append([{k: to_py(x) for k, x in e.items()} if isinstance(e, dict) else e for e in res])
        return res

    def patched_predict_yo(text):
        res = orig_predict_yo(text)
        buf["yo_entities"].append([{k: to_py(x) for k, x in e.items()} if isinstance(e, dict) else e for e in res])
        return res

    def patched_yo(words, sentence):
        res = orig_yo(words, sentence)
        buf["yo_calls"].append({"in": list(words), "out": list(res)})
        return res

    def patched_omograph(words):
        res = orig_omograph(words)
        buf["omograph_resolved"].append(list(res))
        return res

    def patched_put_accent(word):
        # transformers 5.x не возвращает token_type_ids по умолчанию, а ONNX
        # nn_accent их требует; воспроизводим put_accent с явным флагом.
        import numpy as np
        buf["accent_words"].append(word)
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
        pred = [{"label": model.id2label[str(l)], "score": float(sc)} for l, sc in zip(labels, scores)]
        return model.render_stress(word, pred)

    TextPreprocessor.split_by_sentences = patched_split_sent
    TextPreprocessor.split_by_words = patched_split_words
    accentor.stress_usage_predictor.predict_stress_usage = patched_predict_su
    accentor._process_yo = patched_yo
    accentor._process_omographs = patched_omograph
    accentor.accent_model.put_accent = patched_put_accent
    accentor.yo_homograph_model.predict_yo_homographs = patched_predict_yo
    wrap_session(accentor.omograph_model.session, buf["omograph_session"])
    wrap_session(accentor.accent_model.session, buf["accent_session"])
    wrap_session(accentor.yo_homograph_model.session, buf["yo_session"])
    wrap_session(accentor.stress_usage_predictor.session, buf["stress_session"])

    lines = []
    for phrase in corpus:
        for value in buf.values():
            value.clear()
        try:
            final = accentor.process_all(phrase)
        except Exception as exc:  # edge case не должен ломать дамп
            record = {"input": phrase, "error": f"{type(exc).__name__}: {exc}", "final": None}
            lines.append(record)
            print(f"ERROR on {phrase!r}: {exc}", file=sys.stderr)
            continue

        yo_out = buf["yo_calls"][0]["out"] if buf["yo_calls"] else []
        candidates = []
        for i, word in enumerate(yo_out):
            variants = accentor.omographs.get(word)
            if variants:
                candidates.append({"word": word, "variants": variants, "position": i})

        # сверка: final из process_all обязан совпадать с тем, что вернул движок
        record = {
            "input": phrase,
            "sentences": [list(s2) for s2 in buf["sentences"]],
            "words": [list(w) for w, _ in buf["words_calls"]],
            "remaining_text": [list(r) for _, r in buf["words_calls"]],
            "stress_usage": [list(s2) for s2 in buf["stress_usage"]],
            "stress_nn_calls": [
                {"input_ids": c["inputs"].get("input_ids"),
                 "attention_mask": c["inputs"].get("attention_mask"),
                 "token_type_ids": c["inputs"].get("token_type_ids"),
                 "logits": c["outputs"][0]}
                for c in buf["stress_session"]
            ],
            "yo_words": [list(o["out"]) for o in buf["yo_calls"]],
            "yo_nn_calls": [
                {"input_ids": c["inputs"].get("input_ids"),
                 "attention_mask": c["inputs"].get("attention_mask"),
                 "logits": c["outputs"][0]}
                for c in buf["yo_session"]
            ],
            "yo_entities": [list(e) for e in buf["yo_entities"]],
            "omograph_candidates": candidates,
            "omograph_inputs": [
                {"input_ids": c["inputs"].get("input_ids"),
                 "attention_mask": c["inputs"].get("attention_mask"),
                 "token_type_ids": c["inputs"].get("token_type_ids"),
                 "logits": c["outputs"][0]}
                for c in buf["omograph_session"]
            ],
            "omograph_resolved": [list(w) for w in buf["omograph_resolved"]],
            "accent_nn_calls": [
                {"word": w,
                 "input_ids": c["inputs"].get("input_ids"),
                 "attention_mask": c["inputs"].get("attention_mask"),
                 "token_type_ids": c["inputs"].get("token_type_ids"),
                 "logits": c["outputs"][0]}
                for w, c in zip(buf["accent_words"], buf["accent_session"])
            ],
            "final": final,
        }
        lines.append(record)
        print(f"{phrase!r} => {final!r}", file=sys.stderr)

    # снять хуки (не обязательно для процесса, но чисто)
    TextPreprocessor.split_by_sentences = orig_split_sent
    TextPreprocessor.split_by_words = orig_split_words

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        for record in lines:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines)} records to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
