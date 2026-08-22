---
id: RA-001
status: completed
created: 2026-08-22
updated: 2026-08-23
license: код и модели ruaccent — MIT (Denis Petrov); сохранять notice в pack
---

# ruaccent-rs — нативный Rust-порт RUAccent

## Цель

Полностью воспроизвести пайплайн RUAccent (`process_all`, режим `tiny` +
словарь) на Rust без Python sidecar: тот же вход, тот же `+`-выход, попиксельная
сверяемость с Python-эталоном на golden-корпусе. Проект — библиотека для
app-tts-v2 (ROADMAP-081, этап 2), не приложение.

Контракт с приложением — НЕ `+`-строка, а позиционной разметки дело app-tts-v2;
библиотека сначала даёт эталонный string output, слой аннотаций поверх.

## Нефиксированные решения

- Рендер: только `+`-нотация (как в оригинале) на первый инкремент.
- `tiny_mode` (без stress_usage_predictor) — оптимизация второго инкремента;
  сначала полный путь с 4 моделями.

## Этапы

### 1. Golden-корпус и harness эталона

- Скрипт (Python, живёт в `tools/`) прогоняет эталонный ruaccent из
  `E:\TTS\ruaccent-bench\python` на корпусе и снимает промежуточные артефакты:
  разбиение на предложения/слова, input_ids/attention_mask каждой ONNX-сессии,
  logits, итоговую строку. Формат — JSON Lines.
- Корпус: 13 фраз бенчмарка + омографный regression-набор ROADMAP-081 +
  edge-cases (пунктуация, `е/ё`, числа, пустые строки, `skip_regex`).
- Готовность: один запуск `tools/dump_golden.py` регенерирует эталон.

### 2. Текстовый слой

- Порт `text_split.py` / `text_preprocessor.py` / `text_postprocessor.py`:
  razdel-совместимый сплиттер предложений, разбиение по словам, склейка
  `remaining_text`, `fix_capital`, `delete_spaces_before_punc`.
- regex через `fancy-regex` (в оригинале есть lookahead).
- Сверка: строковые артефакты против golden, попакетно.

### 3. Словари и конфиги

- `accents.json.gz`, `omographs.json.gz`, `yo_*.json.gz` (23 MB) →
  `flate2` + `serde_json`, один загрузчик с ленивой инициализацией.
- `custom_dict` / `custom_homographs` как API-параметры (приоритет над файлами).
- koziev/RuleEngine НЕ переносится (мёртвый код в `process_all`, см. research).

### 4. CharTokenizer + nn_accent

- Порт CharTokenizer (vocab.txt, pad/unk/bos) — вручную, ~100 строк.
- ONNX-сессия через `ort`; softmax, `render_stress` (порог 0.55), сверка
  попозиционно против golden input_ids и logits (допуск на float).

### 5. BERT-модели: `tokenizers` + 3 сессии

- nn_omograph (NLI pair-encoding, padding/truncation 512, группировка гипотез
  `group_words`/`transfer_grouping`), nn_stress_usage_predictor
  (offset_mapping, агрегация слов AVERAGE), nn_yo_homograph_resolver.
- Токенизация — официальный crate `tokenizers` с `tokenizer.json` из pack'а.
- Сверка: input_ids/attention_mask побитово, выбор омографа — по фразам.

### 6. Пайплайн `process_all` целиком

- Сшивка `_process_yo → _process_omographs → _process_accent`, `skip_regex`.
- Критерий: 100% совпадение строк с golden-корпусом; расхождение = баг
  (или задокументированное датируемое решение).
- `cargo bench` / criterion: зафиксировать, что скорость не хуже Python
  (Python tiny: p50 7.6 ms, p95 31 ms/предложение).

### 7. Упаковка и лицензии

- Pack layout `nn/ + dictionary/ + manifest.json` совместимый с discovery
  ROADMAP-081 (SHA-256, размеры, license notices MIT).
- README: происхождение (порт ruaccent, MIT, Denis Petrov), инструкция
  сборки pack'а из HF `ruaccent/accentuator`.
- Крейт `ruaccent-rs` опубликовать не раньше стабильного API.

## Итог (2026-08-23)

Все этапы 1–7 выполнены. Golden-parity: разбиение текста, словари,
input_ids (побитово), logits (1e-4), выбор омографов, итоговые строки
process_all — 37 тестов. Бенчмарки: docs/benchmarks.md (~2–3× быстрее
Python-эталона). Pack-инструмент: `cargo run --bin pack`.

## Порядок и трекинг

Этапы строго последовательны 1→6 (каждый следующий сверяется предыдущим),
7 параллельно с 6. Отдельная задача вне скоупа: Silero HomoSolver R&D и
Piper IPA-путь (ruphon, research-14) — здесь не делаем.
