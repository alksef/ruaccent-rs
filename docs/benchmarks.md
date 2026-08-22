# Бенчмарки: Rust vs Python (ROADMAP этап 6, часть 2)

Замер `RuAccent.process_all` (tiny) на одной машине: Rust-бенч
`benches/pipeline.rs` (criterion) и Python-эталон `tools/bench_python.py`
на том же корпусе `golden/corpus.txt`.

## Окружение

- Дата: 2026-08-23
- CPU: AMD Ryzen 7 5700X 8-Core Processor
- Конфиг: workdir `E:\TTS\ruaccent-spike\tiny`, 4 ONNX-модели (nn_yo,
  nn_omograph/tiny, nn_accent, nn_stress_usage_predictor), один поток,
  `use_dictionary=True`, `tiny_mode=False`, device CPU.

## Rust (criterion, медианы)

`RUACCENT_TINY_WORKDIR=E:\TTS\ruaccent-spike\tiny cargo bench` — группа
`process_all`, sample_size 30, measurement_time 3 s, warm_up 1 s.

| Бенч | Медиана, мс | 95% CI, мс |
|---|---|---|
| phrase «Я плачу за заказ.» | 2.983 | 2.932–3.036 |
| phrase «Замок был на холме под замком.» | 4.096 | 3.764–4.395 |
| phrase «А мука в телеге вызывала муку.» | 5.907 | 5.844–5.990 |
| phrase «Он произнёс кварделямбу и ушёл.» | 4.287 | 4.203–4.390 |
| corpus (19 непустых строк корпуса за итерацию) | 88.837 | 87.740–90.077 |

## Python (эталон)

`tools/bench_python.py`: загрузка один раз, прогрев 2 прохода по корпусу,
затем 10 проходов, `time.perf_counter` вокруг каждого вызова
`process_all` (по одной строке за вызов). N=200 сэмплов (20 строк × 10).

| Метрика | мс |
|---|---|
| p50 | 9.339 |
| p95 | 35.077 |
| mean | 13.538 |

Per-phrase mean по общим фразам: «Я плачу за заказ.» — 8.350 мс,
«Замок был на холме под замком.» — 13.766 мс, «А мука в телеге
вызывала муку.» — 20.455 мс, «Он произнёс кварделямбу и ушёл.» —
10.162 мс.

## Вывод против таргетов ROADMAP

- Цель: Rust не медленнее Python (справочные числа Python tiny из research:
  p50 7.6 мс, p95 31 мс на предложение).
- Перепроверка нашим скриптом: Python p50 = 9.34 мс, p95 = 35.08 мс —
  близко к справочным (небольшой разброс машинного прогона), порядок тот же.
- Rust существенно быстрее Python на каждой фразе (2.9–5.9 мс против
  8.4–20.5 мс) и на проходе по корпусу (88.8 мс против ~271 мс на 20 строк;
  у Rust — 19 непустых строк). Критерий ROADMAP выполнен.

Замечание о честности замера: Python-эталон использует тот же
monkey-patch `put_accent` с явным `return_token_type_ids=True`, что и
`tools/golden_dump.py` (transformers 5.x не отдаёт `token_type_ids`
по умолчанию, а ONNX nn_accent их требует); вычисления идентичны
оригинальному `put_accent` (`session.run` + softmax + `render_stress`).
