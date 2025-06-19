# Детальное проектирование

## Система фильтрации LLM‑контента (BERT Pre‑Filter + 7B Safety‑LLM)

> Версия 1.0 – 11 июня 2025

---

### 1. Обзор архитектуры

```
                 ┌──────────────────────────┐
Client ➜ SDK ➜   │  Gateway Service (FastAPI│
                 │  + uvicorn, ASGI)        │
                 └────────┬───────▲─────────┘
                          │       │
        Cache (safe) ─────┘       │
                                  │ unsafe / uncertain
┌──────────────────────────┐       │
│  Pre‑Filter Service      │◄─────┘
│  (BERT, CPU, ONNXRuntime)│ safe
└────────┬─────────────────┘
         │ unsafe          ┌──────────────────────────┐
         └────────────────►│ Safety‑LLM Service       │
                           │ (LlamaGuard‑7B, vLLM)    │
                           └────────┬─────────────────┘
                                    │ verdict JSON
                           ┌────────▼─────────────────┐
                           │ Verdict Aggregator       │
                           │ (combiner + formatter)   │
                           └────────┬─────────────────┘
                                    │ write
                    ┌───────────────▼───────────────┐
                    │  ClickHouse  (logs, metrics)   │
                    └────────┬───────────────────────┘
                             │ REST / WebSocket
                    ┌────────▼───────────────────────┐
                    │  Web UI (React, shadcn/ui)      │
                    └─────────────────────────────────┘
```

- **Стиль**: микросервисная схема с разделением CPU/GPU‑нагрузки.
- **Сетевые протоколы**: внешне REST/gRPC, внутри gRPC + mTLS (SPIFFE).
- **Развёртывание**: docker‑compose (local dev) и Helm (prod/test) – *описание деплоя не входит в проект, но структура образов заложена*.

---

### 2. Компоненты

| № | Сервис                 | Обязанности                                                                                                    | Технологии / Версии                 |
| - | ---------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 1 | **Gateway Service**    | Экспонирует API `/v1/filter`, валидация, auth, rate‑limit, маршрутизация к Pre‑Filter; агрегация результатов   | FastAPI 0.111, HTTPX 0.27, asyncio  |
| 2 | **Pre‑Filter Service** | ONNX‑модель BERT Tiny/Distil (≈80 M params). Быстрая бинарная классификация `safe/unsafe`                      | onnxruntime 1.18, pydantic 2        |
| 3 | **Safety‑LLM Service** | LlamaGuard‑7B‑instruct, 8‑bit GPT‑Q; запускается через [vLLM](https://github.com/vllm-project/vllm) с KV‑кэшем | vLLM 0.4.0, CUDA 12.4               |
| 4 | **Verdict Aggregator** | Объединяет ответы, формирует финальное JSON‑решение; кеширует положительные verdict‑ы (TTL = 10 мин)           | redis‑py 5 (optional)               |
| 5 | **Log Writer**         | Асинхронная запись результатов и метаданных в ClickHouse через HTTP interface                                  | clickhouse‑connect 0.7              |
| 6 | **Web UI**             | Дашборды и поток запросов; WebSocket подписка на /stream; графики KPI                                          | React 18, Vite, shadcn/ui, recharts |

#### 2.1 Взаимодействие

1. **Client** вызывает `POST /v1/filter`.
2. **Gateway** проводит JWT‑аутентификацию, лимитирует запросы (redis‑based token bucket), передаёт текст в Pre‑Filter.
3. **Pre‑Filter** возвращает `safe` или `unsafe`/`uncertain` + score. При `safe` → Gateway возвращает результат немедленно.
4. При `unsafe|uncertain` Gateway передаёт текст и метаданные в **Safety‑LLM**.
5. Safety‑LLM генерирует JSON ответ с категорией и пояснением.
6. **Aggregator** формирует итоговое решение (`unsafe`) и комментарий, записывает логи, отдаёт клиенту.

---

### 3. Соглашения по данным

#### 3.1 JSON Schema (request/response)

```json
// FilterRequest
{
  "type": "object",
  "required": ["text"],
  "properties": {
    "text": { "type": "string", "minLength": 1, "maxLength": 16384 },
    "llm_id": { "type": "string" }
  }
}

// FilterResponse
{
  "type": "object",
  "required": ["status"],
  "properties": {
    "status": { "enum": ["safe", "unsafe"] },
    "category": { "type": "string" },
    "comment": { "type": "string" },
    "processing_ms": { "type": "number" }
  }
}
```

#### 3.2 ClickHouse Schema

```sql
CREATE TABLE filter_logs (
  ts           DateTime64(3) CODEC(DoubleDelta, LZ4),
  request_id   UUID,
  text_hash    FixedString(32),
  status       LowCardinality(String),
  category     LowCardinality(String),
  llm_id       LowCardinality(String),
  latency_ms   UInt16,
  pre_score    Float32,
  safety_score Float32,
  source_ip    IPv6,
  token_count  UInt16
) ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (ts, status);
```

---

### 4. Модели и пайплайны

| Шаг                       | Детали                                                                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ⚙ **Pre‑Filter Training** | `train_bert.py` – загружает готовый датасет, train\:test\:val = 8:1:1, квантизирует в ONNX‑FP16, сохраняет в `artifacts/pre_filter.onnx`. Переобучение ручным запуском. |
| ⚙ **Safety‑LLM LoRA**     | `finetune_loraguard.sh` – HuggingFace PEFT + LoRA; 8‑bit base + 16‑bit adapter; сохраняет в `artifacts/lora/*.safetensors`.                                             |
| 🔄 **Eval**               | `eval_suite.py` – AEGIS, ToxicChat, internal red‑team; генерирует HTML‑отчёт.                                                                                           |
| 📦 **Packaging**          | `build_image.sh` → `Dockerfile.llm`, `Dockerfile.gateway`, `Dockerfile.prefilter`.                                                                                      |

---

### 5. API слои

#### 5.1 Gateway (FastAPI)

```python
@app.post("/v1/filter", response_model=FilterResponse)
async def filter_text(req: FilterRequest, user=Depends(auth)):
    start = monotonic_ns()
    pre = await prefilter.predict(req.text)
    if pre.safe:
        verdict = {
            "status": "safe",
            "processing_ms": (monotonic_ns()-start)//1e6
        }
        await log_writer.enqueue(verdict, pre.score)
        return verdict
    # forward to safety LLM
    llm_resp = await safety_llm.ask(req.text)
    verdict = parse_llm(llm_resp)
    verdict["processing_ms"] = (monotonic_ns()-start)//1e6
    await log_writer.enqueue(verdict, pre.score, llm_resp.score)
    return verdict
```

*Модуль **``** общается по gRPC; **``** — REST к vLLM server.*

#### 5.2 gRPC Stub (FilterService)

Файл `filter.proto` (идентичен разделу ТЗ). Генерируется `make proto`.

---

### 6. Web UI

| Стек        | Деталь                                                                          |
| ----------- | ------------------------------------------------------------------------------- |
| React 18    | Vite, TypeScript, ESLint, Prettier                                              |
| shadcn/ui   | Card, Table, Tabs, Button                                                       |
| State       | Zustand + React Query                                                           |
| Charts      | Recharts (auto colour palette)                                                  |
| Live‑stream | native WebSocket; server в Gateway публикует события из очереди (Redis pub/sub) |

#### 6.1 Главные компоненты

1. **DashboardPage** – KPI‑карты (RPS, latency P95, unsafe share), графики `LineChart`.
2. **StreamPage** – виртуализированная таблица (react‑virtualized) запросов с инкрементальной загрузкой.
3. **SearchDrawer** – фильтры (date‑range, status, category, llm\_id).
4. **ExportModal** – выбор формата и лимита строк; запрос `GET /v1/logs`.

---

### 7. Безопасность

- **Аутентификация**: JWT (HS256) в заголовке `Authorization: Bearer`.
- **Авторизация**: роли `viewer`, `analyst`, `admin` (RBAC в Gateway + UI).
- **Сетевое шифрование**: TLS 1.3 внешне, внутри mTLS SPIFFE (cert‑manager).
- **Лимиты**: per‑token RPS, per‑IP burst.

---

### 8. Локальная среда разработки

- **Prereq**: Docker 25+, Python 3.11, Node 20.
- `make dev` – поднимает `docker-compose.dev.yml` (gateway+prefilter+llm‑stub+clickhouse).
- `make test` – запускает `pytest` + `mypy`.
- `uvicorn app.main:app --reload` – hot reload gateway.
- `npm run dev` – Web UI.

---

### 9. План задач (MVP)

| № | Epic                  | Issues                                     | Оценка |
| - | --------------------- | ------------------------------------------ | ------ |
| 1 | **Модели**            | портирование BERT → ONNX; скрипт инференса | 3 дн   |
| 2 | **Safety‑LLM сервис** | деплой vLLM, REST‑обвязка                  | 2 дн   |
| 3 | **Gateway**           | END‑to‑END поток + jwt, rate‑limit         | 4 дн   |
| 4 | **Логирование**       | ClickHouse schema, async writer            | 1 дн   |
| 5 | **Web UI**            | Dashboard + Stream + Auth                  | 5 дн   |
| 6 | **Тесты**             | unit, integration, red‑team suite          | 3 дн   |
| 7 | **Документация**      | README, OpenAPI, proto, diagrams           | 1 дн   |

**Итого**: \~19 рабочих дней на MVP одним разработчиком + 1 ML‑инженер.

---

> Конец документа

