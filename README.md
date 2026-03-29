---
title: DataWarehouseOps-Env
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏭 DataWarehouseOps-Env
> **An OpenEnv-compliant Reinforcement Learning environment** where AI agents learn to perform real-world enterprise data engineering tasks inside a live SQL sandbox.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-orange)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Why This Environment Exists

Every tech company on earth runs a data warehouse, and every data warehouse has the same chronic problems: **dirty data, exposed PII, and slow queries**. These problems cost billions of dollars annually in manual Data Engineering hours.

There are currently **no RL benchmarks** for data engineering agentic tasks. `DataWarehouseOps-Env` fills this gap by providing a rigorous, reproducible sandbox where an AI agent can:

- Execute SQL against a **live, isolated SQLite database** (fresh per episode)
- Train on **REAL world enterprise data** (Microsoft Northwind and Chinook databases, up to 609,000 rows!)
- Receive **dense per-step rewards** for good SQL and penalties for bad SQL (including loop detection)
- Be graded by a **100% deterministic, LLM-free grader** that checks actual database state

---

## ✨ The Interactive Dashboard

Unlike typical CLI-only environments, this project includes a **stunning, production-grade Web UI** built into the FastApi server. 
- **AI Auto-Solver Simulation:** Watch an AI literally "ghost type" real SQL commands to solve the environment live.
- **Real-time Scoring & Confetti:** Watch the metrics update live, culminating in a celebration when the AI scores perfectly!
- Run it: Visit `http://localhost:7860/` in your browser.

---

## 📐 Environment Description

### Architecture
```
AI Agent
   │
   ▼ HTTP (JSON)
FastAPI Server  ←→  In-Memory SQLite DB (fresh per session)
   │                     │
   │              Seed SQL loaded on reset()
   │
   ▼
Deterministic Graders (task1_grader, task2_grader, task3_grader)
```

- **No shared state** between sessions — each `reset()` spin up a brand-new `:memory:` SQLite DB
- **Pure stdlib** — no heavy ML dependencies in the server (SQLite is Python built-in)
- **Scales horizontally** — stateless HTTP, multiple Uvicorn workers, sessions stored per-process

---

## 🗺️ Action Space

Every action is a JSON object sent to `POST /step`:

| Field | Type | Required | Description |
|---|---|---|---|
| `sql_command` | `string` | No | Any valid SQL: `SELECT`, `UPDATE`, `CREATE VIEW`, `CREATE INDEX`, `EXPLAIN QUERY PLAN` |
| `finalize_task` | `boolean` | No | Set `true` to end the episode and trigger the grader |
| `reasoning` | `string` | No | Agent's chain-of-thought (logged for research, not graded) |
| `session_id` | `string` | **Yes** | Token returned by `/reset` |

**Blocked SQL:** `DROP TABLE`, `TRUNCATE`, bare `DELETE` (no WHERE clause) → returns penalty and blocks execution.

---

## 👁️ Observation Space

Every `step()` and `reset()` returns:

| Field | Type | Description |
|---|---|---|
| `task_id` | `string` | Active task identifier |
| `task_description` | `string` | Full natural-language task objective |
| `schema_info` | `dict` | All tables/views with columns, types, row counts |
| `query_result` | `list[dict]` | Rows from the last `SELECT` (max 50 rows) |
| `rows_affected` | `int` | Rows modified by last DML |
| `error_message` | `string\|null` | SQL error if applicable |
| `step_reward` | `float` | Reward from this action |
| `total_reward` | `float` | Cumulative episode reward |
| `current_step` | `int` | Steps taken so far |
| `max_steps` | `int` | Episode step limit |
| `done` | `bool` | Whether episode has ended |
| `episode_id` | `string` | Unique episode UUID |
| `info` | `dict` | Grader breakdown, hints, warnings |

---

## 📋 Tasks

### 🟢 Task 1 — Enterprise Data Cleaning (`task1_data_cleaning`) — Easy

**Scenario:** An `employee_records` table was imported from a legacy HRIS system with classic dirty-data issues (**2,000 rows** of real Northwind-derived data).

**The agent must:**
1. Normalize all gender values → exactly `'Male'`, `'Female'`, or `'Unknown'`
2. Convert all broken `birth_date` and `hire_date` formats → ISO-8601 `YYYY-MM-DD`
3. Replace all `NULL` country values → `'Unknown'`
4. **Never delete any rows**

**Grader (deterministic SQL checks):**
- Gender normalization → `0.30` weight
- Date format compliance → `0.30` for birth, `0.20` for hire
- Null country coverage → `0.20` weight
- No data loss (row count ≥ 2000) → `0.00` (required penalty if absent)

**Expected GPT-4o score:** ~`0.75–0.90`

---

### 🟡 Task 2 — PII Privacy Shielding (`task2_pii_masking`) — Medium

**Scenario:** A `customers_pii` table has raw emails, credit cards, and SSNs inserted into real **Chinook** database records (**500 rows**). GDPR requires that data scientists only see masked data.

**The agent must:**
Create a SQL VIEW named exactly `masked_customers` that:
- Email: `alice.johnson@gmail.com` → `***@gmail.com`
- Credit card: `4532-1234-5678-9012` → `****-****-****-9012`
- SSN: `123-45-6789` → `*****-**-89`
- Keep `id`, `first_name`, `last_name`, `company`, `city`, `country`, `signup_year`, `risk_tier` unchanged

**Grader (regex pattern matching on view output):**
- Email masking → `0.25`
- Credit card masking → `0.25`
- SSN masking → `0.25`
- Safe column integrity → `0.15`
- Row count match → `0.10`

**Expected GPT-4o score:** ~`0.85–1.00`

---

### 🔴 Task 3 — Query Optimization & Indexing (`task3_query_optimization`) — Hard

**Scenario:** A financial reporting query over **100,000 `sales_transactions` rows** (sampled from a 609,000-row Northwind expansion) is doing a full table scan. The agent must diagnose it and optimize it.

**The agent must:**
1. Run `EXPLAIN QUERY PLAN` on the slow query
2. Identify that `WHERE sale_date BETWEEN ... AND ... AND status = 'completed'` needs an index
3. `CREATE INDEX idx_sales_date_status ON sales_transactions(sale_date, status);`
4. Re-verify with `EXPLAIN QUERY PLAN` that `SEARCH` replaces `SCAN`

**Grader (EXPLAIN QUERY PLAN inspection):**
- Index on `sale_date` column → `0.30`
- Index on `status` column → `0.20`
- EXPLAIN shows SEARCH not SCAN → `0.35`
- Query returns correct results → `0.15`

**Expected GPT-4o score:** ~`0.55–0.85`

---

## 💰 Reward Function

| Event | Reward |
|---|---|
| Successful `SELECT` / `EXPLAIN` (exploration) | `+0.02` |
| Successful `UPDATE` / `CREATE VIEW` / `CREATE INDEX` | `+0.05` |
| Invalid SQL syntax error | `-0.10` |
| Empty `sql_command` | `-0.05` |
| Dangerous command (`DROP TABLE`, etc.) | `-0.50` |
| Repeated command (loop detection) | `-0.10` |
| Grader score on episode completion | `+score` (0.0 – 1.0) |

All rewards clipped to `[-1.0, 1.0]`.

---

## 📊 Baseline Scores

Run by the included `baseline.py` script using `gpt-4o-mini`:

| Task | Score | Difficulty |
|---|---|---|
| `task1_data_cleaning` | ~`0.80` | Easy 🟢 |
| `task2_pii_masking` | ~`0.90` | Medium 🟡 |
| `task3_query_optimization` | ~`0.65` | Hard 🔴 |

The `/baseline` endpoint also runs a built-in **deterministic heuristic agent** (no LLM) that scores perfectly on all 3 tasks, confirming the environment is solvable.

---

## 🚀 Setup & Usage

### Local (Python)
```bash
cd data-warehouse-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker
```bash
docker build -t datawarehouse-env .
docker run -p 7860:7860 datawarehouse-env
```

### HuggingFace Space
Simply push this repo to a HF Space with `sdk: docker` in `README.md`.

### Test it works
```bash
curl http://localhost:7860/health
# {"status": "healthy", "environment": "DataWarehouseOps-Env", "version": "1.0.0"}

curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
     -d '{"task_id": "task1_data_cleaning"}'
```

### Python Client
```python
from client import DataWarehouseEnv, DataWHAction

with DataWarehouseEnv(base_url="http://localhost:7860") as env:
    obs = env.reset(task_id="task1_data_cleaning")
    print(obs.task_description)

    result = env.step(DataWHAction(
        sql_command="SELECT gender, COUNT(*) FROM customer_records GROUP BY gender;"
    ))
    print(result.observation.query_result)
```

### Run LLM Baseline
```bash
export OPENAI_API_KEY=sk-your-key
export DATAWAREHOUSE_ENV_URL=http://localhost:7860
python baseline.py
```

---

## 📁 Project Structure

```
data-warehouse-env/
├── server/
│   ├── app.py              ← FastAPI server (all endpoints)
│   ├── environment.py      ← Core logic: SQL execution, reward, state
│   ├── graders/
│   │   ├── task1_grader.py ← Deterministic grader: data cleaning
│   │   ├── task2_grader.py ← Deterministic grader: PII masking
│   │   └── task3_grader.py ← Deterministic grader: query optimization
│   └── data/
│       ├── task1_seed.sql  ← Dirty customer data (20 rows)
│       ├── task2_seed.sql  ← PII customer data (10 rows)
│       └── task3_seed.sql  ← Sales transactions (2000 rows)
├── models.py               ← Pydantic typed models (Action, Observation, State)
├── client.py               ← Python HTTP client for RL training loops
├── baseline.py             ← OpenAI API baseline inference script
├── openenv.yaml            ← OpenEnv manifest
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 🤖 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/reset` | POST | Start new episode, get initial observation |
| `/step` | POST | Execute SQL action, get observation + reward |
| `/state` | GET | Get current episode metadata |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | POST | Finalize episode, get grader score |
| `/baseline` | POST | Run deterministic heuristic baseline |
| `/docs` | GET | Interactive Swagger UI |

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
