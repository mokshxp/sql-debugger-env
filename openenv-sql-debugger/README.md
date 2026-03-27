# 🔍 SQL Debugger — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **real-world OpenEnv environment** where AI agents debug, fix, and optimize SQL queries against a live SQLite database simulating an e-commerce platform.

---

## Why This Environment?

Data engineers and analysts spend a significant portion of their time debugging faulty SQL — queries that fail with errors, return wrong results, or perform poorly at scale. This is a **genuine, high-value task** where AI agents could provide immediate practical value. Unlike toy environments:

- The database has **realistic relational structure** (5 tables, foreign keys, indexes)
- Bugs are **representative of real mistakes**: wrong aggregation, missing filters, wrong JOIN type, incorrect window functions
- The grading system mirrors how a **real code reviewer** evaluates SQL: correctness first, then efficiency

---

## Environment Description

### Database Schema

An e-commerce SQLite database with:

```sql
customers(customer_id PK, name, email, country, joined_date)
products(product_id PK, name, category, price, stock)
orders(order_id PK, customer_id FK, order_date, status)
  -- status ∈ {pending, shipped, delivered, cancelled}
order_items(item_id PK, order_id FK, product_id FK, quantity, unit_price)
reviews(review_id PK, customer_id FK, product_id FK, rating 1-5, review_date)
```

Seeded with 7 customers, 10 products, 12 orders, 22 order items, 10 reviews — enough to produce meaningful SQL results without overwhelming context windows.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `db_schema` | string | DDL for all tables |
| `business_requirement` | string | Natural language spec of what the query must compute |
| `current_query` | string | Most recently submitted SQL |
| `execution_result.success` | bool | Whether the query ran without error |
| `execution_result.rows` | list[dict] | Result rows |
| `execution_result.error` | string\|null | DB error message if any |
| `execution_result.row_count` | int | Number of rows returned |
| `feedback` | string | Partial grader feedback (pipe-separated checks) |
| `step` | int | Current step (0-indexed) |
| `max_steps` | int | Episode step budget |
| `score` | float | Best score achieved this episode (0.0–1.0) |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `sql` | string | The SQL query to execute and grade |

The agent rewrites the SQL query each step. There are no other action types — the environment is query-in, result-out.

---

## Tasks

### Task 1 — SQL Syntax Fix `task_easy` 🟢

**Difficulty:** Easy | **Budget:** 10 steps | **Threshold:** 0.8

**Business Requirement:**
> Show total revenue (sum of `quantity × unit_price`) per product **category** for **DELIVERED** orders only, ordered from highest revenue to lowest. Columns: `category`, `total_revenue`.

**Bugs planted in starting query:**
1. `SUM(oi.quantity)` — sums quantity, not revenue
2. Missing `WHERE o.status = 'delivered'` filter
3. `ORDER BY total_revenue` with no `DESC`

**Grader checks (each worth ~0.2):**
- Has `category` column
- Revenue column present with price-scale values
- Correct 3 categories returned
- `WHERE status='delivered'` applied
- Ordered `DESC`

---

### Task 2 — SQL Logic Repair `task_medium` 🟡

**Difficulty:** Medium | **Budget:** 15 steps | **Threshold:** 0.7

**Business Requirement:**
> Find customers who have placed **at least 2 delivered orders**. Show: `name`, `delivered_orders`, `total_spent`. Order by `total_spent` descending.

**Bugs planted in starting query:**
1. Spurious `LEFT JOIN reviews` — inflates row counts
2. `COUNT(*)` counts rows not distinct orders
3. `HAVING COUNT(*) >= 1` — wrong threshold (should be ≥ 2)

**Correct answer:** Alice Smith (2 delivered orders), Dave Brown (2 delivered orders)

**Grader checks:**
- Correct customer set (Alice + Dave only)
- `HAVING COUNT >= 2`
- Reviews table not joined
- Uses `COUNT(DISTINCT o.order_id)`
- Ordered by `total_spent DESC`

---

### Task 3 — SQL Optimization & Correctness `task_hard` 🔴

**Difficulty:** Hard | **Budget:** 20 steps | **Threshold:** 0.6

**Business Requirement:**
> For each product sold in at least one delivered order, produce:
> - `name`, `category`
> - `total_units_sold` — quantity across delivered orders
> - `avg_rating` — average review rating (**NULL** if no reviews)
> - `revenue_rank` — rank within category by revenue using **`RANK()`** (not `ROW_NUMBER`)
> - `category_revenue_pct` — this product's revenue as % of category total
>
> Order by `category`, then `revenue_rank`.

**Bugs planted in starting query:**
1. `ROW_NUMBER()` instead of `RANK()` — breaks tie handling
2. Missing `PARTITION BY category` in window — rank is global not per-category
3. `category_revenue_pct` divides by `p.price` instead of category total revenue
4. No `WHERE o.status = 'delivered'` — includes cancelled orders
5. `INNER JOIN reviews` drops products with no reviews

**Grader checks:** column presence, delivered filter, LEFT JOIN reviews, RANK() + PARTITION BY, correct window denominator, result sanity

**Reward** on hard task = `0.7 × correctness + 0.3 × efficiency` (measured via SQLite EXPLAIN QUERY PLAN).

---

## Reward Function

```
reward = clamp(raw_value + step_penalty, 0.0, 1.0)

raw_value =
  easy/medium:  correctness_score
  hard:         0.7 × correctness + 0.3 × efficiency

step_penalty = -0.01 × step_number   # discourages thrashing
```

**Key properties:**
- **Dense signal** — grader evaluates multiple independent checks; each fix is rewarded immediately
- **Partial progress** — fixing one bug out of three yields ~0.4, not 0.0
- **Step penalty** — mild penalty grows with each step, rewarding concise solutions
- **Efficiency proxy** — for the hard task, query plan analysis rewards index usage and penalizes full table scans

---

## Setup & Usage

### Local development

```bash
git clone https://github.com/YOUR_USER/openenv-sql-debugger
cd openenv-sql-debugger
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Open the playground
open http://localhost:7860
```

### Docker

```bash
docker build -t sql-debugger-env .
docker run -p 7860:7860 sql-debugger-env

# Verify
curl http://localhost:7860/health
```

### Python API (direct)

```python
from env.environment import SQLDebuggerEnv
from env.models import Action

env = SQLDebuggerEnv(task_id="task_easy")

# OpenEnv spec
obs            = env.reset()
obs, rew, done, info = env.step(Action(sql="SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM orders o JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id WHERE o.status = 'delivered' GROUP BY p.category ORDER BY total_revenue DESC"))
state          = env.state()

print(f"Score: {info['score']:.3f}")
print(f"Feedback: {obs.feedback}")
```

### HTTP API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM orders o JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id WHERE o.status = '\''delivered'\'' GROUP BY p.category ORDER BY total_revenue DESC"}'

# State
curl http://localhost:7860/state?task_id=task_easy

# Validate
curl http://localhost:7860/validate
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Baseline Inference

```bash
# Requires OPENAI_API_KEY
OPENAI_API_KEY=sk-... python baseline/run_baseline.py

# Single task
OPENAI_API_KEY=sk-... python baseline/run_baseline.py --task task_hard

# Use a different model
OPENAI_API_KEY=sk-... python baseline/run_baseline.py --model gpt-4o

# Against remote HF Space
OPENAI_API_KEY=sk-... python baseline/run_baseline.py \
  --env-url https://YOUR_SPACE.hf.space
```

---

## Baseline Scores

Measured with **GPT-4o-mini** (`gpt-4o-mini-2024-07-18`), temperature=0.2, single run:

| Task | Difficulty | Best Score | Steps Used | Notes |
|------|-----------|-----------|-----------|-------|
| `task_easy` | 🟢 Easy | **0.90** | 2–3 | Model reliably identifies all 3 bugs |
| `task_medium` | 🟡 Medium | **0.72** | 4–6 | Usually fixes JOIN + HAVING; sometimes misses `COUNT(DISTINCT)` |
| `task_hard` | 🔴 Hard | **0.58** | 8–12 | Fixes delivered filter + LEFT JOIN; struggles with window pct formula |
| **Average** | | **0.733** | | |

Scores are reproducible with the provided `baseline/run_baseline.py` script.

---

## Project Structure

```
openenv-sql-debugger/
├── openenv.yaml              # OpenEnv metadata spec
├── app.py                    # FastAPI server + HTML playground
├── Dockerfile                # Container for HF Spaces deployment
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py             # Pydantic: Observation, Action, Reward
│   ├── database.py           # SQLite schema + seed data
│   └── environment.py        # SQLDebuggerEnv (step/reset/state)
├── tasks/
│   ├── __init__.py
│   └── tasks.py              # Task definitions + graders
├── baseline/
│   ├── __init__.py
│   └── run_baseline.py       # OpenAI-based baseline agent
└── tests/
    └── test_environment.py   # Pytest suite (spec compliance, grader determinism)
```

---

## Hugging Face Spaces Deployment

1. Create a new HF Space (Docker SDK)
2. Push this repository:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USER/sql-debugger-env
git push hf main
```

3. The Space will build and serve on port 7860 automatically.

Add the `openenv` tag in the Space settings to be discoverable via OpenEnv search.

---

## Design Notes

**Why SQLite?** Zero setup, runs in-process inside the container, perfect for an evaluation environment where we want deterministic, reproducible state that can be torn down and rebuilt on every `reset()`.

**Why e-commerce?** The domain is universally understood — no domain-specific knowledge required to evaluate whether an agent understood the task. Revenue by category, customer spend, product rankings are self-explanatory requirements.

**Why these three bugs?** They represent a natural progression:
- Easy: *syntactic / obvious semantic* (wrong expression, missing clause)
- Medium: *logical* (spurious JOIN that inflates results, wrong threshold)
- Hard: *expert-level* (window function semantics, NULL handling, efficiency)

A competent junior analyst solves Task 1 in seconds, Task 2 with some thought, and Task 3 requires understanding of SQL window functions and query planning.

---

## License

MIT © 2024 OpenEnv SQL Debugger
