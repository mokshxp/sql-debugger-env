---
title: SQL Debugger OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - sql
  - debugging
  - data-engineering
  - reinforcement-learning
  - agent-evaluation
short_description: Real-world SQL debugging environment for AI agents (OpenEnv)
---

# 🔍 SQL Debugger — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Framework-blue.svg)](https://github.com/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Powered by FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-009688.svg?style=flat&logo=Fastapi&logoColor=white)](https://fastapi.tiangolo.com)

A real-world OpenEnv environment where AI agents are challenged to debug, repair, and optimize SQL queries against a live SQLite database. This environment simulates the high-stakes work of data engineers and analysts, requiring both technical precision and performance awareness.

## 🚀 Overview

The SQL Debugger environment provides a sandbox containing a complex e-commerce database (Customers, Orders, Products, Reviews, etc.). Agents must satisfy specific business requirements by fixing broken queries or optimizing inefficient ones.

### Key Challenges
- **Syntax Errors:** Identifying and fixing broken SQL syntax.
- **Logical Bugs:** Correcting improper JOINs, incorrect aggregations, or missing clauses.
- **Optimization:** Transforming slow, expensive queries into performant ones using indexes and window functions.
- **Constraint Satisfaction:** Ensuring the result set exactly matches the business requirements.

## 🛠️ Features

- **Live SQLite Backend:** Real data execution with immediate feedback.
- **Multi-Level Tasks:** Three distinct difficulty levels:
  - 🟢 **Easy (Syntax Fix):** Identifying typos and basic semantic errors.
  - 🟡 **Medium (Logic Repair):** Fixing subtle JOIN errors and aggregation logic.
  - 🔴 **Hard (Optimization):** Improving performance while maintaining 100% correctness.
- **Rich Observations:** Agents receive the full DDL (Schema), the business requirement, previous execution results (rows/errors), and descriptive feedback.
- **Built-in Playground:** A sleek FastAPI-based web interface for human evaluation and debugging.
- **OpenEnv Standards:** Fully compliant with the OpenEnv protocol (`/reset`, `/step`, `/tasks`, `/validate`).

## 📥 Installation

### Prerequisites
- Python 3.9+
- `pip` or `uv`

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/sql-debugger-openenv.git
cd sql-debugger-openenv

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Server
Start the OpenEnv environment server:
```bash
python app.py
```
This will launch the playground at `http://localhost:8000`.

### Running an Agent (Inference)
To run the baseline baseline inference script (requires OpenAI/Proxy environment variables):
```bash
# Ensure API_BASE_URL and API_KEY are set
python inference.py
```

## 📡 API Specification (OpenEnv)

The environment implements the standard OpenEnv REST API:

- **`GET /tasks`**: Returns a list of available SQL debugging tasks.
- **`POST /reset`**: Initializes a task.
- **`POST /step`**: Submits a SQL query and returns the observation, reward, and completion status.
- **`GET /validate`**: Self-test endpoint to ensure environment integrity.

### Action Space
The agent submits a `sql` string:
```json
{
  "task_id": "task_easy",
  "sql": "SELECT name FROM customers WHERE id = 1;"
}
```

### Observation Space
The agent receives:
- `db_schema`: The full DDL of the database.
- `business_requirement`: A text description of the goal.
- `execution_result`: Data returned (or error encountered) by the SQL engine.
- `feedback`: Heuristic-based guidance on correctness.

## 🏗️ Project Structure

```text
├── app.py              # FastAPI Server & Web UI
├── inference.py        # Baseline Agent Implementation
├── openenv.yaml        # OpenEnv Configuration
├── env/
│   ├── environment.py  # Core SQL Environment Logic
├── tasks/
│   └── tasks.py        # Task Definitions & Business Logic
├── graders/
│   └── sql_grader.py   # Correctness & Efficiency Evaluation
└── requirements.txt    # Project Dependencies
```

## 🏆 Hackathon Details
This environment was built for the **Meta PyTorch Hackathon**. It focuses on evaluating the reasoning capabilities of LLMs in structured data environments.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.