"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.

Usage:
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py
"""
import os
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables (required by competition spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = """You are an expert SQL engineer. Your job is to debug and fix SQL queries.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks.
Just raw SQL."""

# ---------------------------------------------------------------------------
# OpenAI client (using API_BASE_URL and HF_TOKEN as required)
# ---------------------------------------------------------------------------
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, sql: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/step",
        json={"task_id": task_id, "sql": sql},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_user_prompt(observation: dict) -> str:
    exec_res = observation.get("execution_result", {})
    rows = exec_res.get("rows", [])[:5]
    return f"""DATABASE SCHEMA:
{observation['db_schema']}

BUSINESS REQUIREMENT:
{observation['business_requirement']}

CURRENT QUERY (step {observation['step']}/{observation['max_steps']}):
{observation['current_query']}

EXECUTION RESULT:
Success: {exec_res.get('success')}
Error: {exec_res.get('error') or 'None'}
Row count: {exec_res.get('row_count', 0)}
Sample rows (first 5):
{json.dumps(rows, indent=2)}

GRADER FEEDBACK:
{observation['feedback']}

Current score: {observation['score']:.3f}

Write the corrected SQL query:"""


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Task: {task_id}")
    print(f"{'='*55}")

    observation = env_reset(task_id)
    max_steps   = observation["max_steps"]
    done        = False
    step        = 0
    best_score  = 0.0
    history     = []

    while not done and step < max_steps:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": build_user_prompt(observation)}]},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            sql = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Model request failed ({exc}). Using fallback action.")
            sql = observation["current_query"]

        # Strip markdown code fences if model adds them
        sql = sql.replace("```sql", "").replace("```", "").strip()

        print(f"\n  Step {step + 1}: submitting query ({len(sql)} chars)")

        result      = env_step(task_id, sql)
        observation = result["observation"]
        reward      = result["reward"]["value"]
        done        = result["done"]
        score       = observation["score"]
        best_score  = max(best_score, score)

        history_line = f"Step {step + 1}: reward {reward:+.4f} | score {score:.3f}"
        history.append(history_line)

        print(f"  Reward: {reward:+.4f} | Score: {score:.3f} | Done: {done}")
        print(f"  Feedback: {observation['feedback'][:150]}")

        step += 1
        time.sleep(0.5)  # rate-limit courtesy

    print(f"\n  Task {task_id} complete — best_score={best_score:.3f}  steps={step}/{max_steps}")

    return {
        "task_id":    task_id,
        "best_score": round(best_score, 4),
        "steps":      step,
        "max_steps":  max_steps,
        "history":    history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  SQL Debugger — OpenEnv Baseline Inference")
    print("=" * 55)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Base URL: {API_BASE_URL}")
    print(f"  Env URL:  {ENV_URL}")

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN (or OPENAI_API_KEY) environment variable not set."
        )

    results = {}
    for task_id in TASK_IDS:
        results[task_id] = run_task(task_id)

    avg_score = sum(r["best_score"] for r in results.values()) / len(results)

    print(f"\n{'='*55}")
    print("  FINAL BASELINE RESULTS")
    print(f"{'='*55}")
    for tid, r in results.items():
        print(f"  {tid:<15}  best_score={r['best_score']:.3f}  steps={r['steps']}/{r['max_steps']}")
    print(f"  {'AVERAGE':<15}  {avg_score:.3f}")

    output = {
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
        "avg_score":  round(avg_score, 4),
        "results":    results,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n  Saved to baseline_results.json")
    return output


if __name__ == "__main__":
    main()