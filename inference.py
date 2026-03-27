"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
"""
import os
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_IDS     = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------
def env_reset(task_id):
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task_id, sql):
    r = requests.post(f"{ENV_URL}/step", json={"task_id": task_id, "sql": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
def build_user_prompt(observation):
    exec_res = observation.get("execution_result", {})
    rows = exec_res.get("rows", [])[:5]
    return f"""DATABASE SCHEMA:
{observation['db_schema']}

BUSINESS REQUIREMENT:
{observation['business_requirement']}

CURRENT QUERY:
{observation['current_query']}

EXECUTION RESULT:
Success: {exec_res.get('success')}
Error: {exec_res.get('error') or 'None'}
Sample rows: {json.dumps(rows, indent=2)}

GRADER FEEDBACK:
{observation['feedback']}

Write the corrected SQL query:"""

# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(task_id):
    print(f"\n{'='*50}")
    print(f"Task: {task_id}")
    print(f"{'='*50}")

    observation = env_reset(task_id)
    max_steps = observation["max_steps"]
    done = False
    step = 0
    best_score = 0.0

    while not done and step < max_steps:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(observation)},
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
            print(f"Model request failed: {exc}. Using fallback.")
            sql = observation["current_query"]

        # Strip markdown fences if present
        sql = sql.replace("```sql", "").replace("```", "").strip()

        print(f"Step {step+1}: submitting query...")
        result = env_step(task_id, sql)

        observation = result["observation"]
        reward      = result["reward"]["value"]
        done        = result["done"]
        score       = observation["score"]
        best_score  = max(best_score, score)

        print(f"  Score: {score:.3f} | Reward: {reward:.4f} | Done: {done}")
        print(f"  Feedback: {observation['feedback'][:120]}")

        step += 1
        time.sleep(0.5)

    print(f"\nTask {task_id} finished — best score: {best_score:.3f}")
    return {"task_id": task_id, "best_score": round(best_score, 4), "steps": step}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SQL Debugger — OpenEnv Baseline Inference")
    print(f"Model: {MODEL_NAME} | Base URL: {API_BASE_URL}")

    results = {}
    for task_id in TASK_IDS:
        results[task_id] = run_task(task_id)

    avg = sum(r["best_score"] for r in results.values()) / len(results)

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for tid, r in results.items():
        print(f"  {tid:<15} best_score={r['best_score']:.3f}  steps={r['steps']}")
    print(f"  {'AVERAGE':<15} {avg:.3f}")

    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": results, "avg": round(avg,4)}, f, indent=2)
    print("\nSaved to baseline_results.json")

if __name__ == "__main__":
    main()

