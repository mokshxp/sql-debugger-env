"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
"""
import os
import json
import time
import sys
import subprocess

# Install compatible versions
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "openai==1.35.3",
    "httpx==0.27.0",
    "requests>=2.32.3",
])

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
def log_start(task: str, model: str) -> None:
    print(json.dumps({"type": "START", "task": task, "model": model}), flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    print(json.dumps({
        "type": "STEP", "step": step,
        "action": action[:200], "reward": reward,
        "done": done, "error": str(error) if error else None,
    }), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    print(json.dumps({
        "type": "END", "success": success,
        "steps": steps, "score": score, "rewards": rewards,
    }), flush=True)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Env client — try HF Space URL if localhost fails
# ---------------------------------------------------------------------------
def get_env_url():
    """Try localhost first, then HF Space URL."""
    urls_to_try = [
        ENV_URL,
        "https://moksh24-sql-debugger-env.hf.space",
    ]
    for url in urls_to_try:
        try:
            r = requests.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                print(f"[DEBUG] Connected to env at {url}", flush=True)
                return url
        except Exception:
            continue
    raise RuntimeError(f"Could not connect to environment at any URL: {urls_to_try}")

def env_reset(base_url, task_id):
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(base_url, task_id, sql):
    r = requests.post(f"{base_url}/step", json={"task_id": task_id, "sql": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_prompt(obs):
    exec_res = obs.get("execution_result", {})
    rows = exec_res.get("rows", [])[:5]
    return f"""DATABASE SCHEMA:
{obs.get('db_schema', '')}

BUSINESS REQUIREMENT:
{obs.get('business_requirement', '')}

CURRENT QUERY:
{obs.get('current_query', '')}

EXECUTION RESULT:
Success: {exec_res.get('success')}
Error: {exec_res.get('error') or 'None'}
Sample rows: {json.dumps(rows)}

GRADER FEEDBACK:
{obs.get('feedback', '')}

Write the corrected SQL query:"""

# ---------------------------------------------------------------------------
# Run task
# ---------------------------------------------------------------------------
def run_task(base_url, task_id):
    log_start(task=task_id, model=MODEL_NAME)
    obs       = env_reset(base_url, task_id)
    max_steps = obs.get("max_steps", 10)
    done      = False
    step      = 0
    best      = 0.0
    rewards   = []

    try:
        while not done and step < max_steps:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(obs)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            sql = completion.choices[0].message.content or ""
            sql = sql.replace("```sql", "").replace("```", "").strip()

            result  = env_step(base_url, task_id, sql)
            obs     = result["observation"]
            reward  = result["reward"]["value"]
            done    = result["done"]
            score   = obs.get("score", 0.0)
            best    = max(best, score)
            rewards.append(reward)

            log_step(step=step+1, action=sql, reward=reward, done=done)
            step += 1
            time.sleep(0.3)

    except Exception as e:
        log_end(success=False, steps=step, score=best, rewards=rewards)
        raise e

    log_end(success=best >= 0.6, steps=step, score=best, rewards=rewards)
    return {"task_id": task_id, "best_score": round(best, 4), "steps": step}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"SQL Debugger — OpenEnv Inference | model={MODEL_NAME}", flush=True)
    
    # Find working env URL
    base_url = get_env_url()
    print(f"Using env URL: {base_url}", flush=True)

    results = {}
    for task_id in TASK_IDS:
        results[task_id] = run_task(base_url, task_id)

    avg = sum(r["best_score"] for r in results.values()) / len(results)
    print(f"\nAVERAGE SCORE: {avg:.3f}", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "avg_score": round(avg, 4),
            "results": results
        }, f, indent=2)
    print("Saved baseline_results.json", flush=True)

if __name__ == "__main__":
    main()
