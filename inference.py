"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
Strictly follows Phase 2 deep validation log format requirements.
"""
import os
import json
import time
import sys
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — validator injects these variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_IDS     = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE  = 0.2
MAX_TOKENS   = 512
MAX_STEPS    = 10

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# Structured logging (required by validator)
# ---------------------------------------------------------------------------
def log_start(task: str, model: str) -> None:
    print(json.dumps({
        "type": "START",
        "task": task,
        "model": model,
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action[:200],
        "reward": reward,
        "done": done,
        "error": str(error) if error else None,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
    }), flush=True)


# ---------------------------------------------------------------------------
# OpenAI client — strict global init, fails loudly if broken
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

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
{observation.get('db_schema', '')}

BUSINESS REQUIREMENT:
{observation.get('business_requirement', '')}

CURRENT QUERY:
{observation.get('current_query', '')}

EXECUTION RESULT:
Success: {exec_res.get('success')}
Error: {exec_res.get('error') or 'None'}
Sample rows: {json.dumps(rows)}

GRADER FEEDBACK:
{observation.get('feedback', '')}

Write the corrected SQL query:"""


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> dict:
    log_start(task=task_id, model=MODEL_NAME)

    observation  = env_reset(task_id)
    max_steps    = observation.get("max_steps", MAX_STEPS)
    done         = False
    step         = 0
    best_score   = 0.0
    rewards      = []
    success      = False

    try:
        while not done and step < max_steps:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(observation)},
            ]

            # LLM call — exceptions propagate loudly
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            sql = completion.choices[0].message.content or ""
            sql = sql.replace("```sql", "").replace("```", "").strip()

            result      = env_step(task_id, sql)
            observation = result["observation"]
            reward      = result["reward"]["value"]
            done        = result["done"]
            score       = observation.get("score", 0.0)
            best_score  = max(best_score, score)
            rewards.append(reward)

            log_step(step=step+1, action=sql, reward=reward, done=done)

            step += 1
            time.sleep(0.3)

        success = best_score >= 0.6

    except Exception as e:
        log_end(success=False, steps=step, score=best_score, rewards=rewards)
        raise e

    log_end(success=success, steps=step, score=best_score, rewards=rewards)

    return {
        "task_id":    task_id,
        "best_score": round(best_score, 4),
        "steps":      step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"SQL Debugger — OpenEnv Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME} | Base URL: {API_BASE_URL}", flush=True)

    results = {}
    for task_id in TASK_IDS:
        results[task_id] = run_task(task_id)

    avg_score = sum(r["best_score"] for r in results.values()) / len(results)

    print(f"\nFINAL RESULTS", flush=True)
    for tid, r in results.items():
        print(f"  {tid}: best_score={r['best_score']:.3f} steps={r['steps']}", flush=True)
    print(f"  AVERAGE: {avg_score:.3f}", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model":     MODEL_NAME,
            "avg_score": round(avg_score, 4),
            "results":   results,
        }, f, indent=2)
    print("Saved to baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
