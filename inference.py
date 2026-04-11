#!/usr/bin/env python3
"""
inference.py — SQL Debugger OpenEnv baseline inference script.
"""
from __future__ import annotations
import os
import sys
import json
import time

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
# Configuration is now handled inside main() to prevent top-level crashes
API_BASE_URL     = os.environ.get("API_BASE_URL")
API_KEY          = os.environ.get("API_KEY")
MODEL_NAME       = os.environ.get("MODEL_NAME", "gpt-4o-mini")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
ENV_URL          = os.getenv("ENV_URL", "http://localhost:7860")

TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS   = 10
TEMPERATURE = 0.2
MAX_TOKENS  = 512
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# Structured logging — plain text format as required
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Any = None) -> None:
    # Use !r for action to handle multi-line SQL strings correctly in logs if needed, 
    # but the sample implies field=value format.
    # The requirement says 'strictly following the format'.
    # Sample call: log_step(step=step, action=message, reward=reward, done=done, error=error)
    print(f"[STEP] step={step} action={action!r} reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)

# ---------------------------------------------------------------------------
# Wait for env
# ---------------------------------------------------------------------------
def wait_for_env(max_wait: int = 60) -> None:
    print(f"[DEBUG] Waiting for env at {ENV_URL}...", flush=True)
    for i in range(max_wait):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"[DEBUG] Env ready after {i}s", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    print(f"[DEBUG] Env not ready after {max_wait}s — proceeding anyway", flush=True)

# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=60)
    r.raise_for_status()
    return r.json()

def env_step(task_id: str, sql: str) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"task_id": task_id, "sql": sql}, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def get_model_message(client: OpenAI, obs: dict) -> str:
    exec_res = obs.get("execution_result", {})
    rows = exec_res.get("rows", [])[:5]
    user_prompt = f"""DATABASE SCHEMA:
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

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    sql = completion.choices[0].message.content or ""
    return sql.replace("```sql", "").replace("```", "").strip()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("--- SCRIPT INVOKED ---", file=sys.stderr, flush=True)
    print("[DEBUG] Starting inference script...", flush=True)

    try:
        import requests
        from openai import OpenAI
        from typing import List, Any
    except ImportError as e:
        print(f"[ERROR] Dependency missing: {e}", flush=True)
        sys.exit(1)

    # Use globals but handle them locally for safety
    global API_BASE_URL, API_KEY, MODEL_NAME
    
    API_BASE_URL = os.environ.get("API_BASE_URL")
    API_KEY      = os.environ.get("API_KEY")
    MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not API_BASE_URL or not API_KEY:
        print(f"[ERROR] Required environment variables are missing!", flush=True)
        print(f"[ERROR] API_BASE_URL: {'Set' if API_BASE_URL else 'MISSING'}", flush=True)
        print(f"[ERROR] API_KEY: {'Set' if API_KEY else 'MISSING'}", flush=True)
        sys.exit(1)

    print(f"[DEBUG] Config: model={MODEL_NAME} env={ENV_URL}", flush=True)

    # Strict initialization with robust URL handling
    try:
        base_url = API_BASE_URL.strip()
        if base_url and not base_url.startswith("http"):
            base_url = f"http://{base_url}"
            
        client = OpenAI(base_url=base_url, api_key=API_KEY)
        print(f"[DEBUG] OpenAI client initialized with base_url={base_url}", flush=True)
    except Exception as e:
        print(f"[ERROR] Fatal error initializing OpenAI client: {e}", flush=True)
        sys.exit(1)

    wait_for_env(max_wait=60)

    all_results = []

    for task_id in TASK_IDS:
        rewards: List[float] = []
        steps_taken = 0
        score   = 0.0
        success = False

        log_start(task=task_id, env="sql-debugger", model=MODEL_NAME)

        try:
            obs  = env_reset(task_id)
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                # Force model usage
                action = get_model_message(client, obs)

                result = env_step(task_id, action)
                obs    = result["observation"]
                reward = result["reward"]["value"] if result.get("reward") else 0.0
                done   = result.get("done", False)
                score  = obs.get("score", 0.0)

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action, reward=reward, done=done)

                if done:
                    break

            score   = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        all_results.append({
            "task_id": task_id,
            "score":   round(score, 4),
            "steps":   steps_taken,
        })

    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"[DEBUG] Average score: {avg:.3f}", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model":     MODEL_NAME,
            "avg_score": round(avg, 4),
            "results":   all_results,
        }, f, indent=2)
    print("[DEBUG] Done", flush=True)


if __name__ == "__main__":
    main()
