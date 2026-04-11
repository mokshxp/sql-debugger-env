"""
inference.py — SQL Debugger OpenEnv baseline inference script.
Strictly follows the sample inference.py format from the hackathon.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables — exactly as required by the checklist
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN         = os.getenv("HF_TOKEN")          # No default — validator injects this
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional

ENV_URL     = os.getenv("ENV_URL", "http://localhost:7860")
TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS   = 10
TEMPERATURE = 0.2
MAX_TOKENS  = 512
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# Structured logging — strictly following START/STEP/END format
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type":  "START",
        "task":  task,
        "env":   env,
        "model": model,
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    print(json.dumps({
        "type":   "STEP",
        "step":   step,
        "action": action[:300],
        "reward": reward,
        "done":   done,
        "error":  str(error) if error else None,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "type":    "END",
        "success": success,
        "steps":   steps,
        "score":   score,
        "rewards": rewards,
    }), flush=True)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
def get_model_message(client: OpenAI, obs: dict, history: List[str]) -> str:
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

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        sql = completion.choices[0].message.content or ""
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return obs.get("current_query", "SELECT 1")


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


def wait_for_env(max_wait: int = 120) -> None:
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
    raise RuntimeError(f"Env not reachable at {ENV_URL} after {max_wait}s")


# ---------------------------------------------------------------------------
# Main — async to match sample inference.py pattern
# ---------------------------------------------------------------------------
async def main() -> None:
    # OpenAI client — using API_BASE_URL and HF_TOKEN as required
    api_key = HF_TOKEN or "dummy-key"
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    # Wait for environment container to be ready
    wait_for_env(max_wait=120)

    all_results = []

    for task_id in TASK_IDS:
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_id, env="sql-debugger", model=MODEL_NAME)

        try:
            obs = env_reset(task_id)
            last_reward = 0.0
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action = get_model_message(client, obs, history)

                result  = env_step(task_id, action)
                obs     = result["observation"]
                reward  = result["reward"]["value"] if result.get("reward") else 0.0
                done    = result.get("done", False)
                error   = None

                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                score = obs.get("score", 0.0)

                log_step(step=step, action=action, reward=reward, done=done, error=error)
                history.append(f"Step {step}: reward {reward:+.2f}")

                if done:
                    break

            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        all_results.append({
            "task_id": task_id,
            "score":   round(score, 4),
            "steps":   steps_taken,
            "success": success,
        })

    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"[DEBUG] Average score: {avg:.3f}", flush=True)

    with open("baseline_results.json", "w") as f:
        json.dump({
            "model":     MODEL_NAME,
            "avg_score": round(avg, 4),
            "results":   all_results,
        }, f, indent=2)
    print("[DEBUG] Saved baseline_results.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
