"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
Optimized for strict OpenEnv Phase 2 Validation.
"""
import os
import json
import time
import sys

# Silence auto-installers to prevent polluting stdout
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {package} -q")

install_package("requests")
install_package("openai")

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# OpenAI client initialization
# ---------------------------------------------------------------------------
def get_client():
    token = HF_TOKEN if HF_TOKEN else "dummy-token"
    try:
        return OpenAI(
            api_key=token,
            base_url=API_BASE_URL,
            timeout=30.0,
        )
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Environment Helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task_id: str, sql: str) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"task_id": task_id, "sql": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

def build_user_prompt(observation: dict) -> str:
    exec_res = observation.get("execution_result", {})
    rows = exec_res.get("rows", [])[:5]
    return f"""DATABASE SCHEMA:\n{observation.get('db_schema', '')}\n
BUSINESS REQUIREMENT:\n{observation.get('business_requirement', '')}\n
CURRENT QUERY:\n{observation.get('current_query', '')}\n
EXECUTION RESULT:\nSuccess: {exec_res.get('success')}\nError: {exec_res.get('error') or 'None'}\n
Sample rows:\n{json.dumps(rows, indent=2)}\n
GRADER FEEDBACK:\n{observation.get('feedback', '')}\n
Write the corrected SQL query:"""

# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------
def run_task(task_id: str):
    # ✅ MANDATORY: No text should ideally precede this for regex parsers
    print(f"[START] task={task_id}", flush=True)

    try:
        observation = env_reset(task_id)
        max_steps = observation.get("max_steps", 10)
        step = 0
        best_score = 0.0

        client = get_client()

        while step < max_steps:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation)},
            ]

            sql = observation.get("current_query", "SELECT 1")

            if client:
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    sql = completion.choices[0].message.content or sql
                except Exception:
                    pass # Fallback to current_query

            # Clean output
            sql = sql.replace("```sql", "").replace("```", "").strip()

            result = env_step(task_id, sql)
            observation = result["observation"]
            reward = result["reward"]["value"]
            score = observation.get("score", 0.0)
            best_score = max(best_score, score)

            # ✅ MANDATORY: [STEP] block
            print(f"[STEP] step={step+1} reward={reward:.4f}", flush=True)

            if result["done"]:
                break
            step += 1
            time.sleep(0.1)

        # ✅ MANDATORY: [END] block
        print(f"[END] task={task_id} score={best_score:.4f} steps={step+1}", flush=True)

    except Exception as e:
        # Emergency exit tag so validator doesn't hang
        print(f"[END] task={task_id} score=0.0000 steps=0", flush=True)
        print(f"Internal Error: {e}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Removed all intro prints to keep stdout clean for the validator
    for task_id in TASK_IDS:
        run_task(task_id)