"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
Updated to use the official LLM Proxy credentials (API_KEY and API_BASE_URL).
"""
import os
import json
import time
import sys

# Silence auto-installers
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
# Config: EXACT names from the Meta/Scaler Validator instructions
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY      = os.environ.get("API_KEY") # This is the key they track
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASK_IDS    = ["task_easy", "task_medium", "task_hard"]
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = """You are an expert SQL engineer. Debug and fix the given SQL query.
Output ONLY the corrected SQL query — no explanations, no markdown, no backticks."""

# ---------------------------------------------------------------------------
# OpenAI client initialization - Using the Proxy
# ---------------------------------------------------------------------------
def get_client():
    if not API_KEY or not API_BASE_URL:
        # Using stderr for warnings so we don't pollute the structured logs
        print(f"DEBUG ERROR: Missing API_KEY or API_BASE_URL", file=sys.stderr)
        return None
    try:
        return OpenAI(
            api_key=API_KEY,      # Required by LiteLLM Proxy
            base_url=API_BASE_URL, # Required by LiteLLM Proxy
            timeout=30.0,
        )
    except Exception as e:
        print(f"DEBUG ERROR: Client init failed: {e}", file=sys.stderr)
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
    # MANDATORY START TAG
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

            # Use existing query as default if LLM fails
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
                except Exception as e:
                    print(f"LLM Call Failed: {e}", file=sys.stderr)

            # Clean output
            sql = sql.replace("```sql", "").replace("```", "").strip()

            result = env_step(task_id, sql)
            observation = result["observation"]
            reward = result["reward"]["value"]
            score = observation.get("score", 0.0)
            best_score = max(best_score, score)

            # MANDATORY STEP TAG
            print(f"[STEP] step={step+1} reward={reward:.4f}", flush=True)

            if result["done"]:
                break
            step += 1
            time.sleep(0.1)

        # MANDATORY END TAG
        print(f"[END] task={task_id} score={best_score:.4f} steps={step+1}", flush=True)

    except Exception as e:
        print(f"[END] task={task_id} score=0.0000 steps=0", flush=True)
        print(f"Task Error: {e}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for task_id in TASK_IDS:
        run_task(task_id)