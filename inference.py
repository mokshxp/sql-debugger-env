"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
"""
import os
import json
import time
import sys

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    os.system(f"{sys.executable} -m pip install openai -q")
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
# OpenAI client — wrapped in try/except
# ---------------------------------------------------------------------------
client = None

def get_client():
    global client
    if client is not None:
        return client
    try:
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set. Using dummy token for testing.")
            token = "dummy-token"
        else:
            token = HF_TOKEN
        client = OpenAI(
            api_key=token,
            base_url=API_BASE_URL,
            timeout=30.0,
        )
        return client
    except Exception as e:
        print(f"WARNING: Could not initialize OpenAI client: {e}")
        return None

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    try:
        r = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"ERROR: Could not reset environment: {e}")
        raise


def env_step(task_id: str, sql: str) -> dict:
    try:
        r = requests.post(
            f"{ENV_URL}/step",
            json={"task_id": task_id, "sql": sql},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"ERROR: Could not step environment: {e}")
        raise


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

CURRENT QUERY (step {observation.get('step', 0)}/{observation.get('max_steps', 10)}):
{observation.get('current_query', '')}

EXECUTION RESULT:
Success: {exec_res.get('success')}
Error: {exec_res.get('error') or 'None'}
Row count: {exec_res.get('row_count', 0)}
Sample rows:
{json.dumps(rows, indent=2)}

GRADER FEEDBACK:
{observation.get('feedback', '')}

Current score: {observation.get('score', 0.0):.3f}

Write the corrected SQL query:"""


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Task: {task_id}")
    print(f"{'='*50}")

    observation = env_reset(task_id)
    max_steps   = observation.get("max_steps", 10)
    done        = False
    step        = 0
    best_score  = 0.0

    while not done and step < max_steps:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(observation)},
        ]

        sql = observation.get("current_query", "SELECT 1")

        try:
            c = get_client()
            if c is not None:
                completion = c.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                sql = completion.choices[0].message.content or sql
            else:
                print(f"  Step {step+1}: No client available, using current query as fallback")
        except Exception as exc:
            print(f"  Step {step+1}: Model request failed ({exc}). Using fallback.")

        # Strip markdown fences
        sql = sql.replace("```sql", "").replace("```", "").strip()

        try:
            result      = env_step(task_id, sql)
            observation = result["observation"]
            reward      = result["reward"]["value"]
            done        = result["done"]
            score       = observation.get("score", 0.0)
            best_score  = max(best_score, score)

            print(f"  Step {step+1}: score={score:.3f} reward={reward:.4f} done={done}")
        except Exception as e:
            print(f"  Step {step+1}: Environment error: {e}")
            break

        step += 1
        time.sleep(0.3)

    print(f"\nTask {task_id} done — best_score={best_score:.3f} steps={step}")
    return {
        "task_id":    task_id,
        "best_score": round(best_score, 4),
        "steps":      step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SQL Debugger — OpenEnv Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"Base URL: {API_BASE_URL}")
    print(f"Env URL:  {ENV_URL}")

    results = {}
    for task_id in TASK_IDS:
        try:
            results[task_id] = run_task(task_id)
        except Exception as e:
            print(f"ERROR on task {task_id}: {e}")
            results[task_id] = {"task_id": task_id, "best_score": 0.0, "steps": 0}

    avg_score = sum(r["best_score"] for r in results.values()) / len(results)

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    for tid, r in results.items():
        print(f"  {tid:<15} best_score={r['best_score']:.3f}  steps={r['steps']}")
    print(f"  {'AVERAGE':<15} {avg_score:.3f}")

    output = {
        "model":     MODEL_NAME,
        "api_base":  API_BASE_URL,
        "avg_score": round(avg_score, 4),
        "results":   results,
    }

    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to baseline_results.json")
    return output


if __name__ == "__main__":
    main()
