"""
inference.py — Baseline inference script for SQL Debugger OpenEnv.
"""
import os
import json
import time
import sys

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    from openai import OpenAI
except ImportError:
    os.system(f"{sys.executable} -m pip install openai -q")
    from openai import OpenAI

# ---------------------------------------------------------------------------
# ✅ Config — validator injects API_BASE_URL and API_KEY
# ✅ ENV_URL defaults to localhost:7860 (validator runs env in Docker locally)
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

print(f"API_BASE_URL = {API_BASE_URL}", flush=True)
print(f"MODEL_NAME   = {MODEL_NAME}", flush=True)
print(f"ENV_URL      = {ENV_URL}", flush=True)

# ---------------------------------------------------------------------------
# ✅ OpenAI client — strictly uses validator-injected API_BASE_URL and API_KEY
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    timeout=60.0,
)

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task_id: str, sql: str) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"task_id": task_id, "sql": sql}, timeout=30)
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
# Run a single task
# ---------------------------------------------------------------------------
def run_task(task_id: str) -> dict:
    print(f"[START] task={task_id}", flush=True)

    best_score = 0.0
    step = 0

    try:
        observation = env_reset(task_id)
        max_steps = observation.get("max_steps", 10)
        done = False

        while not done and step < max_steps:
            sql = observation.get("current_query", "SELECT 1")

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(observation)},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                sql = completion.choices[0].message.content or sql
                print(f"INFO: LLM responded at step {step+1}", flush=True)
            except Exception as exc:
                print(f"WARNING: LLM call failed at step {step+1}: {exc}", flush=True)

            sql = sql.replace("```sql", "").replace("```", "").strip()

            try:
                result = env_step(task_id, sql)
                observation = result["observation"]
                reward = result["reward"]["value"]
                done = result["done"]
                score = observation.get("score", 0.0)
                best_score = max(best_score, score)
                step += 1
                print(f"[STEP] step={step} reward={reward:.4f}", flush=True)
            except Exception as e:
                print(f"WARNING: env_step failed: {e}", flush=True)
                step += 1
                print(f"[STEP] step={step} reward=0.0000", flush=True)
                break

            time.sleep(0.3)

    except Exception as e:
        print(f"WARNING: Task failed: {e}", flush=True)
        step = max(step, 1)
        print(f"[STEP] step={step} reward=0.0000", flush=True)

    print(f"[END] task={task_id} score={best_score:.4f} steps={step}", flush=True)
    return {"task_id": task_id, "best_score": round(best_score, 4), "steps": step}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("SQL Debugger — OpenEnv Baseline Inference", flush=True)
    print(f"Model:    {MODEL_NAME}", flush=True)
    print(f"Base URL: {API_BASE_URL}", flush=True)
    print(f"Env URL:  {ENV_URL}", flush=True)

    results = {}
    for task_id in TASK_IDS:
        results[task_id] = run_task(task_id)

    avg_score = sum(r["best_score"] for r in results.values()) / len(results)

    print(f"\n{'='*50}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*50}", flush=True)
    for tid, r in results.items():
        print(f"  {tid:<15} best_score={r['best_score']:.3f}  steps={r['steps']}", flush=True)
    print(f"  {'AVERAGE':<15} {avg_score:.3f}", flush=True)

    output = {
        "model":     MODEL_NAME,
        "api_base":  API_BASE_URL,
        "avg_score": round(avg_score, 4),
        "results":   results,
    }

    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
