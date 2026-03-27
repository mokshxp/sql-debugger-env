"""
FastAPI application for the SQL Debugger OpenEnv environment.

Endpoints:
  GET  /            — HTML playground UI
  GET  /health      — liveness probe
  POST /reset       — reset environment, returns Observation
  POST /step        — step with an Action, returns (Observation, Reward, done, info)
  GET  /state       — current environment state
  GET  /tasks       — list available tasks
  POST /validate    — openenv validate endpoint
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from env.environment import SQLDebuggerEnv
from env.models import Action, Observation, Reward
from tasks.tasks import TASKS

app = FastAPI(
    title="SQL Debugger — OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents debug and optimize SQL queries "
        "against a live e-commerce database."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global env registry (one instance per task for the demo; production would
# use session-based instances)
# ---------------------------------------------------------------------------
_envs: Dict[str, SQLDebuggerEnv] = {}


def _get_env(task_id: str) -> SQLDebuggerEnv:
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Use: {list(TASKS)}")
    if task_id not in _envs:
        _envs[task_id] = SQLDebuggerEnv(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class StepRequest(BaseModel):
    task_id: str = "task_easy"
    sql: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "environment": "sql-debugger", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "business_requirement": t.business_requirement,
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset", response_model=Dict[str, Any])
def reset(req: ResetRequest):
    env = _get_env(req.task_id)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id)
    action = Action(sql=req.sql)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=info,
    )


@app.get("/state")
def state(task_id: str = Query("task_easy")):
    env = _get_env(task_id)
    return env.state()


@app.get("/validate")
def validate():
    """openenv validate gate."""
    results = {}
    for task_id in TASKS:
        env = SQLDebuggerEnv(task_id=task_id)
        obs = env.reset()
        # Sanity-check the observation model
        assert obs.task_id == task_id
        assert obs.step == 0
        action = Action(sql="SELECT 1")
        obs2, rew, done, info = env.step(action)
        assert isinstance(rew.value, float)
        state = env.state()
        assert "task_id" in state
        results[task_id] = "PASS"
    return {"validation": results, "status": "ALL_PASS"}


# ---------------------------------------------------------------------------
# Playground HTML UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def playground():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SQL Debugger — OpenEnv</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d2e; --border: #2a2d3e;
    --green: #00d68f; --red: #ff6b6b; --yellow: #ffd166;
    --blue: #4dabf7; --text: #e8eaf6; --muted: #888;
    --code-bg: #12151f;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.4rem; font-weight: 700; color: var(--green); }
  header .badge { background: var(--border); color: var(--muted); font-size: 0.75rem; padding: 2px 8px; border-radius: 99px; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: 0; height: calc(100vh - 60px); }
  .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1.5rem; }
  .main { display: flex; flex-direction: column; gap: 0; overflow: hidden; }
  .task-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; cursor: pointer; transition: border-color 0.2s; }
  .task-card:hover, .task-card.active { border-color: var(--green); }
  .task-card .difficulty { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
  .easy { color: var(--green); } .medium { color: var(--yellow); } .hard { color: var(--red); }
  .task-card h3 { font-size: 0.9rem; margin: 0.4rem 0; }
  .task-card p { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }
  .schema-box { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; font-family: monospace; font-size: 0.72rem; color: #b0b8d8; line-height: 1.6; white-space: pre; overflow-x: auto; }
  .editor-area { flex: 1; display: flex; flex-direction: column; }
  .req-bar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 1.5rem; }
  .req-bar label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); display: block; margin-bottom: 0.4rem; }
  .req-text { font-size: 0.85rem; line-height: 1.6; color: var(--blue); }
  .editor-row { flex: 1; display: grid; grid-template-columns: 1fr 1fr; overflow: hidden; }
  .pane { display: flex; flex-direction: column; border-right: 1px solid var(--border); overflow: hidden; }
  .pane:last-child { border-right: none; }
  .pane-header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 0.6rem 1rem; font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; display: flex; justify-content: space-between; align-items: center; }
  textarea { flex: 1; background: var(--code-bg); color: #c8d6ff; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.82rem; padding: 1rem; border: none; resize: none; outline: none; line-height: 1.7; }
  .results-pane { flex: 1; overflow-y: auto; padding: 1rem; background: var(--code-bg); font-size: 0.8rem; }
  .bottom-bar { background: var(--surface); border-top: 1px solid var(--border); padding: 0.75rem 1.5rem; display: flex; align-items: center; gap: 1rem; }
  button { background: var(--green); color: #000; font-weight: 700; border: none; border-radius: 6px; padding: 0.5rem 1.5rem; cursor: pointer; font-size: 0.85rem; transition: opacity 0.2s; }
  button:hover { opacity: 0.85; }
  button.reset-btn { background: var(--border); color: var(--text); }
  .score-bar { display: flex; align-items: center; gap: 0.8rem; margin-left: auto; }
  .score-label { font-size: 0.75rem; color: var(--muted); }
  .score-val { font-size: 1.1rem; font-weight: 700; color: var(--green); }
  .steps-val { font-size: 0.8rem; color: var(--muted); }
  .feedback { background: #1a2040; border: 1px solid #2a3560; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; font-size: 0.8rem; line-height: 1.6; color: #c8d6ff; }
  .result-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .result-table th { background: var(--surface); padding: 0.4rem 0.8rem; text-align: left; font-weight: 600; color: var(--muted); border-bottom: 1px solid var(--border); }
  .result-table td { padding: 0.4rem 0.8rem; border-bottom: 1px solid #1a1d2e; }
  .error-msg { color: var(--red); font-family: monospace; font-size: 0.8rem; padding: 0.5rem; background: #2a1010; border-radius: 4px; }
  .done-banner { background: linear-gradient(135deg, #00d68f22, #00d68f44); border: 1px solid var(--green); border-radius: 8px; padding: 1rem; text-align: center; color: var(--green); font-weight: 700; }
</style>
</head>
<body>
<header>
  <h1>🔍 SQL Debugger</h1>
  <span class="badge">OpenEnv</span>
  <span class="badge">Real-world</span>
  <span class="badge" style="color:var(--green)">E-Commerce DB</span>
</header>
<div class="layout">
  <div class="sidebar">
    <div>
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:0.8rem">Tasks</div>
      <div id="task-list" style="display:flex;flex-direction:column;gap:0.6rem"></div>
    </div>
    <div>
      <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:0.6rem">Database Schema</div>
      <div class="schema-box" id="schema-box">Loading...</div>
    </div>
  </div>
  <div class="main">
    <div class="req-bar">
      <label>Business Requirement</label>
      <div class="req-text" id="req-text">Select a task to begin.</div>
    </div>
    <div class="editor-row">
      <div class="pane">
        <div class="pane-header"><span>Your SQL Query</span><span id="step-badge">Step 0</span></div>
        <textarea id="sql-editor" spellcheck="false" placeholder="Write your SQL query here..."></textarea>
      </div>
      <div class="pane">
        <div class="pane-header"><span>Results &amp; Feedback</span><span id="score-badge" style="color:var(--green)">Score: —</span></div>
        <div class="results-pane" id="results-pane"><p style="color:var(--muted)">Submit a query to see results.</p></div>
      </div>
    </div>
    <div class="bottom-bar">
      <button onclick="submitQuery()">▶ Run &amp; Grade</button>
      <button class="reset-btn" onclick="resetEnv()">↺ Reset</button>
      <div class="score-bar">
        <span class="score-label">Best Score</span>
        <span class="score-val" id="score-display">—</span>
        <span class="steps-val" id="steps-display"></span>
      </div>
    </div>
  </div>
</div>
<script>
let currentTask = 'task_easy';
let tasks = [];

async function loadTasks() {
  const r = await fetch('/tasks');
  const d = await r.json();
  tasks = d.tasks;
  const el = document.getElementById('task-list');
  el.innerHTML = '';
  tasks.forEach(t => {
    const card = document.createElement('div');
    card.className = 'task-card' + (t.id === currentTask ? ' active' : '');
    card.innerHTML = `<div class="difficulty ${t.difficulty}">${t.difficulty}</div>
      <h3>${t.name}</h3>
      <p>${t.business_requirement.substring(0,100)}...</p>`;
    card.onclick = () => selectTask(t.id);
    el.appendChild(card);
  });
}

async function selectTask(taskId) {
  currentTask = taskId;
  document.querySelectorAll('.task-card').forEach((c,i) => {
    c.classList.toggle('active', tasks[i].id === taskId);
  });
  const r = await fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId})});
  const obs = await r.json();
  updateUI(obs, null, null);
}

function updateUI(obs, reward, info) {
  const t = tasks.find(x => x.id === currentTask);
  document.getElementById('req-text').textContent = obs.business_requirement || (t && t.business_requirement) || '';
  document.getElementById('sql-editor').value = obs.current_query || '';
  document.getElementById('schema-box').textContent = obs.db_schema || '';
  document.getElementById('step-badge').textContent = `Step ${obs.step}/${obs.max_steps}`;
  document.getElementById('score-badge').textContent = `Score: ${(obs.score*100).toFixed(1)}%`;
  document.getElementById('score-display').textContent = (obs.score*100).toFixed(1) + '%';
  document.getElementById('steps-display').textContent = `${obs.step}/${obs.max_steps} steps`;

  const rp = document.getElementById('results-pane');
  let html = '';

  if (obs.feedback && obs.step > 0) {
    html += `<div class="feedback">${obs.feedback.split('|').map(s => s.trim()).map(s => s.startsWith('✓') ? `<span style="color:var(--green)">${s}</span>` : `<span style="color:var(--yellow)">${s}</span>`).join('<br>')}</div>`;
  }

  if (reward) {
    html += `<div style="display:flex;gap:1rem;margin-bottom:0.8rem;font-size:0.78rem">
      <span>Reward: <b style="color:var(--green)">${reward.value.toFixed(3)}</b></span>
      <span>Correctness: <b>${(reward.correctness*100).toFixed(1)}%</b></span>
      <span>Efficiency: <b>${(reward.efficiency*100).toFixed(1)}%</b></span>
    </div>`;
  }

  const exec = obs.execution_result;
  if (exec) {
    if (!exec.success) {
      html += `<div class="error-msg">❌ ${exec.error}</div>`;
    } else if (exec.rows && exec.rows.length > 0) {
      const cols = Object.keys(exec.rows[0]);
      html += `<p style="color:var(--muted);font-size:0.72rem;margin-bottom:0.5rem">${exec.row_count} row(s)</p>`;
      html += `<table class="result-table"><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
      exec.rows.slice(0,50).forEach(row => {
        html += `<tr>${cols.map(c=>`<td>${row[c] ?? 'NULL'}</td>`).join('')}</tr>`;
      });
      html += '</tbody></table>';
    } else {
      html += `<p style="color:var(--muted)">Query ran successfully — no rows returned.</p>`;
    }
  }

  if (reward && reward.done && obs.score >= 0.95) {
    html += `<div class="done-banner" style="margin-top:1rem">🎉 Task Complete! Final Score: ${(obs.score*100).toFixed(1)}%</div>`;
  }

  rp.innerHTML = html || '<p style="color:var(--muted)">Submit a query to see results.</p>';
}

async function submitQuery() {
  const sql = document.getElementById('sql-editor').value.trim();
  if (!sql) return;
  const r = await fetch('/step', {method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task_id:currentTask, sql})});
  const d = await r.json();
  updateUI(d.observation, d.reward, d.info);
}

async function resetEnv() {
  await selectTask(currentTask);
}

// Ctrl+Enter to submit
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') submitQuery();
});

(async () => {
  await loadTasks();
  await selectTask('task_easy');
})();
</script>
</body>
</html>"""
