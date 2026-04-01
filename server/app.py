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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

_envs: Dict[str, SQLDebuggerEnv] = {}


def _get_env(task_id: str) -> SQLDebuggerEnv:
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task_id '{task_id}'. Use: {list(TASKS)}")
    if task_id not in _envs:
        _envs[task_id] = SQLDebuggerEnv(task_id=task_id)
    return _envs[task_id]


class ResetRequest(BaseModel):
    task_id: str = "task_easy"
    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    task_id: str = "task_easy"
    sql: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


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
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
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
    results = {}
    for task_id in TASKS:
        env = SQLDebuggerEnv(task_id=task_id)
        obs = env.reset()
        assert obs.task_id == task_id
        assert obs.step == 0
        action = Action(sql="SELECT 1")
        obs2, rew, done, info = env.step(action)
        assert isinstance(rew.value, float)
        s = env.state()
        assert "task_id" in s
        results[task_id] = "PASS"
    return {"validation": results, "status": "ALL_PASS"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
