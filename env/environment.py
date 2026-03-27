"""
SQLDebuggerEnv — OpenEnv-compliant environment for SQL debugging and optimization.

API surface:
  env = SQLDebuggerEnv(task_id="task_easy")
  obs            = env.reset()
  obs, rew, done, info = env.step(action)
  state          = env.state()
"""
from __future__ import annotations

import sqlite3
import json
from typing import Any, Dict, List, Optional, Tuple

from env.database import get_db, DDL, SCHEMA_TEXT
from env.models import Action, ExecutionResult, Observation, Reward
from tasks.tasks import TASKS, Task

# Step penalty keeps the reward signal dense and discourages thrashing.
STEP_PENALTY = -0.01


class SQLDebuggerEnv:
    """
    OpenEnv environment that challenges an agent to debug and optimize SQL queries
    against a realistic e-commerce SQLite database.
    """

    def __init__(self, task_id: str = "task_easy", seed: int = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS)}")
        self.task_id = task_id
        self.seed = seed
        self._task: Task = TASKS[task_id]
        self._db: Optional[sqlite3.Connection] = None
        self._reset_internal()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_internal(self) -> None:
        """Tear down and rebuild internal state (no DB I/O side-effects)."""
        if self._db is not None:
            self._db.close()
        self._db = get_db(":memory:")
        self._step: int = 0
        self._done: bool = False
        self._best_score: float = 0.0
        self._current_query: str = self._task.buggy_query
        self._last_result: ExecutionResult = self._execute(self._task.buggy_query)
        self._last_feedback: str = "Starting query provided. Identify and fix the bugs."

    def _execute(self, sql: str) -> ExecutionResult:
        """Run a SQL query and return an ExecutionResult."""
        try:
            cur = self._db.execute(sql)
            raw = cur.fetchall()
            rows = [dict(r) for r in raw]
            return ExecutionResult(
                success=True,
                rows=rows,
                error=None,
                row_count=len(rows),
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                rows=[],
                error=str(exc),
                row_count=0,
            )

    def _build_obs(self) -> Observation:
        return Observation(
            task_id=self._task.id,
            db_schema=SCHEMA_TEXT.strip(),
            business_requirement=self._task.business_requirement,
            current_query=self._current_query,
            execution_result=self._last_result,
            feedback=self._last_feedback,
            step=self._step,
            max_steps=self._task.max_steps,
            score=self._best_score,
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._reset_internal()
        return self._build_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Submit a SQL query.

        Returns:
            observation — current env state after executing the action
            reward      — Reward model with value, sub-scores, and done flag
            done        — True if episode is over
            info        — extra diagnostic dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        sql = action.sql.strip()

        # Execute the submitted query
        result = self._execute(sql)
        self._current_query = sql
        self._last_result = result

        # Grade
        score, feedback = self._task.grader(
            result.rows, result.error, sql
        )
        self._last_feedback = feedback
        self._best_score = max(self._best_score, score)

        # Reward shaping
        step_penalty = STEP_PENALTY * self._step  # grows mildly each step
        correctness = score
        efficiency = self._efficiency_score(sql)

        # Combine: weight correctness heavily, efficiency for hard task
        if self._task.difficulty == "hard":
            raw_value = 0.7 * correctness + 0.3 * efficiency
        else:
            raw_value = correctness

        reward_value = max(0.0, min(1.0, raw_value + step_penalty))

        # Episode ends when: score ≥ threshold OR steps exhausted OR perfect score
        done = (
            score >= 1.0
            or self._step >= self._task.max_steps
            or score >= 0.99
        )
        self._done = done

        reward = Reward(
            value=round(reward_value, 4),
            correctness=round(correctness, 4),
            efficiency=round(efficiency, 4),
            step_penalty=round(step_penalty, 4),
            done=done,
        )

        obs = self._build_obs()

        info = {
            "step": self._step,
            "score": score,
            "best_score": self._best_score,
            "done_reason": (
                "perfect" if score >= 0.99
                else "max_steps" if self._step >= self._task.max_steps
                else "in_progress"
            ),
        }

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full current state as a plain dict."""
        return {
            "task_id": self._task.id,
            "task_name": self._task.name,
            "difficulty": self._task.difficulty,
            "step": self._step,
            "max_steps": self._task.max_steps,
            "done": self._done,
            "best_score": self._best_score,
            "current_query": self._current_query,
            "last_execution": self._last_result.model_dump(),
            "last_feedback": self._last_feedback,
        }

    # ------------------------------------------------------------------
    # Efficiency scoring (used in hard task)
    # ------------------------------------------------------------------

    def _efficiency_score(self, sql: str) -> float:
        """
        Proxy for query efficiency: analyze EXPLAIN QUERY PLAN output.
        Penalizes full table scans on large tables; rewards index usage.
        Returns 0.0–1.0.
        """
        try:
            plan_rows = self._db.execute(
                f"EXPLAIN QUERY PLAN {sql}"
            ).fetchall()
            plan_text = " ".join(str(r) for r in plan_rows).lower()

            scans = plan_text.count("scan")
            searches = plan_text.count("search")

            # Cartesian products are very bad
            if "automatic covering index" in plan_text:
                return 0.5
            if scans == 0 and searches > 0:
                return 1.0
            if scans > 0 and searches > 0:
                return max(0.3, 1.0 - 0.15 * scans)
            return max(0.1, 1.0 - 0.2 * scans)
        except Exception:
            return 0.5  # can't plan → neutral
