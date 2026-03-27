"""
Typed Pydantic models for the SQL Debugger OpenEnv environment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    success: bool = Field(..., description="True if the query ran without error")
    rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Result rows as list of dicts (column→value)"
    )
    error: Optional[str] = Field(None, description="Database error message, if any")
    row_count: int = Field(0, description="Number of rows returned")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the current task")
    db_schema: str = Field(..., description="DDL CREATE statements for the database")
    business_requirement: str = Field(
        ..., description="Natural-language description of what the query must compute"
    )
    current_query: str = Field(
        ..., description="The SQL query the agent most recently submitted"
    )
    execution_result: ExecutionResult = Field(
        ..., description="Result of executing current_query"
    )
    feedback: str = Field(
        ..., description="Human-readable partial grader feedback for the agent"
    )
    step: int = Field(..., description="Current step number (0-indexed)")
    max_steps: int = Field(..., description="Maximum steps allowed this episode")
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Best score achieved so far this episode"
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    sql: str = Field(
        ...,
        description="The SQL query to submit. Agent overwrites this each step."
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward for this step")
    correctness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of grader correctness checks passed"
    )
    efficiency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Efficiency score (1.0 if task has no efficiency component)"
    )
    step_penalty: float = Field(
        ..., le=0.0,
        description="Small negative penalty for each step taken"
    )
    done: bool = Field(..., description="True when episode ends")
