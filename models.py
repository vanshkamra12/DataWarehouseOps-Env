"""
DataWarehouseOps-Env: Typed Models (Action, Observation, State)
Compliant with OpenEnv specification using Pydantic v2.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ACTION
# ---------------------------------------------------------------------------

class DataWHAction(BaseModel):
    """
    An action the agent can take inside the DataWarehouse environment.

    Fields
    ------
    sql_command : str
        A valid SQL statement to execute against the sandbox database.
        Examples: SELECT, UPDATE, CREATE VIEW, CREATE INDEX, EXPLAIN QUERY PLAN.
    finalize_task : bool
        When True the episode ends and the grader scores the final state.
        The agent should only set this to True when it believes the task is done.
    reasoning : str (optional)
        The agent's chain-of-thought reasoning. Not used in grading but logged
        for training signal research.
    """
    sql_command: Optional[str] = Field(
        default=None,
        description="SQL statement to execute (SELECT, UPDATE, CREATE VIEW, CREATE INDEX, etc.)"
    )
    finalize_task: bool = Field(
        default=False,
        description="Set True to end the episode and trigger the grader."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning/chain-of-thought (optional, not graded)."
    )


# ---------------------------------------------------------------------------
# OBSERVATION
# ---------------------------------------------------------------------------

class DataWHObservation(BaseModel):
    """
    What the agent sees after each step.

    Fields
    ------
    task_id          : Which task is active (task1 / task2 / task3)
    task_description : Full natural-language description of the task objective
    schema_info      : JSON-serialisable dict describing all tables & columns
    query_result     : Rows returned by the last SQL command (list of dicts)
    rows_affected    : Number of rows affected by the last DML command
    error_message    : SQL error message if the last command failed, else None
    step_reward      : Immediate reward from the last action
    total_reward     : Cumulative reward for this episode so far
    current_step     : How many steps have been taken in this episode
    max_steps        : Maximum allowed steps before the episode is force-ended
    done             : Whether the episode has ended
    episode_id       : Unique identifier for this episode
    info             : Arbitrary extra metadata (hints, warnings, etc.)
    """
    task_id: str = Field(description="Active task identifier.")
    task_description: str = Field(description="Natural language description of what the agent must achieve.")
    schema_info: Dict[str, Any] = Field(description="Database schema: tables, columns, types, row counts.")
    query_result: List[Dict[str, Any]] = Field(default_factory=list, description="Rows returned by last SQL.")
    rows_affected: int = Field(default=0, description="Rows changed by last DML statement.")
    error_message: Optional[str] = Field(default=None, description="SQL error if last command failed.")
    step_reward: float = Field(default=0.0, description="Reward from the last action.")
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode.")
    current_step: int = Field(default=0, description="Steps taken so far.")
    max_steps: int = Field(default=30, description="Episode step limit.")
    done: bool = Field(default=False, description="True when the episode has ended.")
    episode_id: str = Field(description="Unique episode identifier.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra hints or grader feedback.")


# ---------------------------------------------------------------------------
# STATE (episode-level metadata)
# ---------------------------------------------------------------------------

class DataWHState(BaseModel):
    """
    Internal episode state returned by the /state endpoint.
    Includes everything in Observation plus grader-specific internal counters.
    """
    episode_id: str
    task_id: str
    current_step: int
    max_steps: int
    total_reward: float
    done: bool
    sql_commands_executed: int = Field(default=0, description="Total valid SQL commands run.")
    invalid_sql_count: int = Field(default=0, description="Number of invalid SQL attempts.")
    tables_dropped: int = Field(default=0, description="Dangerous DROP TABLE actions taken.")
    task_finalized: bool = Field(default=False, description="Whether the agent chose to finalize.")
    grader_score: Optional[float] = Field(default=None, description="Final grader score (0.0–1.0), set on done.")


# ---------------------------------------------------------------------------
# REWARD INFO (returned in info dict)
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Explains how the step reward was computed (for debugging/training analysis)."""
    exploration_bonus: float = 0.0
    execution_penalty: float = 0.0
    drop_penalty: float = 0.0
    loop_penalty: float = 0.0
    finalization_bonus: float = 0.0
    total: float = 0.0
