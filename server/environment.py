"""
DataWarehouseOps Environment — Core Logic
Manages per-session SQLite in-memory databases, executes SQL actions,
computes step rewards, and delegates final grading to task-specific graders.
"""

from __future__ import annotations

import io
import random
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Local imports
from graders import task1_grader, task2_grader, task3_grader
from data.generator import seed_database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
TASKS: Dict[str, Dict[str, Any]] = {
    "task1_data_cleaning": {
        "grader":    task1_grader,
        "max_steps": 30,
        "description": (
            "You are a Senior Data Engineer at a global trading company.\n"
            "The `employee_records` table was exported from a legacy HRIS system\n"
            "and loaded into the data warehouse with 2,000 rows of dirty HR data.\n\n"
            "Issues to fix:\n"
            "  1) gender values are inconsistent ('m','Male','MALE','f','female','N/A','unknown').\n"
            "     Normalize ALL to exactly: 'Male', 'Female', or 'Unknown'.\n"
            "  2) birth_date AND hire_date formats are broken:\n"
            "     e.g. 'March 5 1985', '05-07-1992', '1990/07/04', '3/5/1985'\n"
            "     Convert ALL date columns to ISO-8601: YYYY-MM-DD.\n"
            "  3) Some rows have NULL in the 'country' column. Replace NULLs with 'Unknown'.\n"
            "  DO NOT delete any rows. Use SQL UPDATE statements.\n"
            "  Hint: Start with SELECT to explore the scale and patterns.\n"
            "  When finished, set finalize_task=true."
        ),
    },
    "task2_pii_masking": {
        "grader":    task2_grader,
        "max_steps": 25,
        "description": (
            "You are a Data Privacy Engineer enforcing GDPR compliance.\n"
            "The `customers_pii` table has 500 rows containing raw sensitive data\n"
            "(emails, credit cards, SSNs, phone numbers, dates of birth).\n"
            "Data scientists MUST NOT see the raw PII.\n"
            "Your task: CREATE a VIEW named exactly 'masked_customers' that:\n"
            "  1) Masks email as '***@domain.com' (hide the username, keep the domain).\n"
            "  2) Masks credit_card as '****-****-****-XXXX' (keep last 4 digits).\n"
            "  3) Masks ssn as '*****-**-XX' (keep last 2 digits only).\n"
            "  4) Keeps these columns UNCHANGED: id, first_name, last_name, country, signup_year, risk_tier.\n"
            "  5) The view must return the same row count as the source table (500 rows).\n"
            "  When done, set finalize_task=true."
        ),
    },
    "task3_query_optimization": {
        "grader":    task3_grader,
        "max_steps": 35,
        "description": (
            "You are a Database Performance Engineer.\n"
            "A critical financial report runs on a 100,000-row sales_transactions table\n"
            "and is extremely slow due to a full table scan.\n\n"
            "The slow query is:\n\n"
            "  SELECT p.category, st.region_name,\n"
            "         SUM(st.total_amount) AS total_revenue,\n"
            "         COUNT(st.id)         AS order_count,\n"
            "         AVG(st.discount_pct) AS avg_discount\n"
            "  FROM sales_transactions st\n"
            "  JOIN products p ON st.product_id = p.id\n"
            "  WHERE st.sale_date BETWEEN '2023-01-01' AND '2023-12-31'\n"
            "    AND st.status = 'completed'\n"
            "  GROUP BY p.category, st.region_name\n"
            "  ORDER BY total_revenue DESC;\n\n"
            "Steps:\n"
            "  1) Run EXPLAIN QUERY PLAN on the slow query to diagnose it.\n"
            "  2) Identify which columns need indexing (WHERE clause columns).\n"
            "  3) CREATE the correct INDEX on sales_transactions.\n"
            "  4) Re-run EXPLAIN QUERY PLAN to verify SEARCH replaces SCAN.\n"
            "  When done, set finalize_task=true."
        ),
    },
}

# Regex for dangerous patterns the agent should not execute
DANGEROUS_SQL_RE = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE|DELETE\s+FROM\s+\w+\s*;?\s*$)\b",
    re.IGNORECASE,
)

# SQL commands that are read-only (safe exploration)
READ_ONLY_RE = re.compile(
    r"^\s*(SELECT|EXPLAIN|PRAGMA|WITH)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DataWarehouseEnvironment:
    """
    Per-session environment. Each HTTP session gets its OWN SQLite in-memory DB,
    so sessions are 100% isolated from each other.
    """

    def __init__(self, task_id: str = "task1_data_cleaning"):
        self._task_id    = task_id
        self._episode_id = str(uuid.uuid4())
        self._step       = 0
        self._total_reward = 0.0
        self._done       = False
        self._conn: Optional[sqlite3.Connection] = None
        self._grader_score: Optional[float] = None
        self._sql_count  = 0
        self._invalid_sql_count = 0
        self._tables_dropped    = 0
        self._last_commands: list[str] = []   # track for loop detection
        self._scenario_seed: int = 42          # set on each reset() for reproducibility

        # Validate task
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: Optional[str] = None, scenario_seed: Optional[int] = None) -> dict:
        """Initialize a fresh database using the Faker-based generator and return the first observation."""
        if task_id and task_id in TASKS:
            self._task_id = task_id

        # Each episode gets a unique seed: random by default, or pinned for reproducibility
        self._scenario_seed  = scenario_seed if scenario_seed is not None else random.randint(0, 99999)
        self._episode_id     = str(uuid.uuid4())
        self._step           = 0
        self._total_reward   = 0.0
        self._done           = False
        self._grader_score   = None
        self._sql_count      = 0
        self._invalid_sql_count = 0
        self._tables_dropped    = 0
        self._last_commands  = []

        # Fresh in-memory SQLite DB
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Generate realistic, production-scale data using Faker
        seed_database(self._conn, self._task_id, scenario_seed=self._scenario_seed)

        return self._build_observation(
            query_result=[],
            rows_affected=0,
            error_message=None,
            step_reward=0.0,
            info={
                "message":       "Episode started. Database seeded with production-scale data.",
                "scenario_seed": self._scenario_seed,
            },
        )

    def step(self, action: dict) -> dict:
        """Execute one action, return the resulting observation."""
        if self._done:
            return self._build_observation(
                query_result=[], rows_affected=0,
                error_message="Episode already done. Call reset() to start a new one.",
                step_reward=0.0, info={"warning": "episode_done"},
            )

        self._step += 1
        sql_command  = (action.get("sql_command") or "").strip()
        finalize     = bool(action.get("finalize_task", False))

        # ── Force-end at max_steps ─────────────────────────────────
        max_steps = TASKS[self._task_id]["max_steps"]
        if self._step >= max_steps and not finalize:
            finalize = True  # auto-finalize at limit

        # ── Handle finalization ────────────────────────────────────
        if finalize:
            return self._finalize()

        # ── Validate SQL ───────────────────────────────────────────
        if not sql_command:
            reward = -0.05
            self._total_reward += reward
            self._invalid_sql_count += 1
            return self._build_observation(
                query_result=[], rows_affected=0,
                error_message="Empty sql_command. Provide a SQL statement.",
                step_reward=reward,
                info={"hint": "You must provide a sql_command string in your action."},
            )

        # ── Danger check (DROP TABLE etc.) ─────────────────────────
        if DANGEROUS_SQL_RE.search(sql_command):
            reward = -0.5
            self._total_reward += reward
            self._tables_dropped += 1
            return self._build_observation(
                query_result=[], rows_affected=0,
                error_message=f"DANGEROUS SQL blocked: {sql_command[:80]}",
                step_reward=reward,
                info={"warning": "Destructive command detected and blocked."},
            )

        # ── Loop detection: same command 3 times → penalize ────────
        if sql_command in self._last_commands[-3:]:
            reward = -0.1
            self._total_reward += reward
            return self._build_observation(
                query_result=[], rows_affected=0,
                error_message="Repeated command detected. Try a different approach.",
                step_reward=reward,
                info={"warning": "loop_detected"},
            )

        # ── Execute SQL ────────────────────────────────────────────
        query_result, rows_affected, error_message, reward = self._execute_sql(sql_command)
        self._last_commands.append(sql_command)
        if len(self._last_commands) > 10:
            self._last_commands.pop(0)

        self._total_reward += reward
        return self._build_observation(
            query_result=query_result,
            rows_affected=rows_affected,
            error_message=error_message,
            step_reward=reward,
            info={},
        )

    @property
    def state(self) -> dict:
        """Return episode-level state metadata."""
        return {
            "episode_id":              self._episode_id,
            "task_id":                 self._task_id,
            "scenario_seed":           self._scenario_seed,
            "current_step":            self._step,
            "max_steps":               TASKS[self._task_id]["max_steps"],
            "total_reward":            round(self._total_reward, 4),
            "done":                    self._done,
            "sql_commands_executed":   self._sql_count,
            "invalid_sql_count":       self._invalid_sql_count,
            "tables_dropped":          self._tables_dropped,
            "task_finalized":          self._done,
            "grader_score":            self._grader_score,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _execute_sql(self, sql: str) -> Tuple[list, int, Optional[str], float]:
        """Run SQL, return (rows, rows_affected, error, reward)."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            self._conn.commit()
            self._sql_count += 1

            rows = []
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, r)) for r in cursor.fetchall()]

            rows_affected = cursor.rowcount if cursor.rowcount > 0 else 0

            # Reward for successful execution
            if READ_ONLY_RE.match(sql):
                reward = 0.02   # Small exploration bonus for SELECTs
            else:
                reward = 0.05   # Slightly larger for DML/DDL (making progress)

            return rows, rows_affected, None, reward

        except sqlite3.Error as e:
            self._invalid_sql_count += 1
            return [], 0, str(e), -0.1   # Penalty for bad SQL

    def _finalize(self) -> dict:
        """Run the grader, compute final reward, end the episode."""
        self._done = True
        grader_module = TASKS[self._task_id]["grader"]

        try:
            score, breakdown = grader_module.grade(self._conn)
        except Exception as e:
            score = 0.0
            breakdown = {"grader_error": str(e)}

        self._grader_score = score

        # Final bonus = grader score (adds to cumulative reward)
        final_reward = score
        self._total_reward += final_reward

        return self._build_observation(
            query_result=[],
            rows_affected=0,
            error_message=None,
            step_reward=final_reward,
            info={
                "grader_score": score,
                "grader_breakdown": breakdown,
                "message": f"Episode complete! Grader score: {score:.4f}",
            },
        )

    def _get_schema_info(self) -> Dict[str, Any]:
        """Return schema metadata for all tables and views."""
        cursor = self._conn.cursor()
        schema: Dict[str, Any] = {}
        try:
            cursor.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;"
            )
            objects = cursor.fetchall()
            for obj_name, obj_type in objects:
                try:
                    cursor.execute(f"PRAGMA table_info('{obj_name}');")
                    cols = [
                        {"name": r[1], "type": r[2], "notnull": bool(r[3]), "pk": bool(r[5])}
                        for r in cursor.fetchall()
                    ]
                    cursor.execute(f"SELECT COUNT(*) FROM '{obj_name}';")
                    row_count = cursor.fetchone()[0]
                    schema[obj_name] = {"type": obj_type, "columns": cols, "row_count": row_count}
                except Exception:
                    schema[obj_name] = {"type": obj_type, "error": "Could not inspect"}
        except Exception as e:
            schema["_error"] = str(e)
        return schema

    def _build_observation(
        self,
        query_result: list,
        rows_affected: int,
        error_message: Optional[str],
        step_reward: float,
        info: dict,
    ) -> dict:
        max_steps = TASKS[self._task_id]["max_steps"]
        return {
            "task_id":           self._task_id,
            "task_description":  TASKS[self._task_id]["description"],
            "schema_info":       self._get_schema_info(),
            "query_result":      query_result[:50],   # Cap rows for network efficiency
            "rows_affected":     rows_affected,
            "error_message":     error_message,
            "step_reward":       round(step_reward, 4),
            "total_reward":      round(self._total_reward, 4),
            "current_step":      self._step,
            "max_steps":         max_steps,
            "done":              self._done,
            "episode_id":        self._episode_id,
            "info":              info,
        }
