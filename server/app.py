"""
DataWarehouseOps-Env — FastAPI Server
Exposes the full OpenEnv API spec + the required extra endpoints:
  POST /reset      → start / restart an episode
  POST /step       → take an action
  GET  /state      → get current episode metadata
  GET  /tasks      → list all tasks + action schema
  POST /grader     → finalize and return grader score
  POST /baseline   → run the built-in baseline agent and return scores
  GET  /health     → liveness probe
  GET  /           → serves the interactive web dashboard
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('datawarehouse-env')

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

# Add parent dir so graders can be imported
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from environment import DataWarehouseEnvironment, TASKS

# ---------------------------------------------------------------------------
# Per-session store  (session_id → environment instance)
# ---------------------------------------------------------------------------

_sessions: Dict[str, DataWarehouseEnvironment] = {}

DEFAULT_TASK = "task1_data_cleaning"


def _get_or_create_session(session_id: Optional[str], task_id: str) -> tuple[str, DataWarehouseEnvironment]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = session_id or str(uuid.uuid4())
    env = DataWarehouseEnvironment(task_id=task_id)
    _sessions[sid] = env
    return sid, env


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id:    Optional[str] = DEFAULT_TASK
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    sql_command:    Optional[str] = None
    finalize_task:  bool          = False
    reasoning:      Optional[str] = None
    session_id:     Optional[str] = None


class GraderRequest(BaseModel):
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown — close all DB connections
    for env in _sessions.values():
        try:
            if env._conn:
                env._conn.close()
        except Exception:
            pass


app = FastAPI(
    title="DataWarehouseOps-Env",
    description=(
        "🏭 An OpenEnv-compliant RL environment simulating real-world enterprise "
        "data engineering tasks: data cleaning, PII privacy shielding, and query optimization."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static dashboard at /ui
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive dashboard at the root URL."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "DataWarehouseOps-Env is running. Visit /docs for the API."}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe — OpenEnv validator hits this first."""
    return {"status": "healthy", "environment": "DataWarehouseOps-Env", "version": "1.0.0"}


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    """
    Start a new episode (or restart an existing session).
    Returns the initial observation with the full task description and DB schema.
    """
    if req is None:
        req = ResetRequest()
        
    task_id = req.task_id or DEFAULT_TASK
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")

    session_id, env = _get_or_create_session(req.session_id, task_id)
    observation = env.reset(task_id=task_id)

    return {"session_id": session_id, "observation": observation}


@app.post("/step")
async def step(req: StepRequest):
    """
    Execute one SQL action in the environment.
    Returns observation, reward, done flag, and info.
    """
    if not req.session_id or req.session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing session_id. Call /reset first to get a session_id."
        )

    env = _sessions[req.session_id]
    action = {
        "sql_command":   req.sql_command,
        "finalize_task": req.finalize_task,
        "reasoning":     req.reasoning,
    }
    observation = env.step(action)

    return {
        "session_id":  req.session_id,
        "observation": observation,
        "reward":      observation["step_reward"],
        "done":        observation["done"],
        "info":        observation["info"],
    }


@app.get("/state")
async def state(session_id: Optional[str] = None):
    """Return current episode state metadata (step count, cumulative reward, etc.)."""
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id.")
    return _sessions[session_id].state


@app.get("/tasks")
async def list_tasks():
    """
    Return all available tasks and the action schema (fields required for /step).
    Required by the hackathon spec.
    """
    tasks_info = []
    for tid, meta in TASKS.items():
        tasks_info.append({
            "task_id":     tid,
            "description": meta["description"],
            "max_steps":   meta["max_steps"],
            "difficulty":  {"task1_data_cleaning": "easy",
                            "task2_pii_masking":   "medium",
                            "task3_query_optimization": "hard"}.get(tid, "unknown"),
        })

    return {
        "tasks": tasks_info,
        "action_schema": {
            "sql_command": {
                "type": "string",
                "required": False,
                "description": "SQL statement to execute (SELECT, UPDATE, CREATE VIEW, CREATE INDEX, EXPLAIN QUERY PLAN, etc.)"
            },
            "finalize_task": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "Set true when you believe the task is complete. Triggers the grader."
            },
            "reasoning": {
                "type": "string",
                "required": False,
                "description": "Optional chain-of-thought reasoning (not graded, logged for research)."
            },
            "session_id": {
                "type": "string",
                "required": True,
                "description": "Session identifier returned by /reset."
            },
        }
    }


@app.post("/grader")
async def grader(req: GraderRequest):
    """
    Finalize the episode and return the grader score.
    Equivalent to calling /step with finalize_task=true.
    """
    if not req.session_id or req.session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id.")

    env = _sessions[req.session_id]
    if env._done:
        return {
            "session_id":   req.session_id,
            "grader_score": env._grader_score,
            "message":      "Episode already finalized.",
        }

    observation = env.step({"sql_command": None, "finalize_task": True})
    return {
        "session_id":   req.session_id,
        "grader_score": observation["info"].get("grader_score", 0.0),
        "breakdown":    observation["info"].get("grader_breakdown", {}),
        "total_reward": observation["total_reward"],
    }


@app.post("/baseline")
async def baseline():
    """
    Run the LLM-powered baseline agent against all 3 tasks using the
    platform-injected LiteLLM proxy (API_BASE_URL + API_KEY).
    All LLM calls go through os.environ["API_BASE_URL"] as required.
    """
    import json as _json
    from openai import OpenAI as _OpenAI

    # Use the platform-injected proxy credentials — required by the hackathon
    llm_client = _OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

    SYSTEM_PROMPT = (
        "You are an expert Senior Data Engineer and SQL specialist.\n"
        "You are operating inside a sandbox SQLite database environment.\n\n"
        "At each turn you will receive:\n"
        "- The current task description\n"
        "- The current database schema (tables, columns, types, row counts)\n"
        "- The result of your last SQL command\n"
        "- Any error messages\n\n"
        "You must respond with a JSON object containing:\n"
        "{\n"
        "  \"sql_command\": \"<valid SQL string, or null if finalizing>\",\n"
        "  \"finalize_task\": <true if you believe the task is complete, false otherwise>,\n"
        "  \"reasoning\": \"<brief explanation of what you are doing>\"\n"
        "}\n\n"
        "Rules:\n"
        "1. ALWAYS start by running SELECT queries to understand the current state.\n"
        "2. Use UPDATE, CREATE VIEW, CREATE INDEX as needed to complete the task.\n"
        "3. NEVER use DROP TABLE, TRUNCATE, or DELETE without a WHERE clause.\n"
        "4. Set finalize_task=true ONLY when you are confident the task is done.\n"
        "5. Respond ONLY with valid JSON. No prose, no markdown.\n"
    )

    def _build_user_msg(obs: dict) -> str:
        schema_str = _json.dumps(obs.get("schema_info", {}), indent=2)
        result_str = _json.dumps(obs.get("query_result", [])[:10], indent=2)
        return (
            f"Task: {obs.get('task_id')}\n\n"
            f"OBJECTIVE:\n{obs.get('task_description')}\n\n"
            f"CURRENT SCHEMA:\n{schema_str}\n\n"
            f"LAST QUERY RESULT (first 10 rows):\n{result_str}\n\n"
            f"ROWS AFFECTED: {obs.get('rows_affected', 0)}\n"
            f"ERROR: {obs.get('error_message') or 'None'}\n"
            f"STEP: {obs.get('current_step')}/{obs.get('max_steps')}\n"
            f"CUMULATIVE REWARD: {obs.get('total_reward')}\n"
        )

    def _call_llm(messages: list) -> tuple:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        if raw.startswith("```json"):
            raw = raw.replace("```json", "", 1).replace("```", "")
        elif raw.startswith("```"):
            raw = raw.replace("```", "", 2)
        try:
            parsed = _json.loads(raw)
        except Exception:
            parsed = {"sql_command": None, "finalize_task": False, "reasoning": raw}
        return (
            parsed.get("sql_command"),
            parsed.get("finalize_task", False),
            parsed.get("reasoning", ""),
        )

    results = []
    max_turns = 20

    for task_id in TASKS.keys():
        print(f"[START] task={task_id}", flush=True)
        env = DataWarehouseEnvironment(task_id=task_id)
        obs = env.reset(task_id=task_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        final_score = 0.0

        for step_num in range(1, max_turns + 1):
            messages.append({"role": "user", "content": _build_user_msg(obs)})
            sql, finalize, reasoning = _call_llm(messages)
            messages.append({
                "role": "assistant",
                "content": _json.dumps({
                    "sql_command": sql,
                    "finalize_task": finalize,
                    "reasoning": reasoning,
                }),
            })
            obs = env.step({
                "sql_command": sql,
                "finalize_task": finalize,
                "reasoning": reasoning,
            })
            print(f"[STEP] step={step_num} reward={obs.get('step_reward', 0.0)}", flush=True)
            if obs.get("done"):
                final_score = obs.get("info", {}).get("grader_score", 0.0)
                break
            if step_num == max_turns:
                obs = env.step({"sql_command": None, "finalize_task": True})
                final_score = obs.get("info", {}).get("grader_score", 0.0)

        print(f"[END] task={task_id} score={final_score}", flush=True)
        results.append({"task_id": task_id, "score": final_score})

    return {"baseline_scores": results, "agent": "llm_baseline_v1"}




# ---------------------------------------------------------------------------
# Date normalization helper for Task 1 baseline
# ---------------------------------------------------------------------------

DATE_FORMATS = [
    "%Y-%m-%d",       # ISO correct
    "%d-%m-%Y",       # EU dash: 15-04-1984
    "%Y/%m/%d",       # Slash ISO: 1990/07/04
    "%B %d %Y",       # "March 05 1985"
    "%B %d, %Y",      # "March 05, 1985"
    "%B %Y",          # Partial month+year (fallback)
    "%d %B %Y",       # "05 March 1985"
    "%m/%d/%Y",       # US slash: 3/15/1985 → need day-padding variant
    "%-m/%-d/%Y",     # Unpadded US (Unix only): 3/5/1985  ← key missing format
]

def _parse_date(val: str) -> str:
    """Try all known formats and return ISO-8601, or original if unknown."""
    if not val:
        return val
    from datetime import datetime
    val = val.strip()
    # Try exact matches first
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Fallback: handle slash-separated dates not covered by strptime on macOS
    # e.g. "3/5/1985" → datetime(1985, 3, 5)
    import re
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', val)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return val  # Return as-is if we can't parse


# ---------------------------------------------------------------------------
# Built-in Heuristic Baseline (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _run_heuristic_baseline(env: DataWarehouseEnvironment, task_id: str) -> float:
    """
    A hand-crafted rule-based agent that solves each task deterministically.
    Task 1 — Real Northwind employee data (birth_date + hire_date + gender + country)
    Task 2 — Real Chinook customer data (email + credit_card + ssn PII masking)
    Task 3 — Real Northwind Order Details (sale_date + status index optimization)
    """
    if task_id == "task1_data_cleaning":
        # Step 1 — normalize genders
        cmds = [
            "UPDATE employee_records SET gender = 'Male'    WHERE LOWER(TRIM(gender)) IN ('m','male','man');",
            "UPDATE employee_records SET gender = 'Female'  WHERE LOWER(TRIM(gender)) IN ('f','female','woman');",
            "UPDATE employee_records SET gender = 'Unknown' WHERE gender NOT IN ('Male','Female') OR gender IS NULL;",
            "UPDATE employee_records SET country = 'Unknown' WHERE country IS NULL;",
        ]
        for cmd in cmds:
            env.step({"sql_command": cmd, "finalize_task": False})

        # Step 2 — Python-level date fix via direct DB access (birth_date + hire_date)
        cursor = env._conn.cursor()
        for col in ("birth_date", "hire_date"):
            cursor.execute(f"SELECT id, {col} FROM employee_records;")
            rows = cursor.fetchall()
            for row in rows:
                rid, raw_date = row[0], row[1]
                if raw_date:
                    fixed = _parse_date(str(raw_date))
                    if fixed != raw_date:
                        cursor.execute(
                            f"UPDATE employee_records SET {col} = ? WHERE id = ?;",
                            (fixed, rid)
                        )
        env._conn.commit()
        env._sql_count += 1

    elif task_id == "task2_pii_masking":
        cmds = [
            """CREATE VIEW IF NOT EXISTS masked_customers AS
               SELECT
                   id, first_name, last_name, company,
                   '***@' || SUBSTR(email, INSTR(email,'@')+1)          AS email,
                   '****-****-****-' || SUBSTR(REPLACE(credit_card,'-',''), -4)  AS credit_card,
                   '*****-**-' || SUBSTR(REPLACE(ssn,'-',''), -2)        AS ssn,
                   city, country, signup_year, risk_tier
               FROM customers_pii;"""
        ]
        for cmd in cmds:
            env.step({"sql_command": cmd, "finalize_task": False})

    elif task_id == "task3_query_optimization":
        cmds = [
            "EXPLAIN QUERY PLAN SELECT st.category, st.region, SUM(st.total_amount) AS total_revenue, COUNT(st.id) AS order_count, AVG(st.discount_pct) AS avg_discount_pct FROM sales_transactions st WHERE st.sale_date >= '2017-01-01' AND st.sale_date < '2018-01-01' AND st.status = 'completed' GROUP BY st.category, st.region ORDER BY total_revenue DESC;",
            "CREATE INDEX IF NOT EXISTS idx_sales_date_status ON sales_transactions(sale_date, status);",
        ]
        for cmd in cmds:
            env.step({"sql_command": cmd, "finalize_task": False})

    obs = env.step({"sql_command": None, "finalize_task": True})
    return obs["info"].get("grader_score", 0.0)


def main():
    import uvicorn
    # The default host and port standard for FastAPI / Docker Space tests
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
