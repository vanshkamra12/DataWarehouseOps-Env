"""
DataWarehouseOps-Env — OpenAI-powered Baseline Inference Script
Uses the OpenAI API client to run GPT-4o-mini against all 3 tasks.
Reads API key from OPENAI_API_KEY environment variable.
Produces reproducible scores.

Usage:
    export OPENAI_API_KEY=sk-...
    export DATAWAREHOUSE_ENV_URL=http://localhost:7860   # or your HF Space URL
    python baseline.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL   = os.environ.get("DATAWAREHOUSE_ENV_URL", "http://localhost:7860").rstrip("/")
API_KEY    = os.environ.get("OPENAI_API_KEY", "")
MODEL      = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TURNS  = 30

TASKS = [
    "task1_data_cleaning",
    "task2_pii_masking",
    "task3_query_optimization",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> tuple[str, dict]:
    """POST /reset → returns (session_id, observation)"""
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["session_id"], data["observation"]


def env_step(session_id: str, sql: Optional[str], finalize: bool = False, reasoning: str = "") -> dict:
    """POST /step → returns full response dict"""
    r = requests.post(
        f"{BASE_URL}/step",
        json={
            "session_id":    session_id,
            "sql_command":   sql,
            "finalize_task": finalize,
            "reasoning":     reasoning,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def call_llm(messages: list, client) -> tuple[str, str]:
    """Call OpenAI and return (sql_command, reasoning)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"sql_command": None, "finalize_task": False, "reasoning": raw}
    return parsed.get("sql_command"), parsed.get("finalize_task", False), parsed.get("reasoning", "")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Senior Data Engineer and SQL specialist.
You are operating inside a sandbox SQLite database environment.

At each turn you will receive:
- The current task description
- The current database schema (tables, columns, types, row counts)
- The result of your last SQL command
- Any error messages

You must respond with a JSON object containing:
{
  "sql_command": "<valid SQL string, or null if finalizing>",
  "finalize_task": <true if you believe the task is complete, false otherwise>,
  "reasoning": "<brief explanation of what you are doing>"
}

Rules:
1. ALWAYS start by running SELECT queries to understand the current state.
2. Use UPDATE, CREATE VIEW, CREATE INDEX as needed to complete the task.
3. NEVER use DROP TABLE, TRUNCATE, or DELETE without a WHERE clause.
4. Set finalize_task=true ONLY when you are confident the task is done.
5. Respond ONLY with valid JSON. No prose, no markdown.
"""


def build_user_message(obs: dict) -> str:
    schema_str = json.dumps(obs.get("schema_info", {}), indent=2)
    result_str = json.dumps(obs.get("query_result", [])[:10], indent=2)
    return f"""Task: {obs.get('task_id')}

OBJECTIVE:
{obs.get('task_description')}

CURRENT SCHEMA:
{schema_str}

LAST QUERY RESULT (first 10 rows):
{result_str}

ROWS AFFECTED: {obs.get('rows_affected', 0)}
ERROR: {obs.get('error_message') or 'None'}
STEP: {obs.get('current_step')}/{obs.get('max_steps')}
CUMULATIVE REWARD: {obs.get('total_reward')}
"""


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    session_id, obs = env_reset(task_id)
    print(f"  Session: {session_id}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0

    for turn in range(MAX_TURNS):
        messages.append({"role": "user", "content": build_user_message(obs)})

        sql, finalize, reasoning = call_llm(messages, client)
        print(f"  Turn {turn+1:02d} | finalize={finalize} | sql={str(sql)[:60]}...")

        messages.append({
            "role": "assistant",
            "content": json.dumps({"sql_command": sql, "finalize_task": finalize, "reasoning": reasoning})
        })

        resp = env_step(session_id, sql, finalize, reasoning)
        obs  = resp["observation"]

        if obs.get("done"):
            final_score = obs.get("info", {}).get("grader_score", 0.0)
            breakdown   = obs.get("info", {}).get("grader_breakdown", {})
            print(f"\n  ✅ Episode Done!")
            print(f"  Grader Score : {final_score:.4f}")
            print(f"  Breakdown    : {json.dumps(breakdown, indent=4)}")
            break

        if turn == MAX_TURNS - 1:
            # Force finalize
            resp = env_step(session_id, None, finalize=True)
            obs  = resp["observation"]
            final_score = obs.get("info", {}).get("grader_score", 0.0)
            print(f"  ⏰ Max turns reached. Final score: {final_score:.4f}")

    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable before running.")
        sys.exit(1)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    print(f"\n🚀 DataWarehouseOps-Env Baseline Inference")
    print(f"   Model   : {MODEL}")
    print(f"   Env URL : {BASE_URL}")
    print(f"   Tasks   : {TASKS}\n")

    scores = {}
    for task in TASKS:
        try:
            score = run_episode(task, client)
            scores[task] = round(score, 4)
        except Exception as e:
            print(f"  ERROR on {task}: {e}")
            scores[task] = 0.0
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"  📊 BASELINE RESULTS")
    print(f"{'='*60}")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<35} {score:.4f}  |{bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score: {avg:.4f}")
    print(f"{'='*60}\n")
