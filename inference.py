"""
DataWarehouseOps-Env — Automated Inference Script
Uses the OpenAI API client to run against all tasks.
Reads credentials from HF_TOKEN, MODEL_NAME, and API_BASE_URL.
Produces reproducible scores.

Usage:
    export HF_TOKEN=hf_...
    export MODEL_NAME=meta-llama/Llama-3-...
    export API_BASE_URL=https://api.endpoints.huggingface.cloud/...
    python inference.py
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

ENV_URL      = os.getenv("DATAWAREHOUSE_ENV_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MAX_TURNS    = 30

TASKS = [
    "task1_data_cleaning",
    "task2_pii_masking",
    "task3_query_optimization",
]

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_local_envs = {}

def env_reset(task_id: str) -> tuple[str, dict]:
    """POST /reset or native fallback"""
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=5)
        r.raise_for_status()
        data = r.json()
        return data["session_id"], data["observation"]
    except Exception:
        from server.environment import DataWarehouseEnvironment
        env = DataWarehouseEnvironment(task_id=task_id)
        _local_envs[task_id] = env
        return task_id, env.reset(task_id=task_id)

def env_step(session_id: str, sql: Optional[str], finalize: bool = False, reasoning: str = "") -> dict:
    """POST /step or native fallback"""
    try:
        r = requests.post(
            f"{ENV_URL}/step",
            json={
                "session_id":    session_id,
                "sql_command":   sql,
                "finalize_task": finalize,
                "reasoning":     reasoning,
            },
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        env = _local_envs[session_id]
        obs = env.step({
            "sql_command": sql,
            "finalize_task": finalize,
            "reasoning": reasoning
        })
        return {"observation": obs}


def call_llm(messages: list, client) -> tuple[str, str]:
    """Call OpenAI and return (sql_command, reasoning)."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )
    raw = response.choices[0].message.content
    
    # Strip markdown if model returned it
    if raw.startswith("```json"):
        raw = raw.replace("```json", "", 1).replace("```", "")
    elif raw.startswith("```"):
        raw = raw.replace("```", "", 2)
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
    print(f"[START] task={task_id}", flush=True)

    session_id, obs = env_reset(task_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0

    for step in range(1, MAX_TURNS + 1):
        messages.append({"role": "user", "content": build_user_message(obs)})

        sql, finalize, reasoning = call_llm(messages, client)
        
        # Format action exactly like sample script
        action_str = f"sql='{str(sql)[:40]}...', finalize={finalize}"
        print(f"Step {step}: model suggested -> {action_str}")

        messages.append({
            "role": "assistant",
            "content": json.dumps({"sql_command": sql, "finalize_task": finalize, "reasoning": reasoning})
        })

        resp = env_step(session_id, sql, finalize, reasoning)
        obs  = resp["observation"]
        reward = obs.get("step_reward", 0.0)
        done = obs.get("done", False)
        error = obs.get("error_message")

        print(
            "  Reward: "
            f"{reward:+.2f} | Done: {done} | Last action error: "
            f"{error}"
        )
        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            final_score = obs.get("info", {}).get("grader_score", 0.0)
            breakdown   = obs.get("info", {}).get("grader_breakdown", {})
            print("Episode complete.")
            print(f"  Grader Score : {final_score:.4f}")
            print(f"  Breakdown    : {json.dumps(breakdown, indent=4)}")
            print(f"[END] task={task_id} score={final_score} steps={step}", flush=True)
            break

        if step == MAX_TURNS:
            resp = env_step(session_id, None, finalize=True)
            obs  = resp["observation"]
            final_score = obs.get("info", {}).get("grader_score", 0.0)
            print(f"Reached max steps ({MAX_TURNS}). Final score: {final_score:.4f}")
            print(f"[END] task={task_id} score={final_score} steps={step}", flush=True)

    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "API_BASE_URL" not in os.environ:
        os.environ["API_BASE_URL"] = "https://api.openai.com/v1"
    if "API_KEY" not in os.environ:
        os.environ["API_KEY"] = os.environ.get("HF_TOKEN", "dummy-key")

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    print(f"\n🚀 DataWarehouseOps-Env Inference")
    print(f"   Model       : {MODEL_NAME}")
    print(f"   API Base    : {API_BASE_URL}")
    print(f"   Env URL     : {ENV_URL}")
    print(f"   Tasks   : {TASKS}\n")

    scores = {}
    for task in TASKS:
        try:
            score = run_episode(task, client)
            scores[task] = round(score, 4)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [CRITICAL PROXY ERROR] on {task}: {e}")
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
