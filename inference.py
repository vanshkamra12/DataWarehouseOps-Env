"""
DataWarehouseOps-Env — Automated Inference Script
Uses the OpenAI API client routed through the platform LiteLLM proxy.
Reads credentials from API_BASE_URL and API_KEY (injected by the platform).
Produces reproducible scores.

Usage:
    export API_BASE_URL=<platform-proxy-url>
    export API_KEY=<platform-api-key>
    export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
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

ENV_URL    = os.getenv("DATAWAREHOUSE_ENV_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
MAX_TURNS  = 30

from openai import OpenAI

# Use the platform-injected LiteLLM proxy with fallbacks exactly as the sample script
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy")),
)


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
# Strict STDOUT Logging Format (Required by Platform)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure action does not have newlines which breaks the parser
    action_str = str(action).replace('\n', '\\n')
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
    benchmark_name = "DataWarehouseOps"
    log_start(task=task_id, env=benchmark_name, model=MODEL_NAME)

    session_id, obs = env_reset(task_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0
    rewards_history = []
    success = False

    try:
        for step in range(1, MAX_TURNS + 1):
            messages.append({"role": "user", "content": build_user_message(obs)})

            sql, finalize, reasoning = call_llm(messages, client)
            
            # Format action exactly as a JSON string for the log
            action_obj = {"sql_command": sql, "finalize_task": finalize}
            action_str = json.dumps(action_obj)

            messages.append({
                "role": "assistant",
                "content": json.dumps({"sql_command": sql, "finalize_task": finalize, "reasoning": reasoning})
            })

            resp = env_step(session_id, sql, finalize, reasoning)
            obs  = resp["observation"]
            reward = float(obs.get("step_reward", 0.0))
            done = bool(obs.get("done", False))
            error = obs.get("error_message")

            rewards_history.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                final_score = float(obs.get("info", {}).get("grader_score", 0.0))
                success = final_score > 0.1
                break

            if step == MAX_TURNS:
                resp = env_step(session_id, None, finalize=True)
                obs  = resp["observation"]
                final_score = float(obs.get("info", {}).get("grader_score", 0.0))
                success = final_score > 0.1
                break

    finally:
        log_end(success=success, steps=len(rewards_history), score=final_score, rewards=rewards_history)

    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for task in TASKS:
        try:
            run_episode(task, client)
        except Exception as e:
            # Important: always emit [START] and [END] even on critical errors to satisfy regex
            log_start(task=task, env="DataWarehouseOps", model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        time.sleep(1)

