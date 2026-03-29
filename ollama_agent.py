"""
DataWarehouseOps-Env — Ollama-powered Inference Script
Uses the locally running Ollama instance to test the environment using Llama 3!
"""

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
MODEL      = "llama3:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_TURNS  = 15  # Keep it short for local testing

TASKS = [
    "task1_data_cleaning",
    "task2_pii_masking",
    "task3_query_optimization",
]

# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> tuple[str, dict]:
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["session_id"], data["observation"]

def env_step(session_id: str, sql: Optional[str], finalize: bool = False, reasoning: str = "") -> dict:
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

def call_ollama(prompt: str) -> tuple[str, bool, str]:
    """Call local Ollama using the /api/generate endpoint."""
    print("  [🧠 Llama 3 is thinking...]")
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json().get("response", "{}")
    
    try:
        parsed = json.loads(raw)
    except Exception:
        print(f"  [Error parsing JSON from Llama: {raw[:50]}]")
        parsed = {"sql_command": None, "finalize_task": False, "reasoning": raw}
        
    return parsed.get("sql_command"), parsed.get("finalize_task", False), parsed.get("reasoning", "")

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
def build_ollama_prompt(obs: dict) -> str:
    schema_str = json.dumps(obs.get("schema_info", {}), indent=2)
    result_str = json.dumps(obs.get("query_result", [])[:5], indent=2)
    
    return f"""You are an elite Senior Data Engineer. You are inside a sandbox database.
Solve the following task step-by-step.

OBJECTIVE:
{obs.get('task_description')}

CURRENT SCHEMA:
{schema_str}

LAST QUERY RESULT (first 5 rows):
{result_str}
ERROR: {obs.get('error_message') or 'None'}
STEP: {obs.get('current_step')}/{obs.get('max_steps')}

You MUST respond strictly with a JSON object. No markdown formatting, no comments, no prose.
Format:
{{
  "sql_command": "<valid SQL string here>",
  "finalize_task": false,
  "reasoning": "<explain your thought process>"
}}

Set finalize_task to true ONLY when you are absolutely completely done with the task.
Output ONLY JSON.
"""

def run_episode(task_id: str) -> float:
    print(f"\n{'='*60}\n  TASK: {task_id}\n{'='*60}")
    session_id, obs = env_reset(task_id)
    final_score = 0.0

    for turn in range(MAX_TURNS):
        prompt = build_ollama_prompt(obs)
        sql, finalize, reasoning = call_ollama(prompt)
        
        # Guard against weird LLM hallucinations
        if sql is not None and not isinstance(sql, str): sql = str(sql)
        if finalize not in [True, False]: finalize = False
        if not isinstance(reasoning, str): reasoning = str(reasoning)

        print(f"  Turn {turn+1:02d} | Finalize: {finalize}")
        print(f"  Reasoning : {reasoning}")
        print(f"  SQL       : {str(sql)[:80]}")

        resp = env_step(session_id, sql, finalize, reasoning)
        obs  = resp["observation"]

        if obs.get("done"):
            final_score = obs.get("info", {}).get("grader_score", 0.0)
            breakdown   = obs.get("info", {}).get("grader_breakdown", {})
            print(f"\n  ✅ Episode Done! Score: {final_score:.4f}")
            print(f"  Breakdown: {json.dumps(breakdown, indent=2)}")
            break

        if turn == MAX_TURNS - 1:
            resp = env_step(session_id, None, finalize=True)
            obs  = resp["observation"]
            final_score = obs.get("info", {}).get("grader_score", 0.0)
            print(f"  ⏰ Max turns reached. Final score: {final_score:.4f}")

    return final_score

if __name__ == "__main__":
    print(f"\n🚀 DataWarehouseOps-Env local Llama 3 Inference")
    print(f"   Model : {MODEL}")
    print(f"   Note  : Requires Ollama running in the background")

    scores = {}
    for task in TASKS:
        try:
            score = run_episode(task)
            scores[task] = round(score, 4)
        except Exception as e:
            print(f"  ERROR on {task}: {e}")
            scores[task] = 0.0
        time.sleep(1)

    print(f"\n{'='*60}\n  📊 OLLAMA RESULTS\n{'='*60}")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<35} {score:.4f}  |{bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score: {avg:.4f}\n{'='*60}\n")
