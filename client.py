"""
DataWarehouseOps-Env — Python Client
A thin HTTP client wrapping the REST API for use in RL training loops.

Usage (sync):
    from client import DataWarehouseEnv, DataWHAction

    with DataWarehouseEnv(base_url="http://localhost:7860") as env:
        obs = env.reset(task_id="task1_data_cleaning")
        result = env.step(DataWHAction(sql_command="SELECT * FROM customer_records LIMIT 5;"))
        print(result.observation)

Usage (async):
    async with DataWarehouseEnv(base_url="http://localhost:7860") as env:
        obs = await env.async_reset(task_id="task2_pii_masking")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from models import DataWHAction, DataWHObservation, DataWHState


@dataclass
class StepResult:
    """Returned by env.step() — mirrors OpenEnv StepResult."""
    observation: DataWHObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = field(default_factory=dict)
    session_id:  str = ""


class DataWarehouseEnv:
    """
    Synchronous HTTP client for DataWarehouseOps-Env.
    Compatible with the OpenEnv HTTPEnvClient pattern.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url   = (base_url or os.environ.get("DATAWAREHOUSE_ENV_URL", "http://localhost:7860")).rstrip("/")
        self.timeout    = timeout
        self.session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Context manager support                                              #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> DataWarehouseEnv:
        return self

    def __exit__(self, *args) -> None:
        pass   # Connection is stateless (HTTP) — nothing to close

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: str = "task1_data_cleaning") -> DataWHObservation:
        """Start a new episode. Returns the initial observation."""
        payload = {"task_id": task_id}
        if self.session_id:
            payload["session_id"] = self.session_id

        resp = self._post("/reset", payload)
        self.session_id = resp["session_id"]
        return DataWHObservation(**resp["observation"])

    def step(self, action: DataWHAction) -> StepResult:
        """Execute one action. Returns (observation, reward, done, info)."""
        if not self.session_id:
            raise RuntimeError("Call reset() before step().")

        payload = {
            "session_id":    self.session_id,
            "sql_command":   action.sql_command,
            "finalize_task": action.finalize_task,
            "reasoning":     action.reasoning,
        }
        resp = self._post("/step", payload)
        return StepResult(
            observation=DataWHObservation(**resp["observation"]),
            reward=resp["reward"],
            done=resp["done"],
            info=resp["info"],
            session_id=self.session_id,
        )

    def state(self) -> DataWHState:
        """Get current episode state metadata."""
        if not self.session_id:
            raise RuntimeError("Call reset() before state().")
        resp = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
            timeout=self.timeout
        )
        resp.raise_for_status()
        return DataWHState(**resp.json())

    def list_tasks(self) -> List[Dict[str, Any]]:
        """Return all available tasks."""
        resp = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["tasks"]

    def finalize(self) -> Dict[str, Any]:
        """Shortcut to finalize the current episode and get grader score."""
        if not self.session_id:
            raise RuntimeError("Call reset() first.")
        resp = self._post("/grader", {"session_id": self.session_id})
        return resp

    def health(self) -> Dict[str, Any]:
        """Check if the server is alive."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"HTTP {resp.status_code} from {path}: {resp.text}") from e
        return resp.json()
