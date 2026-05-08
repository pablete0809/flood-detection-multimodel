from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded, usedforsecurity=False).hexdigest()


def window_key(window: tuple[date, date] | tuple[str, str]) -> str:
    start, end = window
    return f"{str(start).replace('-', '')}_{str(end).replace('-', '')}"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def with_retries(
    func: Callable[..., T],
    *args: Any,
    retries: int = 3,
    sleep_s: float = 2.0,
    label: str = "operation",
    **kwargs: Any,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOGGER.warning(
                "%s failed (%s/%s): %s", label, attempt, retries, exc
            )
            if attempt < retries:
                time.sleep(sleep_s * attempt)
    if last_exc is None:
        raise RuntimeError(f"{label} failed without exception context")
    raise last_exc


def build_run_dirs(base_out_dir: str | Path, run_name: str) -> dict[str, Path]:
    run_dir = ensure_dir(Path(base_out_dir) / run_name)
    paths = {
        "run_dir": run_dir,
        "metadata": ensure_dir(run_dir / "metadata"),
        "stacks": ensure_dir(run_dir / "stacks"),
        "intermediate": ensure_dir(run_dir / "intermediate"),
        "models": ensure_dir(run_dir / "models"),
        "predictions": ensure_dir(run_dir / "predictions"),
        "evaluation": ensure_dir(run_dir / "evaluation"),
    }
    return paths
