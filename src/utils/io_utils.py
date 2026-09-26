"""I/O utilities for VEGA-KG."""
import json
import os
import pickle
import signal
import subprocess
import yaml
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_config(config_path: str | Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    return load_yaml(config_path)


def shutdown_vllm():
    """Stop all vLLM servers owned by the current user."""
    project_dir = Path(__file__).parent.parent.parent
    stop_script = project_dir / "scripts" / "stop_vllm.sh"
    if stop_script.exists():
        subprocess.run(["bash", str(stop_script)], check=False)
    else:
        # Fallback: kill vllm processes directly
        subprocess.run(
            ["pkill", "-u", os.environ.get("USER", ""), "-f", "vllm serve"],
            check=False,
        )
