"""Small provenance/report utilities shared by CLI entrypoints."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def timestamp() -> str:
    """Return an unambiguous UTC timestamp for machine-readable run provenance."""
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str], cwd: Path | None = None) -> dict:
    """Capture a diagnostic command without masking its exit code."""
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {
            "command": args,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "exit_code": None, "error": str(exc)}


def write_json(path: Path, payload: dict) -> None:
    """Save a structured report with no credential discovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config_hash(path: Path) -> str:
    """Hash exact config bytes for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
