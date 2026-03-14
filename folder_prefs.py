from __future__ import annotations

import json
from pathlib import Path


PREFS_FILE = Path(__file__).resolve().parent / ".folder_prefs.json"


def _workspace_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_prefs() -> dict[str, str]:
    if not PREFS_FILE.exists():
        return {}

    try:
        data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    prefs: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            prefs[key] = value
    return prefs


def _save_prefs(prefs: dict[str, str]) -> None:
    try:
        PREFS_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        return


def _display_path(path_value: Path, workspace_dir: Path) -> str:
    try:
        return str(path_value.relative_to(workspace_dir))
    except ValueError:
        return str(path_value)


def _resolve_default_path(pref_key: str, fallback_folder: Path) -> Path:
    workspace_dir = _workspace_dir()
    prefs = _load_prefs()
    stored_path = prefs.get(pref_key)

    if stored_path:
        candidate = Path(stored_path)
        if not candidate.is_absolute():
            candidate = workspace_dir / candidate
        if candidate.exists() and candidate.is_dir():
            return candidate

    return fallback_folder


def prompt_for_folder(pref_key: str, prompt_label: str, fallback_folder: Path) -> Path:
    """Prompt for a folder with remembered default and persist valid choices."""
    workspace_dir = _workspace_dir()
    default_folder = _resolve_default_path(pref_key, fallback_folder)
    default_display = _display_path(default_folder, workspace_dir)

    folder_input = input(f"{prompt_label} [{default_display}]: ").strip()
    chosen_folder = Path(folder_input) if folder_input else default_folder

    if not chosen_folder.is_absolute():
        chosen_folder = workspace_dir / chosen_folder

    if chosen_folder.exists() and chosen_folder.is_dir():
        prefs = _load_prefs()
        prefs[pref_key] = str(chosen_folder)
        _save_prefs(prefs)

    return chosen_folder
