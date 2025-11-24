"""
YAML configuration loader with environment-variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_CONFIG_PATH = "config.yml"
CONFIG_ENV_VAR = "DINOV3SEG_CONFIG"


def _candidate_paths(path: str | None = None) -> List[Path]:
    """
    Build the list of candidate configuration paths ordered by priority.
    """

    candidates: List[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
        return candidates
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        candidates.append(Path(env).expanduser())
        return candidates
    cwd = Path.cwd()
    search_dirs = [cwd] + list(cwd.parents)
    script_dir = Path(__file__).resolve().parent
    if script_dir not in search_dirs:
        search_dirs.append(script_dir)
    for directory in search_dirs:
        candidates.append(directory / DEFAULT_CONFIG_PATH)
    return candidates


def load_config(path: str | None = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary.

    >>> import tempfile, textwrap, os
    >>> tmp = tempfile.NamedTemporaryFile(delete=False)
    >>> _ = tmp.write(textwrap.dedent(\"\"\"
    ... logging:
    ...   level: debug
    ... paths:
    ...   raw_images_dir: /tmp/data
    ... \"\"\").encode())
    >>> tmp.close()
    >>> cfg = load_config(tmp.name)
    >>> cfg["logging"]["level"]
    'debug'

    >>> import tempfile, os
    >>> tmpdir = tempfile.mkdtemp()
    >>> cfg_path = Path(tmpdir) / "config.yml"
    >>> _ = cfg_path.write_text("logging:\\n  level: error\\n")
    >>> cwd = os.getcwd()
    >>> try:
    ...     os.chdir(tmpdir)
    ...     cfg = load_config()
    ... finally:
    ...     os.chdir(cwd)
    >>> cfg["logging"]["level"]
    'error'
    """

    candidate_path = None
    for candidate in _candidate_paths(path):
        if candidate.exists():
            candidate_path = candidate
            break
    if candidate_path is None:
        raise FileNotFoundError(
            f"Configuration file not found. Checked: {', '.join(str(p) for p in _candidate_paths(path))}"
        )
    with candidate_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must define a mapping: {candidate_path}")
    data["_config_path"] = str(candidate_path)
    return data
