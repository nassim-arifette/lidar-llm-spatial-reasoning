"""Read ``.env`` into the process environment, without ever printing a value.

The repository is public. Two rules follow, and they are enforced here rather
than left to discipline:

* a credential lives in ``.env``, which is gitignored, and never in a source
  file, a notebook cell, a command line or a results JSON;
* this module reports only *names*, never values, so a log, a traceback or a
  committed output file cannot leak one.

Nothing in the experiments needs a credential. The parser comparison runs Qwen
from open weights on the Colab GPU: that is free, reproducible by a reader who
has no account, and it is the only way to get grammar-constrained decoding,
which masks logits during sampling. A hosted endpoint does not expose them.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

ENV_FILENAME = ".env"


def load_env(path: Optional[str] = None, override: bool = False) -> List[str]:
    """Load ``KEY=VALUE`` lines into :data:`os.environ`.

    Returns the names that were set, so a caller can report what it found
    without quoting anything. A variable already present in the environment
    wins unless ``override`` is set: an exported value is a deliberate act,
    and a stale file should not silently replace it.
    """
    path = path or os.environ.get("SPATIAL_REASONING_ENV", ENV_FILENAME)
    if not os.path.exists(path):
        return []

    loaded: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if not name or not value:
                continue
            if name in os.environ and not override:
                continue
            os.environ[name] = value
            loaded.append(name)
    return loaded


def require(name: str, why: str) -> str:
    """Fetch a credential, with an error that says what to do and names no value."""
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set, and {why}. Put it in {ENV_FILENAME} "
            f"(see .env.example) or export it in your shell. "
            f"{ENV_FILENAME} is gitignored; never paste a credential into a "
            f"source file or a notebook cell.")
    return value


def present() -> Dict[str, bool]:
    """Which known credentials are available. Values are not returned."""
    load_env()
    return {name: bool(os.environ.get(name))
            for name in ("AI_GATEWAY_API_KEY", "HF_TOKEN")}
