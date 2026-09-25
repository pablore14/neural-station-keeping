"""
Experiment traceability helpers.

Records which exact version of the code produced a given result, so any output
can be reproduced later. The git commit hash is the pointer: from it the full
state of the code (including config) can be recovered, so there is no need to
copy the config into every result file.
"""

import os
import subprocess
from datetime import datetime


def get_git_commit():
    """Return the current git commit hash, or 'unknown' if unavailable.

    A trailing '-dirty' marks that there are uncommitted changes, which is a
    warning sign: the result may not be exactly reproducible from the hash.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        changed = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()
        return commit + ("-dirty" if changed else "")
    except Exception:
        return "unknown"


def make_run_dir(base_dir, prefix="run"):
    """Create and return a timestamped output directory: base_dir/prefix_YYYYmmdd_HHMMSS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    print(f"[run] {path}")
    return path