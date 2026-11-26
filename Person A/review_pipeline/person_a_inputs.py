"""Find the shared source datasets and the earlier Amazon label audit."""

import os
from pathlib import Path


def shared_project(root):
    configured = os.environ.get("REVIEW_SHARED_PROJECT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(root).resolve()
    if (root / "datasets" / "ECommerce_Project").is_dir():
        return root
    return root.parent
