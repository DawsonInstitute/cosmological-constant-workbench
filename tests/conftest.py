from __future__ import annotations

import sys
from pathlib import Path

# hts-coils style: tests run without requiring editable install.
# This prepends the repo's src/ so `import ccw` works after a fresh clone.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
