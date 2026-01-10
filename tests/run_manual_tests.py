from __future__ import annotations

import sys

import pytest


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else ["-q"]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
