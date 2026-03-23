#!/usr/bin/env python3
"""Convenience entry point — run from repo root:

    python serve.py --checkpoint /path/to/ckpt --port 8000

Equivalent to:
    PYTHONPATH=examples python -m maniparena.launch --checkpoint ...
"""

import sys
from pathlib import Path

# Add examples/ to path so `from my_policy import MyPolicy` works in launch.py
sys.path.insert(0, str(Path(__file__).parent / "examples"))

from maniparena.launch import main

if __name__ == "__main__":
    main()
