"""Author: DEVADATH H K

Project: QAOA Max-Cut Optimization

Pytest path setup for Project 04 test execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
