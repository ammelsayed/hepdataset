"""Python adapters to the optional C++/CPython loop bridge.

The event processing still executes the exact Python loop implementation;
the compiled bridge provides C++-launchable and C++-callable entry points.
Build the optional extension and C++ launcher with ``make -C src/cpp``.
"""

from __future__ import annotations

import sys
from typing import Any

from . import _cpp_loop_bridge

SUPPORTED_LOOPS = {
    "adaptive_delphes",
    "explicit_delphes",
    "basic1_delphes",
    "basic2_delphes",
    "basic3_delphes",
}


def call_loop(loop_name: str, **loop_args: Any):
    """Invoke a HEPDataset ``loop_tree`` through the compiled C++ bridge."""
    if loop_name not in SUPPORTED_LOOPS:
        raise ValueError(f"Unsupported loop {loop_name!r}; choose from {sorted(SUPPORTED_LOOPS)}")
    return _cpp_loop_bridge.run_loop(loop_name, loop_args)


def run_cli(loop_name: str) -> int:
    """Run one supported loop's standard CLI via the C++ bridge."""
    if loop_name not in SUPPORTED_LOOPS:
        raise ValueError(f"Unsupported loop {loop_name!r}; choose from {sorted(SUPPORTED_LOOPS)}")
    result = _cpp_loop_bridge.run_cli(loop_name, list(sys.argv[1:]))
    return int(result or 0)
