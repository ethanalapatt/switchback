"""Benchmark tooling: workload preparation, execution, and validation.

Kept out of ``src/switchback`` because the engine must not be able to import
anything that knows about datasets, cohorts, or engine labels.
"""

from __future__ import annotations
