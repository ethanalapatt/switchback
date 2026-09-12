"""Switchback: an auditable speculative decoding engine.

Public surface is deliberately small. Import the typed state from
:mod:`switchback.types`; everything that touches Hugging Face Transformers lives
under :mod:`switchback.models`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
