"""Compatibility wrapper around stdlib argparse.

Use this module from library code where legacy parsing still exists during
migration away from argparse-driven entrypoints.
"""

from __future__ import annotations

import argparse as _argparse

ArgumentParser = _argparse.ArgumentParser
BooleanOptionalAction = _argparse.BooleanOptionalAction
Namespace = _argparse.Namespace
SUPPRESS = _argparse.SUPPRESS

__all__ = [
    "ArgumentParser",
    "BooleanOptionalAction",
    "Namespace",
    "SUPPRESS",
]
