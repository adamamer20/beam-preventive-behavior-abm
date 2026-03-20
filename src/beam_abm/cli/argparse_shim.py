"""Compatibility shim for legacy argparse-based modules.

Evaluation modules are being migrated to Typer CLI wrappers under evaluation/scripts.
Source modules should not parse CLI arguments directly; this shim intentionally fails
on parser usage to enforce the library-only contract.
"""

from __future__ import annotations

from typing import Any

SUPPRESS = object()
BooleanOptionalAction = object
Namespace = object


class _ArgparseDisabledError(RuntimeError):
    pass


def _fail(*_: Any, **__: Any) -> Any:
    raise _ArgparseDisabledError(
        "argparse usage in src/beam_abm/evaluation is disabled; use evaluation/scripts Typer wrappers."
    )


class _ArgParserStub:
    def __init__(self, *_: Any, **__: Any) -> None:
        _fail()

    def __getattr__(self, _: str) -> Any:
        return _fail


ArgumentParser = _ArgParserStub
