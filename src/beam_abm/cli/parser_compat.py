"""Parser compatibility aliases for library modules."""

from __future__ import annotations

from beam_abm.cli.argparse_compat import (
    SUPPRESS,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)

__all__ = [
    "ArgumentParser",
    "BooleanOptionalAction",
    "Namespace",
    "SUPPRESS",
]
