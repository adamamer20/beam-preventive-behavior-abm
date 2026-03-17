"""P-reference perturbation effects for belief-update microvalidation.

Facade module preserving the stable import surface while delegating
responsibilities to focused internal modules.
"""

from __future__ import annotations

from .reference_model_loading import default_ref_state_models
from .reference_pe_table import SignalReferenceBundle, build_signal_reference_bundle, compute_pe_ref_p_table
from .reference_summary_metrics import compute_choice_like_summary

__all__ = [
    "SignalReferenceBundle",
    "build_signal_reference_bundle",
    "compute_choice_like_summary",
    "compute_pe_ref_p_table",
    "default_ref_state_models",
]
