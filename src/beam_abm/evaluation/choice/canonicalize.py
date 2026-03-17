"""Canonicalization utilities façade for choice-validation outputs."""

from __future__ import annotations

from beam_abm.evaluation.choice import _canonicalize_ids as _ids
from beam_abm.evaluation.choice import _canonicalize_metrics as _metrics
from beam_abm.evaluation.choice import _canonicalize_sample_rebuild as _sample_rebuild
from beam_abm.evaluation.choice import _canonicalize_summary as _summary
from beam_abm.evaluation.choice import _canonicalize_types as _types
from beam_abm.evaluation.choice._canonicalize_predictions import samples_to_predictions_df

CalibrationStatus = _types.CalibrationStatus
CanonicalizeStats = _types.CanonicalizeStats

normalize_choice_dataset_id = _ids.normalize_choice_dataset_id
normalize_model_slug = _ids.normalize_model_slug
_safe_token = _ids._safe_token
_row_dataset_id = _ids._row_dataset_id
_sample_dedupe_id = _ids._sample_dedupe_id

find_samples_jsonl = _sample_rebuild.find_samples_jsonl
read_jsonl = _sample_rebuild.read_jsonl
write_jsonl = _sample_rebuild.write_jsonl
merge_samples = _sample_rebuild.merge_samples
leaf_lock = _sample_rebuild.leaf_lock
rebuild_canonical_leaf = _sample_rebuild.rebuild_canonical_leaf
_git_commit_hash = _sample_rebuild._git_commit_hash
_find_git_root = _sample_rebuild._find_git_root

_spearman = _metrics._spearman
_auc = _metrics._auc
_mae_skill_score = _metrics._mae_skill_score
compute_metrics_df = _metrics.compute_metrics_df

write_strategies_summary = _summary.write_strategies_summary
write_models_strategies_summary = _summary.write_models_strategies_summary
_write_csv = _summary._write_csv

__all__ = [
    "CalibrationStatus",
    "CanonicalizeStats",
    "compute_metrics_df",
    "find_samples_jsonl",
    "leaf_lock",
    "merge_samples",
    "normalize_model_slug",
    "read_jsonl",
    "rebuild_canonical_leaf",
    "samples_to_predictions_df",
    "write_jsonl",
    "write_models_strategies_summary",
    "write_strategies_summary",
]
