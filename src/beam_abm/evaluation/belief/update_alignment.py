"""Compute belief-update alignment metrics for Phase 0 prompt tournament."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beam_abm.evaluation.artifacts import utc_timestamp_iso
from beam_abm.evaluation.belief.reference import compute_choice_like_summary
from beam_abm.evaluation.common.labels import load_possible_numeric_bounds_from_clean_spec


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _strength_score(raw: str | None) -> int:
    strength = str(raw or "").strip().lower()
    mapping = {"primary": 3, "secondary": 2, "weak": 1, "none": 0}
    return mapping.get(strength, 0)


def _direction_pass(expected: str, delta: float, eps: float) -> bool:
    exp = str(expected or "").strip().lower()
    if exp in {"none", "flat", "no_change"}:
        return abs(delta) <= eps
    if exp in {"up", "increase", "higher"}:
        return delta > eps
    if exp in {"down", "decrease", "lower"}:
        return delta < -eps
    return False


def _signal_has_manual_expectations(signal: dict[str, Any]) -> bool:
    return any(
        [
            isinstance(signal.get("mutable_state_expectations"), list),
            isinstance(signal.get("mutable_state_ranking"), list),
            isinstance(signal.get("effects"), dict) and bool(signal.get("effects")),
            isinstance(signal.get("primary_targets"), list) and bool(signal.get("primary_targets")),
            isinstance(signal.get("secondary_targets"), list) and bool(signal.get("secondary_targets")),
        ]
    )


def _build_expectations(signal: dict, mutable_state: list[str]) -> dict[str, dict[str, Any]]:
    expectations: dict[str, dict[str, Any]] = {}
    for var in mutable_state:
        expectations[var] = {"direction": "none", "strength": "none", "rank": None}

    if isinstance(signal.get("mutable_state_expectations"), list):
        for entry in signal.get("mutable_state_expectations", []):
            if not isinstance(entry, dict):
                continue
            var = str(entry.get("var") or entry.get("name") or "").strip()
            if not var:
                continue
            expectations.setdefault(var, {})
            expectations[var]["direction"] = entry.get("direction", expectations[var].get("direction", "none"))
            strength = entry.get("strength")
            expectations[var]["strength"] = strength or expectations[var].get("strength", "none")
            expectations[var]["rank"] = entry.get("rank")
        return expectations

    if isinstance(signal.get("mutable_state_ranking"), list):
        for entry in signal.get("mutable_state_ranking", []):
            if not isinstance(entry, dict):
                continue
            var = str(entry.get("var") or "").strip()
            if not var:
                continue
            direction = str(entry.get("direction") or "none").strip().lower()
            rank = entry.get("rank")
            strength = "none"
            if direction != "none":
                if rank == 1:
                    strength = "primary"
                elif rank == 2:
                    strength = "secondary"
                else:
                    strength = "weak"
            expectations[var] = {"direction": direction, "strength": strength, "rank": rank}
        return expectations

    effects = signal.get("effects") if isinstance(signal.get("effects"), dict) else {}
    primary = {str(x) for x in signal.get("primary_targets", []) if str(x).strip()}
    secondary = {str(x) for x in signal.get("secondary_targets", []) if str(x).strip()}
    for var in mutable_state:
        direction = str(effects.get(var) or "none").strip().lower() if isinstance(effects, dict) else "none"
        if direction == "none":
            strength = "none"
        elif var in primary:
            strength = "primary"
        elif var in secondary:
            strength = "secondary"
        else:
            strength = "weak"
        expectations[var] = {"direction": direction, "strength": strength, "rank": None}
    return expectations


def _build_eps_map(
    *,
    mutable_state: list[str],
    clean_spec: Path,
    default_eps: float,
    small_eps: float,
    eps_override: dict[str, float] | None,
) -> dict[str, float]:
    bounds: dict[str, tuple[float, float]] = {}
    try:
        bounds = load_possible_numeric_bounds_from_clean_spec(clean_spec)
    except (OSError, ValueError, TypeError):
        bounds = {}

    eps_map: dict[str, float] = {}
    for var in mutable_state:
        if eps_override and var in eps_override:
            eps_map[var] = float(eps_override[var])
            continue
        lo, hi = bounds.get(var, (math.nan, math.nan))
        if math.isfinite(lo) and math.isfinite(hi):
            span = abs(float(hi) - float(lo))
            eps_map[var] = float(small_eps if span <= 1.0 else default_eps)
        else:
            eps_map[var] = float(default_eps)
    return eps_map


def _load_bounds(clean_spec: Path) -> dict[str, tuple[float, float]]:
    try:
        return load_possible_numeric_bounds_from_clean_spec(clean_spec)
    except (OSError, ValueError, TypeError):
        return {}


def _kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    if not x or not y or len(x) != len(y):
        return None
    pairs = [(float(a), float(b)) for a, b in zip(x, y, strict=False) if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_xy = 0

    for i in range(n - 1):
        x_i, y_i = pairs[i]
        for j in range(i + 1, n):
            x_j, y_j = pairs[j]
            dx = x_i - x_j
            dy = y_i - y_j
            if dx == 0 and dy == 0:
                ties_xy += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            else:
                prod = dx * dy
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return None
    return float((concordant - discordant) / denom)


def _extract_post_state(parsed: object, mutable_state: list[str]) -> dict[str, float] | None:
    if not isinstance(parsed, dict):
        return None
    if "A" in parsed and "B" in parsed:
        # Belief paired_profile prompts use A=baseline, B=perturbed.
        if isinstance(parsed.get("B"), dict):
            parsed = parsed.get("B")
        elif isinstance(parsed.get("A"), dict):
            parsed = parsed.get("A")
    out: dict[str, float] = {}
    for var in mutable_state:
        val = _safe_float(parsed.get(var))
        if val is None and var == "covid_perceived_danger_Y":
            val = _safe_float(parsed.get("covid_perceived_danger_T12"))
        if val is None and var == "covid_perceived_danger_T12":
            val = _safe_float(parsed.get("covid_perceived_danger_Y"))
        if val is not None:
            out[var] = val
    return out


def _extract_ref_maps(meta: dict[str, Any], mutable_state: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    ref_map_raw = meta.get("pe_ref_p") if isinstance(meta.get("pe_ref_p"), dict) else {}
    ref_std_raw = meta.get("pe_ref_p_std") if isinstance(meta.get("pe_ref_p_std"), dict) else {}
    ref_map: dict[str, float] = {}
    ref_std_map: dict[str, float] = {}
    for var in mutable_state:
        v = _safe_float(ref_map_raw.get(var))
        if v is None and var == "covid_perceived_danger_Y":
            v = _safe_float(ref_map_raw.get("covid_perceived_danger_T12"))
        if v is not None:
            ref_map[var] = float(v)
        v_std = _safe_float(ref_std_raw.get(var))
        if v_std is None and var == "covid_perceived_danger_Y":
            v_std = _safe_float(ref_std_raw.get("covid_perceived_danger_T12"))
        if v_std is not None:
            ref_std_map[var] = float(v_std)
    return ref_map, ref_std_map


def _derive_tiers_from_ref(
    *,
    ref_abs_by_var: dict[str, float],
    ref_present_vars: set[str],
    mutable_state: list[str],
    alpha_secondary: float,
    epsilon_ref: float,
    top_n_primary: int,
) -> dict[str, str]:
    ranked = sorted(
        ((var, float(ref_abs_by_var.get(var, float("nan")))) for var in mutable_state),
        key=lambda kv: (-kv[1] if np.isfinite(kv[1]) else float("inf"), kv[0]),
    )
    ranked = [(k, v) for k, v in ranked if np.isfinite(v)]
    if not ranked:
        return {var: ("unmapped" if var not in ref_present_vars else "none") for var in mutable_state}

    max_val = float(ranked[0][1])
    primary_vars = {k for k, _ in ranked[: max(1, int(top_n_primary))]}
    tiers: dict[str, str] = {}
    for var in mutable_state:
        if var not in ref_present_vars:
            tiers[var] = "unmapped"
            continue
        val = float(ref_abs_by_var.get(var, float("nan")))
        if not np.isfinite(val) or val < float(epsilon_ref):
            tiers[var] = "none"
        elif var in primary_vars:
            tiers[var] = "primary"
        elif max_val > 0.0 and val >= float(alpha_secondary) * max_val:
            tiers[var] = "secondary"
        else:
            tiers[var] = "weak"
    return tiers


def _cosine_alignment(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    x_use = x[mask]
    y_use = y[mask]
    nx = float(np.linalg.norm(x_use))
    ny = float(np.linalg.norm(y_use))
    if nx <= 0.0 or ny <= 0.0:
        return None
    return float(np.dot(x_use, y_use) / (nx * ny))


def _group_key(row: dict) -> tuple[str, str]:
    strategy = str(row.get("strategy") or "data_only")
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    signal_id = str(meta.get("signal_id") or "unknown")
    return strategy, signal_id


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSONL from run_llm_sampling.py")
    parser.add_argument("--spec", dest="spec_path", required=True, help="Belief update targets JSON")
    parser.add_argument("--outdir", dest="outdir", required=True, help="Output directory root")
    parser.add_argument("--clean-spec", default="preprocess/specs/5_clean_transformed_questions.json")
    parser.add_argument("--eps", type=float, default=0.05, help="Default epsilon for sign checks.")
    parser.add_argument("--eps-small", type=float, default=0.02, help="Epsilon for 0-1 scale variables.")
    parser.add_argument("--eps-map", default=None, help="Optional JSON dict or JSON file with per-var epsilons.")
    parser.add_argument("--moderation-quantile", type=float, default=0.25)
    parser.add_argument("--moderation-by-anchor", action="store_true")
    parser.add_argument("--epsilon-ref", type=float, default=0.03, help="Low-signal threshold in PE_ref_P_z units.")
    parser.add_argument(
        "--epsilon-move-mult",
        type=float,
        default=3.0,
        help="False-movement cutoff multiplier (epsilon_move = epsilon_ref * multiplier).",
    )
    parser.add_argument("--alpha-secondary", type=float, default=0.33)
    parser.add_argument("--top-n-primary", type=int, default=1)
    parser.add_argument("--denom", type=float, default=1.0, help="NMAE denominator for choice-style summaries.")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec_path).read_text(encoding="utf-8"))
    mutable_state = [str(x) for x in spec.get("mutable_state", []) if str(x).strip()]
    signals = spec.get("signals", []) if isinstance(spec, dict) else []

    signals_by_id: dict[str, dict] = {}
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        sid = str(signal.get("id") or "").strip()
        if sid:
            signals_by_id[sid] = signal

    eps_override: dict[str, float] | None = None
    if args.eps_map:
        raw = args.eps_map
        if Path(raw).exists():
            eps_override = json.loads(Path(raw).read_text(encoding="utf-8"))
        else:
            eps_override = json.loads(raw)
    eps_map = _build_eps_map(
        mutable_state=mutable_state,
        clean_spec=Path(args.clean_spec),
        default_eps=float(args.eps),
        small_eps=float(args.eps_small),
        eps_override=eps_override if isinstance(eps_override, dict) else None,
    )
    bounds_map = _load_bounds(Path(args.clean_spec))
    span_map: dict[str, float] = {}
    for var in mutable_state:
        lo, hi = bounds_map.get(var, (math.nan, math.nan))
        if math.isfinite(lo) and math.isfinite(hi):
            span = abs(float(hi) - float(lo))
            if math.isfinite(span) and span > 0.0:
                span_map[var] = float(span)

    rows = _read_jsonl(Path(args.in_path))

    outdir = Path(args.outdir)
    _ensure_dir(outdir)
    strategies_in_run = sorted({str(r.get("strategy") or "data_only") for r in rows})
    _write_json(
        outdir / "run_config.json",
        {
            "inputs": {
                "samples": str(Path(args.in_path)),
                "spec": str(Path(args.spec_path)),
            },
            "strategies": strategies_in_run,
            "eps": eps_map,
            "moderation_quantile": float(args.moderation_quantile),
            "moderation_by_anchor": bool(args.moderation_by_anchor),
            "epsilon_ref": float(args.epsilon_ref),
            "epsilon_move_mult": float(args.epsilon_move_mult),
            "alpha_secondary": float(args.alpha_secondary),
            "top_n_primary": int(args.top_n_primary),
            "denom": float(args.denom),
            "timestamp_utc": utc_timestamp_iso(),
        },
    )
    spec_copy_path = outdir / "belief_update_targets.json"
    spec_copy_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    grouped_rows: dict[tuple[str, str], list[dict]] = {}
    grouped_conversations: dict[tuple[str, str], list[dict]] = {}
    row_records: list[dict] = []
    pe_long_records: list[dict[str, Any]] = []
    epsilon_ref = float(args.epsilon_ref)
    epsilon_move = float(args.epsilon_move_mult) * epsilon_ref

    for row in rows:
        strategy, signal_id = _group_key(row)
        grouped_rows.setdefault((strategy, signal_id), []).append(row)

        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        base_id = meta.get("base_id") or row.get("id")
        baseline_state = meta.get("baseline_state") if isinstance(meta, dict) else None
        if not isinstance(baseline_state, dict):
            baseline_state = {}

        signal = signals_by_id.get(signal_id, {})
        expectations = _build_expectations(signal, mutable_state)
        has_manual_expectations = _signal_has_manual_expectations(signal)
        ref_map, ref_std_map = _extract_ref_maps(meta, mutable_state)
        ref_abs_map = {k: abs(float(v)) for k, v in ref_std_map.items() if np.isfinite(float(v))}
        tier_by_var = (
            meta.get("pe_ref_p_tier_by_var")
            if isinstance(meta.get("pe_ref_p_tier_by_var"), dict)
            else _derive_tiers_from_ref(
                ref_abs_by_var=ref_abs_map,
                ref_present_vars=set(ref_std_map.keys()),
                mutable_state=mutable_state,
                alpha_secondary=float(args.alpha_secondary),
                epsilon_ref=float(args.epsilon_ref),
                top_n_primary=int(args.top_n_primary),
            )
        )
        delta_z_meta = _safe_float(meta.get("pe_ref_p_delta_z"))

        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        sample_posts: list[dict[str, float]] = []
        parsed_count = 0
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            parsed = sample.get("parsed")
            post_state = _extract_post_state(parsed, mutable_state)
            if post_state is not None:
                parsed_count += 1
                sample_posts.append(post_state)

            prompt_text = row.get("_prompt_text") or row.get("prompt_text") or row.get("prompt") or ""
            raw_text = sample.get("raw_text_original") or sample.get("raw_text") or ""
            grouped_conversations.setdefault((strategy, signal_id), []).append(
                {
                    "id": base_id,
                    "strategy": strategy,
                    "signal_id": signal_id,
                    "prompt_text": prompt_text,
                    "response_text": raw_text,
                    "parsed_ok": bool(post_state),
                    "error": sample.get("error"),
                }
            )

        post_means: dict[str, float | None] = {}
        post_sds: dict[str, float | None] = {}
        for var in mutable_state:
            vals: list[float] = []
            for post in sample_posts:
                if var in post:
                    vals.append(post[var])
            baseline_val = _safe_float(baseline_state.get(var))
            if not vals and baseline_val is not None:
                vals = [baseline_val]
            if vals:
                post_means[var] = float(np.mean(vals))
                post_sds[var] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                post_means[var] = None
                post_sds[var] = None

        metrics: dict[str, Any] = {
            "id": base_id,
            "strategy": strategy,
            "signal_id": signal_id,
            "country": meta.get("country"),
            "row_index": meta.get("row_index"),
            "n_samples": len(samples),
            "n_parsed": parsed_count,
            "epsilon_ref": epsilon_ref,
            "epsilon_move": epsilon_move,
        }
        if isinstance(meta.get("always_on_moderators"), dict):
            metrics.update(meta.get("always_on_moderators"))

        per_var_non_none: list[bool] = []
        per_var_none: list[bool] = []
        per_var_primary: list[bool] = []
        control_passes: list[bool] = []
        strength_scores: list[int] = []
        strength_abs_deltas: list[float] = []
        abs_deltas: list[float] = []
        abs_deltas_norm: list[float] = []

        tier_buckets: dict[str, list[float]] = {"primary": [], "secondary": [], "weak": [], "none": []}
        tier_buckets_norm: dict[str, list[float]] = {"primary": [], "secondary": [], "weak": [], "none": []}
        rank_ref_abs: list[float] = []
        rank_llm_abs: list[float] = []
        ref_vec_std: list[float] = []
        llm_vec_std: list[float] = []
        ref_vec_raw: list[float] = []
        llm_vec_raw: list[float] = []

        for var in mutable_state:
            baseline_val = _safe_float(baseline_state.get(var))
            post_val = post_means.get(var)
            delta = None
            if baseline_val is not None and post_val is not None:
                delta = float(post_val) - float(baseline_val)
            eps = float(eps_map.get(var, float(args.eps)))
            ref_val = _safe_float(ref_map.get(var))
            ref_std_val = _safe_float(ref_std_map.get(var))
            if ref_std_val is None and ref_val is not None:
                dz_meta = _safe_float(meta.get("pe_ref_p_delta_z"))
                if dz_meta is not None and dz_meta != 0.0:
                    ref_std_val = float(ref_val) / float(dz_meta)

            expected = expectations.get(var, {}).get("direction", "none")
            tier_val = tier_by_var.get(var)
            if tier_val is not None:
                strength = str(tier_val).strip().lower()
            elif has_manual_expectations:
                strength = str(expectations.get(var, {}).get("strength", "none")).strip().lower()
            else:
                strength = "unmapped"
            sign_ok: bool | None = None
            abs_d = None
            delta_norm = None
            abs_d_norm = None
            pe_llm_std = None
            sign_ref = None
            sign_llm = None
            sign_agree = None
            low_signal = None
            false_positive_move = None
            if delta is not None:
                abs_d = abs(delta)
                abs_deltas.append(abs_d)
                tier_buckets.setdefault(str(strength), []).append(abs_d)
                span = span_map.get(var)
                if span is not None and span > 0.0:
                    delta_norm = float(delta) / float(span)
                    abs_d_norm = abs(float(delta_norm))
                    abs_deltas_norm.append(abs_d_norm)
                    tier_buckets_norm.setdefault(str(strength), []).append(abs_d_norm)
                if delta_z_meta is not None and delta_z_meta != 0.0:
                    pe_llm_std = float(delta) / float(delta_z_meta)
                if ref_std_val is not None and np.isfinite(ref_std_val):
                    sign_ref = float(np.sign(float(ref_std_val)))
                    sign_llm = float(np.sign(float(pe_llm_std))) if pe_llm_std is not None else None
                    low_signal = bool(abs(float(ref_std_val)) <= epsilon_ref)
                    informative = bool(abs(float(ref_std_val)) > epsilon_ref)
                    if informative and sign_llm is not None:
                        sign_agree = bool(sign_ref == sign_llm)
                        sign_ok = sign_agree
                    elif low_signal and pe_llm_std is not None:
                        sign_ok = bool(abs(float(pe_llm_std)) <= epsilon_move)
                    false_positive_move = (
                        bool(abs(float(pe_llm_std)) > epsilon_move) if low_signal and pe_llm_std is not None else None
                    )
                elif has_manual_expectations:
                    # Legacy fallback only when expectations are explicitly provided in the spec.
                    sign_ok = _direction_pass(str(expected), float(delta), float(eps))

            metrics[f"pre_{var}"] = baseline_val
            metrics[f"post_{var}"] = post_val
            metrics[f"delta_{var}"] = delta
            metrics[f"abs_delta_{var}"] = abs_d
            metrics[f"delta_norm_{var}"] = delta_norm
            metrics[f"abs_delta_norm_{var}"] = abs_d_norm
            metrics[f"pe_ref_p_{var}"] = ref_val
            metrics[f"pe_ref_p_std_{var}"] = ref_std_val
            metrics[f"pe_llm_p_std_{var}"] = pe_llm_std
            metrics[f"expected_{var}"] = expected
            metrics[f"strength_{var}"] = strength
            metrics[f"sign_ok_{var}"] = sign_ok
            metrics[f"sign_ref_{var}"] = sign_ref
            metrics[f"sign_llm_{var}"] = sign_llm
            metrics[f"sign_agree_{var}"] = sign_agree
            metrics[f"low_signal_{var}"] = low_signal
            metrics[f"false_positive_move_{var}"] = false_positive_move

            if strength in {"none"}:
                if sign_ok is not None:
                    per_var_none.append(bool(sign_ok))
            elif strength in {"primary", "secondary", "weak"}:
                if sign_ok is not None:
                    per_var_non_none.append(bool(sign_ok))

            if strength == "primary":
                if sign_ok is not None:
                    per_var_primary.append(bool(sign_ok))

            if strength == "none" and pe_llm_std is not None and ref_std_val is not None:
                control_passes.append(abs(float(pe_llm_std)) <= epsilon_move)

            if strength in {"primary", "secondary", "weak", "none"} and abs_d is not None:
                strength_scores.append(_strength_score(strength))
                strength_abs_deltas.append(float(abs_d))

            if delta is not None and ref_val is not None and np.isfinite(float(ref_val)):
                ref_vec_raw.append(float(ref_val))
                llm_vec_raw.append(float(delta))
            if pe_llm_std is not None and ref_std_val is not None and np.isfinite(float(ref_std_val)):
                ref_vec_std.append(float(ref_std_val))
                llm_vec_std.append(float(pe_llm_std))
                rank_ref_abs.append(abs(float(ref_std_val)))
                rank_llm_abs.append(abs(float(pe_llm_std)))

            if delta is not None and ref_val is not None:
                pe_long_records.append(
                    {
                        "id": base_id,
                        "strategy": strategy,
                        "signal_id": signal_id,
                        "target": var,
                        "pe_ref_p": float(ref_val),
                        "pe_llm_p": float(delta),
                        "pe_ref_p_std": float(ref_std_val) if ref_std_val is not None else float("nan"),
                        "pe_llm_p_std": float(pe_llm_std) if pe_llm_std is not None else float("nan"),
                        "delta_z": _safe_float(meta.get("pe_ref_p_delta_z")),
                        "delta_x": _safe_float(meta.get("pe_ref_p_delta_x")),
                        "da": (float(sign_agree) if sign_agree is not None else float("nan")),
                        "da_status": "scored" if sign_agree is not None else "no_signal",
                        "sign_ref": sign_ref,
                        "sign_llm": sign_llm,
                        "sign_agree": (float(sign_agree) if sign_agree is not None else float("nan")),
                        "low_signal": float(low_signal) if low_signal is not None else float("nan"),
                        "low_signal_abs_pe_llm_std": (
                            abs(float(pe_llm_std)) if (low_signal and pe_llm_std is not None) else float("nan")
                        ),
                        "false_positive_move": (
                            float(false_positive_move) if false_positive_move is not None else float("nan")
                        ),
                        "epsilon_ref": float(epsilon_ref),
                        "epsilon_move": float(epsilon_move),
                    }
                )

        metrics["sign_pass_rate_primary"] = float(np.mean(per_var_primary)) if per_var_primary else None
        metrics["sign_pass_rate_all_non_none"] = float(np.mean(per_var_non_none)) if per_var_non_none else None
        metrics["none_stability_rate"] = float(np.mean(per_var_none)) if per_var_none else None
        metrics["control_stability_rate"] = float(np.mean(control_passes)) if control_passes else None

        kendall_tau = _kendall_tau_b(strength_scores, strength_abs_deltas) if strength_abs_deltas else None
        metrics["kendall_strength_alignment"] = kendall_tau
        metrics["kendall_rank_abs_ref"] = _kendall_tau_b(rank_ref_abs, rank_llm_abs) if len(rank_ref_abs) >= 2 else None
        if len(rank_ref_abs) >= 1:
            metrics["top1_match_ref"] = bool(int(np.argmax(rank_ref_abs)) == int(np.argmax(rank_llm_abs)))
        else:
            metrics["top1_match_ref"] = None
        if ref_vec_std and llm_vec_std:
            metrics["vector_cosine_std"] = _cosine_alignment(np.asarray(ref_vec_std), np.asarray(llm_vec_std))
            metrics["vector_mae_std"] = float(
                np.mean(np.abs(np.asarray(llm_vec_std, dtype=float) - np.asarray(ref_vec_std, dtype=float)))
            )
        else:
            metrics["vector_cosine_std"] = None
            metrics["vector_mae_std"] = None
        if ref_vec_raw and llm_vec_raw:
            metrics["vector_cosine"] = _cosine_alignment(np.asarray(ref_vec_raw), np.asarray(llm_vec_raw))
            metrics["vector_mae"] = float(
                np.mean(np.abs(np.asarray(llm_vec_raw, dtype=float) - np.asarray(ref_vec_raw, dtype=float)))
            )
        else:
            metrics["vector_cosine"] = None
            metrics["vector_mae"] = None

        try:
            primary_mean = float(np.mean(tier_buckets["primary"])) if tier_buckets["primary"] else 0.0
            secondary_mean = float(np.mean(tier_buckets["secondary"])) if tier_buckets["secondary"] else 0.0
            weak_mean = float(np.mean(tier_buckets["weak"])) if tier_buckets["weak"] else 0.0
            none_mean = float(np.mean(tier_buckets["none"])) if tier_buckets["none"] else 0.0
            metrics["tier_monotonicity"] = bool(primary_mean >= secondary_mean >= weak_mean >= none_mean)
            metrics["tier_mean_primary"] = primary_mean
            metrics["tier_mean_secondary"] = secondary_mean
            metrics["tier_mean_weak"] = weak_mean
            metrics["tier_mean_none"] = none_mean
            primary_mean_norm = float(np.mean(tier_buckets_norm["primary"])) if tier_buckets_norm["primary"] else 0.0
            secondary_mean_norm = (
                float(np.mean(tier_buckets_norm["secondary"])) if tier_buckets_norm["secondary"] else 0.0
            )
            weak_mean_norm = float(np.mean(tier_buckets_norm["weak"])) if tier_buckets_norm["weak"] else 0.0
            none_mean_norm = float(np.mean(tier_buckets_norm["none"])) if tier_buckets_norm["none"] else 0.0
            metrics["tier_mean_primary_norm"] = primary_mean_norm
            metrics["tier_mean_secondary_norm"] = secondary_mean_norm
            metrics["tier_mean_weak_norm"] = weak_mean_norm
            metrics["tier_mean_none_norm"] = none_mean_norm
        except (TypeError, ValueError, ZeroDivisionError):
            metrics["tier_monotonicity"] = None

        row_records.append(metrics)

    pe_long_df = pd.DataFrame(pe_long_records)
    if not pe_long_df.empty:
        try:
            pe_long_df.to_parquet(outdir / "pe_alignment_rows.parquet", index=False)
        except (OSError, ValueError):
            pe_long_df.to_csv(outdir / "pe_alignment_rows.csv", index=False)

    for (strategy, signal_id), _ in grouped_rows.items():
        signal_dir = outdir / strategy / "signals" / signal_id
        _ensure_dir(signal_dir)

        convs = grouped_conversations.get((strategy, signal_id), [])
        _write_jsonl(signal_dir / "conversations.jsonl", convs)

        rows_for_group = [r for r in row_records if r.get("strategy") == strategy and r.get("signal_id") == signal_id]
        _write_jsonl(signal_dir / "rows.jsonl", rows_for_group)

        if rows_for_group:
            wrote_metrics = False
            try:
                import polars as pl

                # Some sparse bool fields are all-null in early rows and become
                # bool later; scan full rows to avoid schema inference mismatch.
                pl.DataFrame(rows_for_group, infer_schema_length=None).write_parquet(signal_dir / "metrics.parquet")
                wrote_metrics = True
            except Exception:
                wrote_metrics = False

            if not wrote_metrics:
                frame = pd.DataFrame(rows_for_group)
                try:
                    frame.to_parquet(signal_dir / "metrics.parquet", index=False)
                except (OSError, ValueError):
                    frame.to_csv(signal_dir / "metrics.csv", index=False)
        else:
            (signal_dir / "metrics.parquet").write_bytes(b"")

        df = pd.DataFrame(rows_for_group)
        summary = {
            "strategy": strategy,
            "signal_id": signal_id,
            "n_rows": int(df.shape[0]),
            "n_primary_mover_rows": None,
            "sign_pass_rate_primary_movers": None,
            "sign_pass_rate_all_non_none_movers": None,
            "primary_activation_rate": None,
            "primary_saturation_rate": None,
        }
        for key in [
            "sign_pass_rate_primary",
            "sign_pass_rate_all_non_none",
            "none_stability_rate",
            "control_stability_rate",
            "kendall_strength_alignment",
            "kendall_rank_abs_ref",
            "top1_match_ref",
            "vector_cosine_std",
            "vector_mae_std",
            "vector_cosine",
            "vector_mae",
            "tier_monotonicity",
        ]:
            if key in df.columns and not df.empty:
                summary[key] = float(df[key].mean())
            else:
                summary[key] = None

        if not df.empty:

            def _tier_for(var: str, frame: pd.DataFrame) -> str:
                col = f"strength_{var}"
                if col not in frame.columns:
                    return "none"
                vals = frame[col].dropna().astype(str)
                if vals.empty:
                    return "none"
                return str(vals.iloc[0]).strip().lower()

            primary_targets = [var for var in mutable_state if _tier_for(var, df) == "primary"]
            expected_movers = [var for var in mutable_state if _tier_for(var, df) in {"primary", "secondary", "weak"}]

            def _eps_for(var: str) -> float:
                return float(eps_map.get(var, float(args.eps)))

            def _any_mover_mask(vars_list: list[str], df: pd.DataFrame) -> pd.Series | None:
                mask: pd.Series | None = None
                for var in vars_list:
                    col = f"abs_delta_{var}"
                    if col not in df.columns:
                        continue
                    vals = pd.to_numeric(df[col], errors="coerce")
                    mover = vals > _eps_for(var)
                    mask = mover if mask is None else (mask | mover)
                return mask

            def _mean_sign_row(vars_list: list[str], df: pd.DataFrame) -> pd.Series | None:
                cols = [f"sign_ok_{var}" for var in vars_list if f"sign_ok_{var}" in df.columns]
                if not cols:
                    return None
                return df[cols].astype(float).mean(axis=1)

            primary_mover = _any_mover_mask(primary_targets, df)
            expected_mover = _any_mover_mask(expected_movers, df)
            primary_sign = _mean_sign_row(primary_targets, df)
            expected_sign = _mean_sign_row(expected_movers, df)

            if primary_mover is not None:
                n_primary_mover_rows = float(primary_mover.sum())
                summary["n_primary_mover_rows"] = n_primary_mover_rows
                summary["primary_activation_rate"] = float(n_primary_mover_rows / df.shape[0]) if df.shape[0] else None
                if primary_sign is not None and n_primary_mover_rows > 0:
                    summary["sign_pass_rate_primary_movers"] = float(primary_sign[primary_mover].mean())

            if expected_mover is not None and expected_sign is not None:
                if expected_mover.any():
                    summary["sign_pass_rate_all_non_none_movers"] = float(expected_sign[expected_mover].mean())

            sat_masks: list[pd.Series] = []
            for var in primary_targets:
                pre_col = f"pre_{var}"
                ref_std_col = f"pe_ref_p_std_{var}"
                if pre_col not in df.columns or ref_std_col not in df.columns:
                    continue
                bounds = bounds_map.get(var)
                if not bounds:
                    continue
                lo, hi = bounds
                eps = _eps_for(var)
                ref_std = pd.to_numeric(df[ref_std_col], errors="coerce")
                pre = pd.to_numeric(df[pre_col], errors="coerce")
                sat_masks.append(
                    ((ref_std > 0) & (pre >= (float(hi) - eps))) | ((ref_std < 0) & (pre <= (float(lo) + eps)))
                )
            if sat_masks:
                sat_any = sat_masks[0]
                for mask in sat_masks[1:]:
                    sat_any = sat_any | mask
                summary["primary_saturation_rate"] = float(sat_any.mean())

        if not pe_long_df.empty:
            long_sub = pe_long_df[
                (pe_long_df["strategy"].astype(str) == str(strategy))
                & (pe_long_df["signal_id"].astype(str) == str(signal_id))
            ]
            choice_metrics = compute_choice_like_summary(
                long_df=long_sub,
                epsilon_ref=float(args.epsilon_ref),
                epsilon_move_mult=float(args.epsilon_move_mult),
                denom=float(args.denom),
            )
            summary.update(choice_metrics)
            if not long_sub.empty:
                try:
                    long_sub.to_parquet(signal_dir / "pe_alignment_rows.parquet", index=False)
                except (OSError, ValueError):
                    long_sub.to_csv(signal_dir / "pe_alignment_rows.csv", index=False)
        _write_json(signal_dir / "summary.json", summary)

    for strategy in {str(r.get("strategy")) for r in row_records}:
        strategy_records = [r for r in row_records if r.get("strategy") == strategy]
        by_signal: dict[str, list[dict]] = {}
        for r in strategy_records:
            by_signal.setdefault(str(r.get("signal_id") or "unknown"), []).append(r)

        for signal_id, sig_rows in by_signal.items():
            signal = signals_by_id.get(signal_id, {})
            moderation_results: list[dict] = []
            moderation_tests = signal.get("moderation_tests", []) if isinstance(signal, dict) else []
            quant = float(args.moderation_quantile)
            for test in moderation_tests:
                if not isinstance(test, dict):
                    continue
                moderator = str(test.get("moderator") or "").strip()
                if not moderator:
                    continue
                targets = [str(x) for x in test.get("targets", []) if str(x).strip()]
                expected = str(test.get("expected") or "").strip()

                df = pd.DataFrame(sig_rows)
                if df.empty or moderator not in df.columns:
                    moderation_results.append(
                        {
                            "moderator": moderator,
                            "targets": targets,
                            "expected": expected,
                            "moderation_pass": None,
                            "moderation_gap": None,
                            "reason": "missing moderator column",
                        }
                    )
                    continue

                if args.moderation_by_anchor and "id" in df.columns:
                    anchor = df["id"].astype(str).str.split("::").str[0]
                    df = df.assign(anchor_id=anchor)
                    group_cols = ["country", "anchor_id"] if "country" in df.columns else ["anchor_id"]
                else:
                    group_cols = ["country"] if "country" in df.columns else []

                high_idx: list[int] = []
                low_idx: list[int] = []
                for _, sub in df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]:
                    # `moderator` columns can be boolean (e.g., sex_female). Pandas/numpy
                    # quantile on boolean can error under newer numpy implementations.
                    vals = pd.to_numeric(sub[moderator], errors="coerce").astype(float).dropna()
                    if vals.empty:
                        continue
                    q_low = vals.quantile(quant)
                    q_high = vals.quantile(1.0 - quant)
                    low_idx.extend(vals.index[vals <= q_low].tolist())
                    high_idx.extend(vals.index[vals >= q_high].tolist())

                high = df.loc[high_idx]
                low = df.loc[low_idx]

                if not targets:
                    targets = list(mutable_state)

                did_llm_vals: list[float] = []
                did_ref_vals: list[float] = []
                direction_match_vals: list[float] = []
                restraint_vals: list[float] = []
                for var in targets:
                    llm_col = f"pe_llm_p_std_{var}"
                    ref_col = f"pe_ref_p_std_{var}"
                    if llm_col not in df.columns or ref_col not in df.columns:
                        continue
                    high_llm = pd.to_numeric(high[llm_col], errors="coerce").dropna()
                    low_llm = pd.to_numeric(low[llm_col], errors="coerce").dropna()
                    high_ref = pd.to_numeric(high[ref_col], errors="coerce").dropna()
                    low_ref = pd.to_numeric(low[ref_col], errors="coerce").dropna()
                    if high_llm.empty or low_llm.empty or high_ref.empty or low_ref.empty:
                        continue
                    did_llm = float(np.median(high_llm) - np.median(low_llm))
                    did_ref = float(np.median(high_ref) - np.median(low_ref))
                    did_llm_vals.append(did_llm)
                    did_ref_vals.append(did_ref)
                    if abs(did_ref) > epsilon_ref:
                        direction_match_vals.append(float(np.sign(did_llm) == np.sign(did_ref)))
                    else:
                        restraint_vals.append(float(abs(did_llm) <= epsilon_move))

                pass_high = float(np.mean(did_llm_vals)) if did_llm_vals else None
                pass_low = float(np.mean(did_ref_vals)) if did_ref_vals else None
                gap = pass_high - pass_low if pass_high is not None and pass_low is not None else None
                direction_match_rate = float(np.mean(direction_match_vals)) if direction_match_vals else None
                restraint_rate = float(np.mean(restraint_vals)) if restraint_vals else None
                did_mae = (
                    float(np.mean(np.abs(np.asarray(did_llm_vals) - np.asarray(did_ref_vals))))
                    if did_llm_vals and did_ref_vals
                    else None
                )
                moderation_pass = None
                if expected == "high >= low" and pass_high is not None and pass_low is not None:
                    moderation_pass = pass_high >= pass_low
                elif expected == "high <= low" and pass_high is not None and pass_low is not None:
                    moderation_pass = pass_high <= pass_low
                elif direction_match_rate is not None:
                    moderation_pass = direction_match_rate >= 0.5
                elif restraint_rate is not None:
                    moderation_pass = restraint_rate >= 0.5

                moderation_results.append(
                    {
                        "moderator": moderator,
                        "targets": targets,
                        "expected": expected,
                        "did_llm_high_minus_low": pass_high,
                        "did_ref_high_minus_low": pass_low,
                        "moderation_direction_match_rate": direction_match_rate,
                        "moderation_restraint_rate": restraint_rate,
                        "moderation_did_mae": did_mae,
                        "moderation_pass": moderation_pass,
                        "moderation_gap": gap,
                    }
                )

            signal_dir = outdir / strategy / "signals" / signal_id
            _ensure_dir(signal_dir)
            _write_json(
                signal_dir / "paired_checks.json",
                {
                    "signal_id": signal_id,
                    "strategy": strategy,
                    "moderation_tests": moderation_results,
                },
            )

    df_all = pd.DataFrame(row_records)
    strategy_summary = []
    if not df_all.empty:
        for strategy, sub in df_all.groupby("strategy", dropna=False):
            summary: dict[str, Any] = {
                "strategy": strategy,
                "n_rows": int(sub.shape[0]),
            }
            for key in [
                "sign_pass_rate_primary",
                "sign_pass_rate_all_non_none",
                "none_stability_rate",
                "control_stability_rate",
                "kendall_strength_alignment",
                "kendall_rank_abs_ref",
                "top1_match_ref",
                "vector_cosine_std",
                "vector_mae_std",
                "vector_cosine",
                "vector_mae",
                "tier_monotonicity",
            ]:
                if key in sub.columns:
                    vals = pd.to_numeric(sub[key], errors="coerce")
                    summary[key] = float(vals.mean()) if vals.notna().any() else None
                else:
                    summary[key] = None
            if not pe_long_df.empty:
                long_sub = pe_long_df[pe_long_df["strategy"].astype(str) == str(strategy)]
                summary.update(
                    compute_choice_like_summary(
                        long_df=long_sub,
                        epsilon_ref=float(args.epsilon_ref),
                        epsilon_move_mult=float(args.epsilon_move_mult),
                        denom=float(args.denom),
                    )
                )
            strategy_summary.append(summary)
            _write_json(outdir / strategy / "strategy_summary.json", summary)

    _write_json(outdir / "overall_summary.json", {"strategies": strategy_summary})


if __name__ == "__main__":
    main()
