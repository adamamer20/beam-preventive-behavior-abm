"""Compute PE_ref_P tables for belief-update microvalidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beam_abm.cli import parser_compat as cli
from beam_abm.evaluation.belief.reference_model_loading import default_ref_state_models
from beam_abm.evaluation.belief.reference_pe_table import compute_pe_ref_p_table
from beam_abm.evaluation.utils.jsonl import read_frame as _read_rows
from beam_abm.evaluation.utils.jsonl import read_jsonl as _read_jsonl


def _parse_ref_map(spec: dict[str, Any]) -> dict[str, str]:
    raw = spec.get("ref_state_models")
    if isinstance(raw, dict):
        out = {str(k): str(v) for k, v in raw.items() if str(k).strip() and str(v).strip()}
        if out:
            return out
    return default_ref_state_models()


def run_belief_pe_ref_p(argv: list[str] | None = None) -> None:
    parser = cli.ArgumentParser()
    parser.add_argument("--data", required=True, help="Baseline profiles (csv/parquet/jsonl).")
    parser.add_argument("--spec", required=True, help="Belief-update spec JSON.")
    parser.add_argument("--out", required=True, help="Output PE_ref_P rows (parquet/csv).")
    parser.add_argument("--meta-out", required=True, help="Output signal metadata JSON.")
    parser.add_argument("--prompts", default=None, help="Optional prompts JSONL to filter ids/signals.")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--country-col", default="country")
    parser.add_argument("--modeling-dir", default="empirical/output/modeling")
    parser.add_argument("--alpha-secondary", type=float, default=0.33)
    parser.add_argument("--epsilon-ref", type=float, default=0.03)
    parser.add_argument("--top-n-primary", type=int, default=1)
    parser.add_argument("--strict-missing-models", action="store_true")
    args = parser.parse_args(argv)

    data_df = _read_rows(Path(args.data), skip_invalid_jsonl=True)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    signals = spec.get("signals") if isinstance(spec.get("signals"), list) else []
    mutable_state = [str(x) for x in (spec.get("mutable_state") or []) if str(x).strip()]

    if args.prompts:
        prompt_rows = _read_jsonl(Path(args.prompts))
        keep_ids: set[str] = set()
        keep_signals: set[str] = set()
        for r in prompt_rows:
            rid = str(r.get("id") or "").strip()
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            sid = str(meta.get("signal_id") or "").strip()
            if rid:
                keep_ids.add(rid)
            if sid:
                keep_signals.add(sid)

        if keep_ids and args.id_col in data_df.columns:
            data_df = data_df[data_df[args.id_col].astype(str).isin(keep_ids)].copy()
        if keep_signals:
            signals = [s for s in signals if isinstance(s, dict) and str(s.get("id") or "").strip() in keep_signals]

    pe_df, signal_meta, model_status = compute_pe_ref_p_table(
        profiles_df=data_df,
        source_df=data_df,
        signals=[s for s in signals if isinstance(s, dict)],
        mutable_state=mutable_state,
        ref_state_models=_parse_ref_map(spec),
        modeling_dir=Path(args.modeling_dir),
        id_col=str(args.id_col),
        country_col=str(args.country_col),
        alpha_secondary=float(args.alpha_secondary),
        epsilon_ref=float(args.epsilon_ref),
        top_n_primary=int(args.top_n_primary),
        strict_missing_models=bool(args.strict_missing_models),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        pe_df.to_csv(out_path, index=False)
    else:
        pe_df.to_parquet(out_path, index=False)

    meta_payload = {
        "signal_meta": signal_meta,
        "model_status": model_status,
        "config": {
            "alpha_secondary": float(args.alpha_secondary),
            "epsilon_ref": float(args.epsilon_ref),
            "top_n_primary": int(args.top_n_primary),
            "id_col": str(args.id_col),
            "country_col": str(args.country_col),
        },
    }
    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


