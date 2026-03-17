#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# API credentials are expected from your shell environment.
# Example:
#   export OPENAI_API_KEY=...
#   export OPENAI_BASE_URL=https://api.groq.com

RUN_TAG="${RUN_TAG:-belief-unpert-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-evaluation/output/belief_update_validation/unperturbed_altacct}"
DATA="${DATA:-empirical/output/anchors/eval/eval_profiles.parquet}"
SPEC="${SPEC:-evaluation/specs/belief_update_targets.json}"
MODEL_PLAN="${MODEL_PLAN:-empirical/output/modeling/model_plan.json}"

MODELS="${MODELS:-openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3-32b}"
PROMPT_FAMILIES="${PROMPT_FAMILIES:-data_only,first_person,third_person,multi_expert}"
BACKEND="${BACKEND:-azure_openai}"

PER_PROFILE="${PER_PROFILE:-100}"
SEED="${SEED:-0}"
K="${K:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-24}"
MODEL_OPTIONS="${MODEL_OPTIONS:-{\"reasoning\":{\"effort\":\"medium\"}}}"
ONLY_MISSING="${ONLY_MISSING:-1}"

RAN_ANY=0

normalize_model_slug() {
  local model="$1"
  MODEL_RAW="$model" uv run python - <<'PY'
import os
from beam_abm.evaluation.choice.canonicalize import normalize_model_slug
print(normalize_model_slug(os.environ["MODEL_RAW"], reasoning_effort=None))
PY
}

model_options_for_model() {
  local model="$1"
  MODEL_NAME="$model" MODEL_OPTIONS_RAW="$MODEL_OPTIONS" uv run python - <<'PY'
import json
import os

raw = os.environ.get("MODEL_OPTIONS_RAW", "").strip()
model = os.environ["MODEL_NAME"]

if raw:
  payload = json.loads(raw)
else:
  payload = {}

if not isinstance(payload, dict):
  raise SystemExit("MODEL_OPTIONS must be a JSON object")

selected = payload
if model in payload and isinstance(payload[model], dict):
  selected = payload[model]

if not isinstance(selected, dict):
  raise SystemExit("Selected model options must be a JSON object")

if model.startswith("qwen/"):
  selected = {k: v for k, v in selected.items() if k != "reasoning"}

print(json.dumps(selected, separators=(",", ":")))
PY
}

rebuild_canonical_and_metrics() {
  uv run python evaluation/scripts/belief_update_validation/rebuild_belief_update_canonical.py \
    --output-root "$OUT"

  uv run python evaluation/scripts/belief_update_validation/rebuild_unperturbed_metrics.py \
    --output-root "$OUT" \
    --spec "$SPEC"
}

echo "[1/4] Syncing canonical + metrics from any existing unperturbed runs"
mkdir -p "$OUT"
rebuild_canonical_and_metrics

echo "[2/4] Detecting missing model/strategy cells"
missing_jobs="$(
  OUT="$OUT" MODELS="$MODELS" PROMPT_FAMILIES="$PROMPT_FAMILIES" ONLY_MISSING="$ONLY_MISSING" uv run python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from beam_abm.evaluation.choice.canonicalize import normalize_model_slug

out = Path(os.environ["OUT"])
models = [x.strip() for x in os.environ["MODELS"].split(",") if x.strip()]
families = [x.strip() for x in os.environ["PROMPT_FAMILIES"].split(",") if x.strip()]
only_missing = str(os.environ.get("ONLY_MISSING", "1")).strip() == "1"

for model in models:
    slug = normalize_model_slug(model, reasoning_effort=None)
    for family in families:
        marker = out / "canonical" / slug / family / "strategy_summary.json"
        if (not only_missing) or (not marker.exists()):
            print(f"{model}\t{slug}\t{family}")
PY
)"

if [[ -z "$missing_jobs" ]]; then
  echo "[2/4] No missing cells detected; nothing to run."
  exit 0
fi

echo "[3/4] Sampling missing unperturbed belief cells"
while IFS=$'\t' read -r model model_slug family; do
  [[ -z "${model:-}" ]] && continue
  run_id="${RUN_TAG}-${model_slug}__${family}"
  run_root="$OUT/_runs/$run_id"
  batch_dir="$run_root/_batch/$model_slug"
  prompts_path="$run_root/prompts.jsonl"
  samples_path="$batch_dir/samples.jsonl"

  mkdir -p "$batch_dir"

  echo "  -> model=$model family=$family run_id=$run_id"
  model_options_json="$(model_options_for_model "$model")"

  uv run python evaluation/scripts/choice_validation/phase0_prompt_tournament/generate_prompts.py \
    --mv-task belief_update \
    --run-type unperturbed \
    --data "$DATA" \
    --spec "$SPEC" \
    --out "$prompts_path" \
    --prompt-families "$family" \
    --profile-col anchor_id \
    --per-profile "$PER_PROFILE" \
    --seed "$SEED" \
    --anchor-col anchor_id \
    --country-col country \
    --id-col row_id \
    --model-plan "$MODEL_PLAN"

  uv run python -m evaluation.scripts.cli.run_llm_sampling \
    --in "$prompts_path" \
    --out "$samples_path" \
    --backend "$BACKEND" \
    --model "$model" \
    --k "$K" \
    --two-pass \
    --max-concurrency "$MAX_CONCURRENCY" \
    --response-model beam_abm.llm.schemas.predictions:BeliefUpdatePrediction \
    --max-model-len-by-strategy-file evaluation/specs/prompt_tournament_max_model_len.json \
    --model-options "$model_options_json"

  RAN_ANY=1
done <<< "$missing_jobs"

echo "[4/4] Rebuilding canonical + unperturbed metrics"
if [[ "$RAN_ANY" == "1" || "$ONLY_MISSING" != "1" ]]; then
  rebuild_canonical_and_metrics
else
  echo "[4/4] No new sampling jobs; canonical/metrics already synced."
fi

echo "Done."
echo "Unperturbed belief output root: $OUT"
