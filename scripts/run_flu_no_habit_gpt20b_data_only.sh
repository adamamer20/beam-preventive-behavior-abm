#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-flu-no-habit-gpt20b-data-only-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation/output/choice_validation/perturbed}"
EVAL_ROOT="${EVAL_ROOT:-empirical/output/anchors/pe/mutable_engines/reduced/eval_flu_no_habit}"
SOURCE_ROOT="${SOURCE_ROOT:-empirical/output/anchors/pe/mutable_engines/reduced}"

if [[ ! -d "$SOURCE_ROOT/flu_vaccinated_2023_2024_no_habit" ]]; then
  echo "Missing source design: $SOURCE_ROOT/flu_vaccinated_2023_2024_no_habit" >&2
  exit 1
fi

mkdir -p "$EVAL_ROOT/flu_vaccinated_2023_2024"

# Build an eval root where flu uses the no-habit design/ref ICE but canonical
# flu outcome name, and explicitly drop flu-history context columns so prompts
# cannot condition on habit.
SOURCE_ROOT="$SOURCE_ROOT" EVAL_ROOT="$EVAL_ROOT" uv run python - <<'PY'
import os
from pathlib import Path
import polars as pl

src = Path(os.environ["SOURCE_ROOT"]) / "flu_vaccinated_2023_2024_no_habit" / "design.parquet"
dst = Path(os.environ["EVAL_ROOT"]) / "flu_vaccinated_2023_2024" / "design.parquet"

history_cols = [
    "flu_vaccinated_pre_pandemic",
    "flu_vaccinated_2020_2021",
    "flu_vaccinated_2021_2022",
    "flu_vaccinated_2022_2023",
]

df = pl.read_parquet(src)
drop_cols = [c for c in history_cols if c in df.columns]
if drop_cols:
    df = df.drop(drop_cols)
df.write_parquet(dst)
print("Wrote design without flu-history cols:", drop_cols)
PY

if [[ -f "$SOURCE_ROOT/flu_vaccinated_2023_2024_no_habit/ice_ref.csv" ]]; then
  cp -f "$SOURCE_ROOT/flu_vaccinated_2023_2024_no_habit/ice_ref.csv" \
    "$EVAL_ROOT/flu_vaccinated_2023_2024/ice_ref.csv"
fi

if [[ -f "$SOURCE_ROOT/flu_vaccinated_2023_2024_no_habit/ice_ref__B_FLU_vaccinated_2023_2024_ENGINES.csv" ]]; then
  cp -f "$SOURCE_ROOT/flu_vaccinated_2023_2024_no_habit/ice_ref__B_FLU_vaccinated_2023_2024_ENGINES.csv" \
    "$EVAL_ROOT/flu_vaccinated_2023_2024/ice_ref__B_FLU_vaccinated_2023_2024_ENGINES.csv"
fi

KEEP_TARGETS="$(uv run python - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path("config/levers.json").read_text(encoding="utf-8"))
entry = cfg.get("flu_vaccinated_2023_2024_no_habit", {})
levers = [str(v).strip() for v in entry.get("levers", []) if str(v).strip()]
print(",".join(levers))
PY
)"

if [[ -z "$KEEP_TARGETS" ]]; then
  echo "Could not derive keep targets for flu_vaccinated_2023_2024_no_habit from config/levers.json" >&2
  exit 1
fi

BACKEND="${BACKEND:-azure_openai}"
MODEL_OPTIONS="${MODEL_OPTIONS:-{\"reasoning\":{\"effort\":\"medium\"}}}"
K="${K:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-24}"

uv run python evaluation/scripts/run_behavioural_outcomes_perturbed.py \
  --output-root "$OUTPUT_ROOT" \
  --run-tag "$RUN_TAG" \
  --models "$MODEL" \
  --prompt-families "data_only" \
  --keep-outcomes "flu_vaccinated_2023_2024" \
  --ref-models "linear" \
  --generate-prompts-args "--in-dir $EVAL_ROOT --keep-outcomes flu_vaccinated_2023_2024 --keep-targets $KEEP_TARGETS --keep-grid-points low,high --dedup-levers" \
  --sample-args "--backend $BACKEND --k $K --max-concurrency $MAX_CONCURRENCY --model-options '$MODEL_OPTIONS'"

echo
echo "Done. Canonical flu output:"
echo "  $OUTPUT_ROOT/flu_vaccinated_2023_2024/gpt-oss-20b/data_only"
