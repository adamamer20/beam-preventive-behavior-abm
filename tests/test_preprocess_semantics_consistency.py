from __future__ import annotations

import json
from collections import deque

from beam_abm.preprocess.cleaning.graph import build_transformation_graph
from beam_abm.preprocess.cleaning.models import (
    Column,
    DataCleaningModel,
    EncodingTransformation,
    MergedTransformation,
    RenamingTransformation,
)
from beam_abm.preprocess.cleaning.references import (
    build_rename_to_map,
    resolve_ref,
    validate_references_before_processing,
)
from beam_abm.preprocess.specs.derivations import build_derivations


def test_reference_and_dependency_semantics_are_consistent(tmp_path) -> None:
    model = DataCleaningModel(
        old_columns=[
            Column(
                id="raw",
                section="demo",
                transformations=deque(
                    [
                        EncodingTransformation(encoding="$NUM"),
                        RenamingTransformation(rename_to=["alias_raw"]),
                    ]
                ),
            ),
            Column(
                id="derived",
                section="demo",
                transformations=deque([MergedTransformation(merging_type="mean", merged_from=["alias_raw"])]),
            ),
        ]
    )

    all_columns = list(model.old_columns)
    spec_ids = {c.id for c in all_columns}
    rename_to_map = build_rename_to_map(all_columns)

    final_refs = validate_references_before_processing(
        model=model,
        spec_ids=spec_ids,
        rename_to_map=rename_to_map,
    )
    assert "alias_raw" in final_refs

    nodes = build_transformation_graph(
        model=model,
        resolve_ref=lambda name: resolve_ref(name, spec_ids=spec_ids, rename_to_map=rename_to_map),
    )
    order = {(n.col_id, n.index_in_col): i for i, n in enumerate(nodes)}
    assert order[("raw", 0)] < order[("derived", 0)]

    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "derivations.tsv"
    spec_path.write_text(
        json.dumps(
            {
                "new_columns": [
                    {
                        "id": "raw",
                        "section": "demo",
                        "transformations": [
                            {"type": "encoding", "encoding": "$NUM"},
                            {"type": "renaming", "rename_to": ["alias_raw"]},
                        ],
                    },
                    {
                        "id": "derived",
                        "section": "demo",
                        "transformations": [{"type": "merged", "merging_type": "mean", "merged_from": ["alias_raw"]}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = build_derivations(spec_path=spec_path, out_path=out_path, include_splits=False)
    assert any(r.col == "derived" and r.derivation_col == "alias_raw" and r.method == "merged:mean" for r in rows)
