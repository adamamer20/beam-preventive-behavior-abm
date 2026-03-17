"""Network validation benchmarking gate."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import KDTree

from beam_abm.abm.network import build_contact_network, compute_peer_means
from beam_abm.abm.state_schema import MUTABLE_COLUMNS

from .core import GateResult, _pearson_corr_safe, _safe_quantiles


def run_network_validation_gate(
    agents_t0: pl.DataFrame,
    *,
    k: int = 10,
    mixing: float = 0.3,
    seed: int = 42,
) -> GateResult:
    """Validate kNN network geometry and rewiring robustness."""
    del k
    gate = GateResult(gate_name="network_validation")

    feature_columns = sorted(col for col in MUTABLE_COLUMNS if col in agents_t0.columns)
    if not feature_columns or agents_t0.height < 3:
        gate.add_check(
            "network_inputs_available",
            passed=False,
            details={"reason": "insufficient feature columns or agents"},
        )
        gate.evaluate()
        return gate

    X = agents_t0.select(feature_columns).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    jitter_rng = np.random.default_rng(seed + 104_729)
    col_scale = np.std(X, axis=0)
    col_scale = np.where(col_scale > 1e-8, col_scale, 1.0)
    X_knn = X + jitter_rng.normal(0.0, 1e-6, size=X.shape) * col_scale
    rounded = np.round(X_knn, decimals=8)
    unique_rows = np.unique(rounded, axis=0).shape[0]
    tie_rate = float(1.0 - (unique_rows / max(1, agents_t0.height)))
    gate.add_check(
        "feature_space_tie_rate",
        passed=tie_rate < 0.5,
        details={"tie_rate": tie_rate, "n_agents": agents_t0.height},
    )

    tree = KDTree(X_knn)
    dists, _ = tree.query(X, k=2)
    nn = dists[:, 1]
    nn_q10, nn_q50, nn_q90 = _safe_quantiles(nn.astype(np.float64))
    gate.add_check(
        "nearest_neighbor_distance_distribution",
        passed=nn_q90 > 1e-8 or tie_rate < 0.95,
        details={
            "nn_q10": nn_q10,
            "nn_q50": nn_q50,
            "nn_q90": nn_q90,
            "nn_max": float(np.max(nn)),
        },
    )
    if "country" in agents_t0.columns:
        countries = sorted({str(c) for c in agents_t0["country"].to_list()})
        for country in countries:
            mask = agents_t0["country"] == country
            idx = np.where(mask.to_numpy())[0]
            if idx.size < 3:
                continue
            Xc = X_knn[idx]
            tree_c = KDTree(Xc)
            d_c, _ = tree_c.query(Xc, k=2)
            zero_mass = float(np.mean(d_c[:, 1] <= 1e-10))
            gate.add_check(
                f"nn_zero_distance_mass_{country}",
                passed=zero_mass <= 0.05,
                details={"zero_distance_mass": zero_mass, "n_agents": int(idx.size)},
            )

    network = build_contact_network(agents_t0, bridge_prob=float(np.clip(mixing, 0.0, 1.0)), seed=seed)
    degrees = network.core_degree.astype(np.float64)
    gate.add_check(
        "degree_sanity",
        passed=float(np.min(degrees)) >= 1.0,
        details={
            "degree_min": float(np.min(degrees)),
            "degree_p50": float(np.median(degrees)),
            "degree_p90": float(np.nanpercentile(degrees, 90)),
        },
    )

    edge_rows: list[int] = []
    edge_cols: list[int] = []
    for i, nbrs in enumerate(network.core_neighbors):
        for j in nbrs:
            edge_rows.append(i)
            edge_cols.append(int(j))
    edge_row_arr = np.array(edge_rows, dtype=np.int64)
    edge_col_arr = np.array(edge_cols, dtype=np.int64)

    if "row_id" in agents_t0.columns:
        row_ids = agents_t0["row_id"].to_numpy()
        same_row_share = (
            float(np.mean(row_ids[edge_row_arr] == row_ids[edge_col_arr])) if edge_row_arr.size > 0 else 0.0
        )
        gate.add_check(
            "edge_composition_same_row",
            passed=same_row_share <= min(0.4, tie_rate + 0.15),
            details={"same_row_share": same_row_share, "tie_rate": tie_rate},
        )

    # Homophily sanity: edge-level feature correlation should not be degenerate.
    key_features = [
        c
        for c in (
            "institutional_trust_avg",
            "social_norms_vax_avg",
            "vaccine_risk_avg",
        )
        if c in agents_t0.columns
    ][:2]
    for feature in key_features:
        values = agents_t0[feature].cast(pl.Float64).to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if edge_row_arr.size == 0:
            corr = 0.0
        else:
            corr = _pearson_corr_safe(values[edge_row_arr], values[edge_col_arr])
        gate.add_check(
            f"homophily_sanity_{feature}",
            passed=np.isfinite(corr) and (abs(corr) < 0.995),
            details={"edge_corr": corr},
        )

    net_lo = build_contact_network(agents_t0, bridge_prob=0.05, seed=seed)
    net_hi = build_contact_network(agents_t0, bridge_prob=0.70, seed=seed)
    edges_lo: set[tuple[int, int]] = set()
    for i, nbrs in enumerate(net_lo.core_neighbors):
        edges_lo |= {(int(i), int(j)) for j in nbrs}
    edges_hi: set[tuple[int, int]] = set()
    for i, nbrs in enumerate(net_hi.core_neighbors):
        edges_hi |= {(int(i), int(j)) for j in nbrs}
    union = len(edges_lo | edges_hi)
    inter = len(edges_lo & edges_hi)
    jaccard = float(inter / union) if union > 0 else 1.0
    gate.add_check(
        "mixing_robustness",
        passed=jaccard < 0.90,
        details={"edge_jaccard_low_vs_high_mixing": jaccard},
    )

    # Sensitivity to tiny jitter in feature space: macro peer shifts should
    # remain directionally stable.
    jitter_rng = np.random.default_rng(seed)
    jitter_cols = [c for c in feature_columns if c in agents_t0.columns]
    jitter_series: list[pl.Series] = []
    for col in jitter_cols:
        vals = agents_t0[col].to_numpy().astype(np.float64)
        vals_clean = np.nan_to_num(vals, nan=0.0)
        col_std = float(np.std(vals_clean))
        scale = max(col_std, 1e-3) * 1e-4
        jitter = jitter_rng.normal(0.0, scale, size=agents_t0.height)
        jitter_series.append(pl.Series(name=col, values=(vals_clean + jitter)))
    jittered = agents_t0.with_columns(jitter_series) if jitter_series else agents_t0

    net_base = build_contact_network(agents_t0, bridge_prob=float(np.clip(mixing, 0.0, 1.0)), seed=seed)
    net_jitter = build_contact_network(jittered, bridge_prob=float(np.clip(mixing, 0.0, 1.0)), seed=seed)
    peer_base = compute_peer_means(agents_t0, net_base, columns=key_features)
    peer_jitter = compute_peer_means(agents_t0, net_jitter, columns=key_features)
    for feature in key_features:
        vals = agents_t0[feature].to_numpy().astype(np.float64)
        vals_clean = np.nan_to_num(vals, nan=0.0)
        macro_base = float(np.mean(peer_base[feature] - vals_clean))
        macro_jitter = float(np.mean(peer_jitter[feature] - vals_clean))
        flip = np.sign(macro_base) != np.sign(macro_jitter)
        meaningful = abs(macro_base) > 1e-6 and abs(macro_jitter) > 1e-6
        gate.add_check(
            f"jitter_sensitivity_{feature}",
            passed=not (flip and meaningful),
            details={
                "macro_shift_base": macro_base,
                "macro_shift_jitter": macro_jitter,
                "sign_flip": bool(flip),
            },
        )
    if key_features:
        sign_flips = 0
        checked = 0
        for feature in key_features:
            vals = agents_t0[feature].to_numpy().astype(np.float64)
            vals_clean = np.nan_to_num(vals, nan=0.0)
            macro_base = float(np.mean(peer_base[feature] - vals_clean))
            macro_jitter = float(np.mean(peer_jitter[feature] - vals_clean))
            if abs(macro_base) <= 1e-6 or abs(macro_jitter) <= 1e-6:
                continue
            checked += 1
            if np.sign(macro_base) != np.sign(macro_jitter):
                sign_flips += 1
        gate.add_check(
            "network_sensitivity_jitter",
            passed=sign_flips == 0,
            details={"n_checked_features": checked, "n_sign_flips": sign_flips},
        )

    gate.evaluate()
    logger.info(
        "Network validation gate: {status} ({n_pass}/{n_total} checks).",
        status="PASSED" if gate.passed else "FAILED",
        n_pass=sum(1 for c in gate.checks if c.passed),
        n_total=len(gate.checks),
    )
    return gate
