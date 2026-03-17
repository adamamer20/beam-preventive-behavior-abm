"""Locality-based recurring/one-shot contact network utilities.

Network structure is static at init:
- recurring core ties (from group-size propensity),
- optional household ties,
- locality membership index.

Per tick, contacts are sampled from one unified contact-rate process that
splits volume across household/core/one-shot sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

from beam_abm.abm.outcome_scales import OUTCOME_BOUNDS, normalize_outcome_to_unit_interval
from beam_abm.abm.state_schema import MUTABLE_COLUMNS

__all__ = [
    "ContactDiagnostics",
    "NetworkState",
    "build_contact_network",
    "export_network_adjacency",
    "compute_pairwise_symmetric_peer_pull",
    "compute_peer_means",
    "compute_homophily_weighted_peer_means",
    "compute_sampled_peer_adoption",
]


@dataclass(slots=True)
class NetworkState:
    """Static network state for locality-based contact sampling."""

    locality_members: dict[str, np.ndarray]
    locality_of_agent: np.ndarray
    core_neighbors: list[np.ndarray]
    household_neighbors: list[np.ndarray]
    core_degree: np.ndarray
    has_household: np.ndarray
    bridge_prob: float
    recurring_share: np.ndarray
    household_share: np.ndarray
    recurring_edge_src: np.ndarray
    recurring_edge_dst: np.ndarray
    core_neighbors_flat: np.ndarray
    core_neighbors_offsets: np.ndarray
    core_neighbor_counts: np.ndarray
    household_neighbors_flat: np.ndarray
    household_neighbors_offsets: np.ndarray
    household_neighbor_counts: np.ndarray
    locality_code_of_agent: np.ndarray
    locality_members_flat: np.ndarray
    locality_members_offsets: np.ndarray
    locality_member_counts: np.ndarray
    locality_pos_in_members: np.ndarray


@dataclass(slots=True)
class ContactDiagnostics:
    """Tick-level contact diagnostics."""

    mean_contacts: float
    recurring_share: float
    household_share: float
    one_shot_share: float
    cross_locality_share: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_contacts": self.mean_contacts,
            "recurring_share": self.recurring_share,
            "household_share": self.household_share,
            "one_shot_share": self.one_shot_share,
            "cross_locality_share": self.cross_locality_share,
        }


def _safe_numeric_column(agents: pl.DataFrame, column: str, n_agents: int) -> np.ndarray:
    if column not in agents.columns:
        return np.zeros(n_agents, dtype=np.float64)
    return np.nan_to_num(agents[column].cast(pl.Float64).to_numpy(), nan=0.0)


def _build_locality_members(locality_ids: np.ndarray) -> dict[str, np.ndarray]:
    locality_members: dict[str, np.ndarray] = {}
    for locality in np.unique(locality_ids):
        key = str(locality)
        locality_members[key] = np.where(locality_ids == locality)[0]
    return locality_members


def _sample_from_pool_excluding_self(
    rng: np.random.Generator,
    pool: np.ndarray,
    agent_idx: int,
) -> int | None:
    if pool.size <= 1:
        return None
    sampled = int(rng.choice(pool))
    if sampled != agent_idx:
        return sampled
    if pool.size == 2:
        return int(pool[0] if pool[1] == agent_idx else pool[1])
    candidates = pool[pool != agent_idx]
    if candidates.size == 0:
        return None
    return int(rng.choice(candidates))


def _build_recurring_edge_index(
    core_neighbors: list[np.ndarray],
    household_neighbors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []

    for i, (core, hh) in enumerate(zip(core_neighbors, household_neighbors, strict=False)):
        if core.size == 0 and hh.size == 0:
            continue
        if hh.size == 0:
            nbrs = core
        elif core.size == 0:
            nbrs = hh
        else:
            nbrs = np.union1d(core, hh)
        nbrs = nbrs[nbrs != i]
        if nbrs.size == 0:
            continue
        src_parts.append(np.full(nbrs.size, i, dtype=np.int64))
        dst_parts.append(nbrs.astype(np.int64, copy=False))

    if not src_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.concatenate(src_parts), np.concatenate(dst_parts)


def _build_ragged_index(
    neighbors: list[np.ndarray],
    n_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.fromiter((arr.size for arr in neighbors), dtype=np.int64, count=n_agents)
    offsets = np.empty(n_agents + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    if int(offsets[-1]) == 0:
        flat = np.array([], dtype=np.int64)
    else:
        flat = np.concatenate(neighbors).astype(np.int64, copy=False)
    return flat, offsets, counts


def _build_locality_index(
    locality_ids: np.ndarray,
    locality_members: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_agents = locality_ids.shape[0]
    locality_keys = tuple(locality_members.keys())
    n_localities = len(locality_keys)
    code_by_locality = {key: idx for idx, key in enumerate(locality_keys)}

    locality_code_of_agent = np.empty(n_agents, dtype=np.int64)
    for agent_idx, locality in enumerate(locality_ids):
        locality_code_of_agent[agent_idx] = code_by_locality[str(locality)]

    locality_member_counts = np.array(
        [int(locality_members[key].size) for key in locality_keys],
        dtype=np.int64,
    )
    locality_members_offsets = np.empty(n_localities + 1, dtype=np.int64)
    locality_members_offsets[0] = 0
    np.cumsum(locality_member_counts, out=locality_members_offsets[1:])

    if int(locality_members_offsets[-1]) == 0:
        locality_members_flat = np.array([], dtype=np.int64)
    else:
        locality_members_flat = np.concatenate([locality_members[key] for key in locality_keys]).astype(
            np.int64,
            copy=False,
        )

    locality_pos_in_members = np.zeros(n_agents, dtype=np.int64)
    for _loc_idx, key in enumerate(locality_keys):
        members = locality_members[key].astype(np.int64, copy=False)
        if members.size == 0:
            continue
        locality_pos_in_members[members] = np.arange(members.size, dtype=np.int64)

    return (
        locality_code_of_agent,
        locality_members_flat,
        locality_members_offsets,
        locality_member_counts,
        locality_pos_in_members,
    )


def _build_households(
    locality_members: dict[str, np.ndarray],
    household_size: np.ndarray,
    rng: np.random.Generator,
    household_max_size: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    n_agents = household_size.shape[0]
    households: list[list[int]] = []
    household_neighbors: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(n_agents)]
    has_household = np.zeros(n_agents, dtype=bool)

    for members in locality_members.values():
        if members.size == 0:
            continue
        shuffled = members.copy()
        rng.shuffle(shuffled)
        cursor = 0
        while cursor < shuffled.size:
            leader = int(shuffled[cursor])
            desired = int(np.clip(round(float(household_size[leader])), 1, household_max_size))
            take = min(desired, int(shuffled.size - cursor))
            block = shuffled[cursor : cursor + take].astype(np.int64)
            households.append(block.tolist())
            cursor += take

    for household in households:
        if len(household) <= 1:
            continue
        members_arr = np.array(household, dtype=np.int64)
        for agent_idx in household:
            neighbors = members_arr[members_arr != agent_idx]
            household_neighbors[agent_idx] = neighbors
            has_household[agent_idx] = neighbors.size > 0

    return household_neighbors, has_household


def build_contact_network(
    agents: pl.DataFrame,
    *,
    bridge_prob: float = 0.08,
    household_enabled: bool = True,
    household_max_size: int = 7,
    seed: int = 42,
) -> NetworkState:
    """Build a locality-based recurring contact network state."""
    n_agents = agents.height
    if n_agents == 0:
        return NetworkState(
            locality_members={},
            locality_of_agent=np.array([], dtype=object),
            core_neighbors=[],
            household_neighbors=[],
            core_degree=np.array([], dtype=np.int64),
            has_household=np.array([], dtype=bool),
            bridge_prob=bridge_prob,
            recurring_share=np.array([], dtype=np.float64),
            household_share=np.array([], dtype=np.float64),
            recurring_edge_src=np.array([], dtype=np.int64),
            recurring_edge_dst=np.array([], dtype=np.int64),
            core_neighbors_flat=np.array([], dtype=np.int64),
            core_neighbors_offsets=np.array([0], dtype=np.int64),
            core_neighbor_counts=np.array([], dtype=np.int64),
            household_neighbors_flat=np.array([], dtype=np.int64),
            household_neighbors_offsets=np.array([0], dtype=np.int64),
            household_neighbor_counts=np.array([], dtype=np.int64),
            locality_code_of_agent=np.array([], dtype=np.int64),
            locality_members_flat=np.array([], dtype=np.int64),
            locality_members_offsets=np.array([0], dtype=np.int64),
            locality_member_counts=np.array([], dtype=np.int64),
            locality_pos_in_members=np.array([], dtype=np.int64),
        )

    if "locality_id" not in agents.columns:
        msg = "Missing required column 'locality_id' for locality-based network construction."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    locality_ids = agents["locality_id"].cast(pl.Utf8).to_numpy().astype(object)
    locality_members = _build_locality_members(locality_ids)
    all_agents = np.arange(n_agents, dtype=np.int64)

    if "core_degree" in agents.columns:
        raw_degree = _safe_numeric_column(agents, "core_degree", n_agents)
    elif "group_size_score" in agents.columns:
        score = _safe_numeric_column(agents, "group_size_score", n_agents)
        raw_degree = 4.0 + score * 26.0
    else:
        raw_degree = np.full(n_agents, 17.0, dtype=np.float64)

    core_degree = np.clip(np.rint(raw_degree), 4, 30).astype(np.int64)
    core_neighbors: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(n_agents)]

    for agent_idx in range(n_agents):
        loc_key = str(locality_ids[agent_idx])
        local_pool = locality_members.get(loc_key, all_agents)
        desired = int(core_degree[agent_idx])
        selected: set[int] = set()

        attempts = 0
        max_attempts = max(desired * 20, 100)
        while len(selected) < desired and attempts < max_attempts:
            attempts += 1
            if rng.random() < (1.0 - bridge_prob):
                candidate = _sample_from_pool_excluding_self(rng, local_pool, agent_idx)
            else:
                candidate = _sample_from_pool_excluding_self(rng, all_agents, agent_idx)
            if candidate is None:
                continue
            selected.add(candidate)

        if selected:
            core_neighbors[agent_idx] = np.array(sorted(selected), dtype=np.int64)

    household_neighbors: list[np.ndarray]
    has_household: np.ndarray
    if household_enabled:
        household_size = _safe_numeric_column(agents, "household_size", n_agents)
        household_neighbors, has_household = _build_households(
            locality_members,
            household_size,
            rng,
            household_max_size=household_max_size,
        )
    else:
        household_neighbors = [np.array([], dtype=np.int64) for _ in range(n_agents)]
        has_household = np.zeros(n_agents, dtype=bool)

    if "recurring_share" in agents.columns:
        recurring_share = np.clip(_safe_numeric_column(agents, "recurring_share", n_agents), 0.05, 0.90)
    elif "friend_closure" in agents.columns:
        recurring_share = np.clip(0.20 + 0.65 * _safe_numeric_column(agents, "friend_closure", n_agents), 0.05, 0.90)
    else:
        recurring_share = np.full(n_agents, 0.45, dtype=np.float64)

    if "household_contact_share" in agents.columns:
        household_share = np.clip(_safe_numeric_column(agents, "household_contact_share", n_agents), 0.0, 0.70)
    else:
        household_share = np.where(has_household, 0.35, 0.0).astype(np.float64)

    household_share = np.where(has_household, household_share, 0.0)
    recurring_share = np.clip(recurring_share, 0.0, 1.0 - household_share)

    cross_edges = 0
    edge_total = 0
    for i, nbrs in enumerate(core_neighbors):
        edge_total += int(nbrs.size)
        if nbrs.size > 0:
            cross_edges += int(np.sum(locality_ids[nbrs] != locality_ids[i]))
    realised_bridge = (cross_edges / edge_total) if edge_total > 0 else 0.0

    logger.info(
        "Built locality network: n={n}, mean_core_degree={d:.2f}, bridge={b:.3f}, households={h:.2f}",
        n=n_agents,
        d=float(np.mean(core_degree)) if core_degree.size > 0 else 0.0,
        b=realised_bridge,
        h=float(np.mean(has_household.astype(np.float64))) if has_household.size > 0 else 0.0,
    )

    recurring_edge_src, recurring_edge_dst = _build_recurring_edge_index(core_neighbors, household_neighbors)
    core_neighbors_flat, core_neighbors_offsets, core_neighbor_counts = _build_ragged_index(core_neighbors, n_agents)
    household_neighbors_flat, household_neighbors_offsets, household_neighbor_counts = _build_ragged_index(
        household_neighbors,
        n_agents,
    )
    (
        locality_code_of_agent,
        locality_members_flat,
        locality_members_offsets,
        locality_member_counts,
        locality_pos_in_members,
    ) = _build_locality_index(locality_ids, locality_members)

    return NetworkState(
        locality_members=locality_members,
        locality_of_agent=locality_ids,
        core_neighbors=core_neighbors,
        household_neighbors=household_neighbors,
        core_degree=core_degree,
        has_household=has_household,
        bridge_prob=bridge_prob,
        recurring_share=recurring_share.astype(np.float64),
        household_share=household_share.astype(np.float64),
        recurring_edge_src=recurring_edge_src,
        recurring_edge_dst=recurring_edge_dst,
        core_neighbors_flat=core_neighbors_flat,
        core_neighbors_offsets=core_neighbors_offsets,
        core_neighbor_counts=core_neighbor_counts,
        household_neighbors_flat=household_neighbors_flat,
        household_neighbors_offsets=household_neighbors_offsets,
        household_neighbor_counts=household_neighbor_counts,
        locality_code_of_agent=locality_code_of_agent,
        locality_members_flat=locality_members_flat,
        locality_members_offsets=locality_members_offsets,
        locality_member_counts=locality_member_counts,
        locality_pos_in_members=locality_pos_in_members,
    )


def export_network_adjacency(network: NetworkState) -> pl.DataFrame:
    """Export directed adjacency edges for recurring contacts.

    Returns a long-form edge table with columns ``src``, ``dst`` and
    ``edge_type`` where ``edge_type`` is one of ``core`` or ``household``.
    """
    rows: list[dict[str, int | str]] = []
    for src, nbrs in enumerate(network.core_neighbors):
        for dst in nbrs:
            rows.append({"src": int(src), "dst": int(dst), "edge_type": "core"})
    for src, nbrs in enumerate(network.household_neighbors):
        for dst in nbrs:
            rows.append({"src": int(src), "dst": int(dst), "edge_type": "household"})
    if not rows:
        return pl.DataFrame(
            {
                "src": pl.Series([], dtype=pl.Int64),
                "dst": pl.Series([], dtype=pl.Int64),
                "edge_type": pl.Series([], dtype=pl.Utf8),
            }
        )
    return pl.DataFrame(rows).sort(["src", "dst", "edge_type"])


def compute_peer_means(
    agents: pl.DataFrame,
    network: NetworkState,
    columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute simple means over recurring ties (core + household)."""
    if columns is None:
        columns = sorted(col for col in MUTABLE_COLUMNS if col in agents.columns)

    n_agents = agents.height
    peer_means: dict[str, np.ndarray] = {}

    src = network.recurring_edge_src
    dst = network.recurring_edge_dst
    counts = np.bincount(src, minlength=n_agents).astype(np.float64) if src.size > 0 else np.zeros(n_agents)

    for col in columns:
        if col not in agents.columns:
            continue
        values = np.nan_to_num(agents[col].cast(pl.Float64).to_numpy(), nan=0.0)
        out = values.copy()
        if src.size > 0:
            sums = np.bincount(src, weights=values[dst], minlength=n_agents).astype(np.float64)
            has_neighbors = counts > 0.0
            out[has_neighbors] = sums[has_neighbors] / counts[has_neighbors]
        peer_means[col] = out

    return peer_means


def compute_homophily_weighted_peer_means(
    agents: pl.DataFrame,
    network: NetworkState,
    *,
    columns: tuple[str, ...],
    homophily_gamma: float,
    homophily_w_trust: float,
    homophily_w_moral: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Compute homophily-weighted peer means on selected constructs."""
    n_agents = agents.height
    if n_agents == 0:
        return {}, 0.0

    if "institutional_trust_avg" not in agents.columns or "moral_progressive_orientation" not in agents.columns:
        fallback = compute_peer_means(agents, network, list(columns))
        return fallback, 0.0

    trust = np.nan_to_num(agents["institutional_trust_avg"].cast(pl.Float64).to_numpy(), nan=0.0)
    moral = np.nan_to_num(agents["moral_progressive_orientation"].cast(pl.Float64).to_numpy(), nan=0.0)

    values_by_col: dict[str, np.ndarray] = {}
    for col in columns:
        if col in agents.columns:
            values_by_col[col] = np.nan_to_num(agents[col].cast(pl.Float64).to_numpy(), nan=0.0)

    if not values_by_col:
        return {}, 0.0

    src = network.recurring_edge_src
    dst = network.recurring_edge_dst
    if src.size == 0:
        return {col: vals.copy() for col, vals in values_by_col.items()}, 0.0

    d_trust = np.abs(trust[dst] - trust[src])
    d_moral = np.abs(moral[dst] - moral[src])
    distance = homophily_w_trust * d_trust + homophily_w_moral * d_moral
    weights = np.exp(-homophily_gamma * distance).astype(np.float64)

    counts = np.bincount(src, minlength=n_agents).astype(np.float64)
    weight_sums = np.bincount(src, weights=weights, minlength=n_agents).astype(np.float64)
    has_neighbors = counts > 0.0
    valid_weighted = weight_sums > 0.0

    out: dict[str, np.ndarray] = {}
    for col, vals in values_by_col.items():
        result = vals.copy()
        weighted_num = np.bincount(src, weights=weights * vals[dst], minlength=n_agents).astype(np.float64)
        weighted_mask = has_neighbors & valid_weighted
        result[weighted_mask] = weighted_num[weighted_mask] / weight_sums[weighted_mask]

        fallback_mask = has_neighbors & (~valid_weighted)
        if np.any(fallback_mask):
            simple_num = np.bincount(src, weights=vals[dst], minlength=n_agents).astype(np.float64)
            result[fallback_mask] = simple_num[fallback_mask] / counts[fallback_mask]
        out[col] = result

    similarity_by_agent = np.zeros(n_agents, dtype=np.float64)
    similarity_by_agent[has_neighbors] = weight_sums[has_neighbors] / counts[has_neighbors]
    mean_similarity = float(np.mean(similarity_by_agent[has_neighbors])) if np.any(has_neighbors) else 0.0
    return out, mean_similarity


def compute_pairwise_symmetric_peer_pull(
    agents: pl.DataFrame,
    network: NetworkState,
    *,
    columns: tuple[str, ...],
    gains_by_column: dict[str, float],
    homophily_gamma: float,
    homophily_w_trust: float,
    homophily_w_moral: float,
    rng: np.random.Generator,
    interactions_per_agent: float = 0.5,
    per_agent_interactions: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Apply pairwise symmetric gossip updates on sampled recurring edges.

    Edges are sampled with homophily weights; each sampled pair ``(i, j)`` is
    updated symmetrically for every configured column:

    ``x_i <- x_i + g (x_j - x_i)``, ``x_j <- x_j + g (x_i - x_j)``.

    This preserves the pair mean exactly per interaction (for ``g in [0, 1]``)
    and therefore preserves population means in expectation under edge sampling.
    """
    n_agents = agents.height
    if n_agents == 0:
        return {}, 0.0

    src = network.recurring_edge_src
    dst = network.recurring_edge_dst
    if src.size == 0:
        return {}, 0.0

    values_by_col: dict[str, np.ndarray] = {}
    for col in columns:
        gain = float(np.clip(gains_by_column.get(col, 0.0), 0.0, 1.0))
        if gain <= 0.0:
            continue
        if col not in agents.columns:
            continue
        values_by_col[col] = np.nan_to_num(agents[col].cast(pl.Float64).to_numpy(), nan=0.0).astype(np.float64)

    if not values_by_col:
        return {}, 0.0

    if "institutional_trust_avg" in agents.columns and "moral_progressive_orientation" in agents.columns:
        trust = np.nan_to_num(agents["institutional_trust_avg"].cast(pl.Float64).to_numpy(), nan=0.0)
        moral = np.nan_to_num(agents["moral_progressive_orientation"].cast(pl.Float64).to_numpy(), nan=0.0)
        d_trust = np.abs(trust[dst] - trust[src])
        d_moral = np.abs(moral[dst] - moral[src])
        distance = homophily_w_trust * d_trust + homophily_w_moral * d_moral
        weights = np.exp(-homophily_gamma * distance).astype(np.float64)
    else:
        weights = np.ones(src.size, dtype=np.float64)

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        probs = None
    else:
        probs = weights / total_weight

    sampled_src: np.ndarray
    sampled_dst: np.ndarray
    sampled_w: np.ndarray

    if per_agent_interactions is not None and per_agent_interactions.shape[0] == n_agents:
        per_agent_rates = np.clip(
            np.nan_to_num(per_agent_interactions.astype(np.float64), nan=0.0, posinf=8.0, neginf=0.0),
            0.0,
            8.0,
        )
        total_rate = float(np.sum(per_agent_rates))
        if (not np.isfinite(total_rate)) or total_rate <= 0.0:
            return {}, 0.0

        # Poisson splitting: sample a global event count and draw edges from
        # src-normalized edge intensities to match per-agent expected rates.
        n_events = int(rng.poisson(total_rate))
        if n_events <= 0:
            return {}, 0.0

        edge_counts = np.bincount(src, minlength=n_agents).astype(np.int64)
        local_weight_sums = np.bincount(src, weights=weights, minlength=n_agents).astype(np.float64)
        src_local_sum = local_weight_sums[src]
        src_edge_count = edge_counts[src]

        edge_intensity = np.zeros(src.size, dtype=np.float64)
        valid_weight = src_local_sum > 0.0
        edge_intensity[valid_weight] = per_agent_rates[src[valid_weight]] * (
            weights[valid_weight] / src_local_sum[valid_weight]
        )
        fallback = (~valid_weight) & (src_edge_count > 0)
        edge_intensity[fallback] = per_agent_rates[src[fallback]] / src_edge_count[fallback]
        edge_intensity = np.nan_to_num(edge_intensity, nan=0.0, posinf=0.0, neginf=0.0)

        total_intensity = float(np.sum(edge_intensity))
        if (not np.isfinite(total_intensity)) or total_intensity <= 0.0:
            return {}, 0.0
        edge_probs = edge_intensity / total_intensity

        sampled_edge_idx = rng.choice(src.size, size=n_events, replace=True, p=edge_probs)
        sampled_src = src[sampled_edge_idx]
        sampled_dst = dst[sampled_edge_idx]
        sampled_w = weights[sampled_edge_idx]
    else:
        n_events = int(max(1, round(float(np.clip(interactions_per_agent, 0.05, 8.0)) * float(n_agents))))
        sampled_edge_idx = rng.choice(src.size, size=n_events, replace=True, p=probs)
        sampled_src = src[sampled_edge_idx]
        sampled_dst = dst[sampled_edge_idx]
        sampled_w = weights[sampled_edge_idx]

    mean_similarity = float(np.mean(sampled_w)) if sampled_w.size > 0 else 0.0

    col_names = tuple(values_by_col.keys())
    gains = np.array([float(np.clip(gains_by_column.get(col, 0.0), 0.0, 1.0)) for col in col_names], dtype=np.float64)
    matrix = np.column_stack([values_by_col[col].copy() for col in col_names]).astype(np.float64, copy=False)

    for edge_idx in range(sampled_src.size):
        i = int(sampled_src[edge_idx])
        j = int(sampled_dst[edge_idx])
        if i == j:
            continue
        x_i = matrix[i]
        x_j = matrix[j]
        delta = gains * (x_j - x_i)
        matrix[i] = x_i + delta
        matrix[j] = x_j - delta

    updated = {col: matrix[:, k].copy() for k, col in enumerate(col_names)}
    return updated, mean_similarity


def compute_sampled_peer_adoption(
    network: NetworkState,
    outcomes: dict[str, np.ndarray],
    sample_size: int,
    rng: np.random.Generator,
    attention: np.ndarray | None = None,
    sharing: np.ndarray | None = None,
    *,
    contact_rate_base: float = 1.0,
    contact_rate_multiplier: float = 1.0,
    activity_score: np.ndarray | None = None,
    contact_propensity: np.ndarray | None = None,
    per_agent_sample_size: np.ndarray | None = None,
) -> tuple[np.ndarray, ContactDiagnostics]:
    """Sample observed peer adoption with vectorised contact events."""
    n_agents = network.locality_of_agent.shape[0]
    if not outcomes or n_agents == 0:
        return np.zeros(n_agents, dtype=np.float64), ContactDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    outcome_arrays: list[np.ndarray] = []
    for name, values in outcomes.items():
        if not isinstance(values, np.ndarray):
            continue
        unit_vals = normalize_outcome_to_unit_interval(values, name)
        bounds = OUTCOME_BOUNDS.get(name)
        if bounds is not None and (bounds[1] - bounds[0]) > 1.0:
            mean_norm = float(np.nanmean(unit_vals))
            if mean_norm > 0.95:
                logger.warning(
                    "Outcome '{name}' normalized mean unusually high: {mean:.3f}",
                    name=name,
                    mean=mean_norm,
                )
        outcome_arrays.append(unit_vals)
    if not outcome_arrays:
        return np.zeros(n_agents, dtype=np.float64), ContactDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    protection_index = np.mean(np.vstack(outcome_arrays), axis=0)

    if activity_score is None or activity_score.shape[0] != n_agents:
        activity = np.zeros(n_agents, dtype=np.float64)
    else:
        activity = np.nan_to_num(activity_score.astype(np.float64), nan=0.0)

    if contact_propensity is None or contact_propensity.shape[0] != n_agents:
        propensity = np.ones(n_agents, dtype=np.float64)
    else:
        propensity = np.clip(np.nan_to_num(contact_propensity.astype(np.float64), nan=1.0), 0.1, 5.0)

    if attention is None or attention.shape[0] != n_agents:
        attention_arr = np.ones(n_agents, dtype=np.float64)
    else:
        attention_arr = np.clip(np.nan_to_num(attention.astype(np.float64), nan=1.0), 0.0, np.inf)

    if sharing is None or sharing.shape[0] != n_agents:
        sharing_arr = np.ones(n_agents, dtype=np.float64)
    else:
        sharing_arr = np.clip(np.nan_to_num(sharing.astype(np.float64), nan=1.0), 0.0, np.inf)

    sample_size_cap = max(1, int(sample_size))
    observed_share = protection_index.copy()

    sharing_gate = np.clip(sharing_arr / (1.0 + sharing_arr), 0.0, 1.0)
    attention_gate = np.clip(attention_arr / (1.0 + attention_arr), 0.0, 1.0)
    scale = np.clip(propensity * (0.65 + activity + 0.35 * sharing_gate), 0.1, 5.0)
    lam = np.maximum(0.0, contact_rate_base * contact_rate_multiplier * scale)

    global_cap = int(min(12, sample_size_cap))
    k_by_agent = np.clip(rng.poisson(lam).astype(np.int64), 0, global_cap)
    if per_agent_sample_size is not None and per_agent_sample_size.shape[0] == n_agents:
        per_agent_cap = np.clip(np.rint(per_agent_sample_size).astype(np.int64), 1, global_cap)
        k_by_agent = np.minimum(k_by_agent, per_agent_cap)

    total_requested = int(np.sum(k_by_agent))
    if total_requested <= 0:
        return observed_share, ContactDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    event_agents = np.repeat(np.arange(n_agents, dtype=np.int64), k_by_agent)
    event_sources = np.full(total_requested, 2, dtype=np.int8)  # 0=household,1=recurring,2=one-shot
    source_draws = rng.random(total_requested)
    p_household = network.household_share[event_agents]
    p_recurring = network.recurring_share[event_agents]
    event_sources[source_draws < p_household] = 0
    recurring_mask = (event_sources == 2) & (source_draws < (p_household + p_recurring))
    event_sources[recurring_mask] = 1

    targets = np.full(total_requested, -1, dtype=np.int64)

    household_event_idx = np.flatnonzero(event_sources == 0)
    if household_event_idx.size > 0:
        household_agents = event_agents[household_event_idx]
        household_counts = network.household_neighbor_counts[household_agents]
        valid_household = household_counts > 0
        if np.any(valid_household):
            valid_idx = household_event_idx[valid_household]
            valid_agents = household_agents[valid_household]
            base = network.household_neighbors_offsets[valid_agents]
            draw = (rng.random(valid_idx.size) * household_counts[valid_household]).astype(np.int64)
            targets[valid_idx] = network.household_neighbors_flat[base + draw]

    recurring_event_idx = np.flatnonzero(event_sources == 1)
    if recurring_event_idx.size > 0:
        recurring_agents = event_agents[recurring_event_idx]
        recurring_counts = network.core_neighbor_counts[recurring_agents]
        valid_recurring = recurring_counts > 0
        if np.any(valid_recurring):
            valid_idx = recurring_event_idx[valid_recurring]
            valid_agents = recurring_agents[valid_recurring]
            base = network.core_neighbors_offsets[valid_agents]
            draw = (rng.random(valid_idx.size) * recurring_counts[valid_recurring]).astype(np.int64)
            targets[valid_idx] = network.core_neighbors_flat[base + draw]

    one_shot_event_idx = np.flatnonzero(event_sources == 2)
    if one_shot_event_idx.size > 0:
        one_shot_agents = event_agents[one_shot_event_idx]
        bridge_draw = rng.random(one_shot_event_idx.size)
        local_mask = bridge_draw < (1.0 - network.bridge_prob)
        global_mask = ~local_mask

        if np.any(local_mask):
            local_idx = one_shot_event_idx[local_mask]
            local_agents = one_shot_agents[local_mask]
            local_codes = network.locality_code_of_agent[local_agents]
            local_sizes = network.locality_member_counts[local_codes]
            valid_local = local_sizes > 1
            if np.any(valid_local):
                valid_idx = local_idx[valid_local]
                valid_agents = local_agents[valid_local]
                valid_codes = local_codes[valid_local]
                starts = network.locality_members_offsets[valid_codes]
                self_pos = network.locality_pos_in_members[valid_agents]
                draw = (rng.random(valid_idx.size) * (local_sizes[valid_local] - 1)).astype(np.int64)
                adjusted = draw + (draw >= self_pos).astype(np.int64)
                targets[valid_idx] = network.locality_members_flat[starts + adjusted]

        if np.any(global_mask):
            global_idx = one_shot_event_idx[global_mask]
            global_agents = one_shot_agents[global_mask]
            if n_agents > 1:
                draw = rng.integers(0, n_agents - 1, size=global_idx.size, dtype=np.int64)
                draw += (draw >= global_agents).astype(np.int64)
                targets[global_idx] = draw

    valid_contacts = targets >= 0
    total_contacts = int(np.sum(valid_contacts))
    if total_contacts <= 0:
        return observed_share, ContactDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    valid_agents = event_agents[valid_contacts]
    valid_targets = targets[valid_contacts]
    valid_sources = event_sources[valid_contacts]

    household_contacts = int(np.sum(valid_sources == 0))
    recurring_contacts = int(np.sum(valid_sources == 1))
    one_shot_contacts = int(np.sum(valid_sources == 2))
    cross_locality_contacts = int(
        np.sum(network.locality_code_of_agent[valid_targets] != network.locality_code_of_agent[valid_agents])
    )

    seen_mask = rng.random(total_contacts) <= attention_gate[valid_agents]
    if np.any(seen_mask):
        seen_agents = valid_agents[seen_mask]
        seen_targets = valid_targets[seen_mask]
        seen_values = protection_index[seen_targets]
        seen_counts = np.bincount(seen_agents, minlength=n_agents).astype(np.int64)
        seen_sums = np.bincount(seen_agents, weights=seen_values, minlength=n_agents).astype(np.float64)
        has_seen = seen_counts > 0
        observed_share[has_seen] = seen_sums[has_seen] / seen_counts[has_seen]

    denom = float(total_contacts)
    diagnostics = ContactDiagnostics(
        mean_contacts=(float(total_contacts) / float(max(n_agents, 1))),
        recurring_share=float(recurring_contacts) / denom,
        household_share=float(household_contacts) / denom,
        one_shot_share=float(one_shot_contacts) / denom,
        cross_locality_share=float(cross_locality_contacts) / denom,
    )
    return np.clip(observed_share, 0.0, 1.0), diagnostics
