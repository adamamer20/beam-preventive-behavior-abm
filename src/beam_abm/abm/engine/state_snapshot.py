"""State snapshot, restore, and init-modifier methods for VaccineBehaviourModel."""

from __future__ import annotations

import numpy as np
import polars as pl

from beam_abm.abm.engine.types import ModelStateSnapshot, TimingBreakdown
from beam_abm.abm.state_schema import CONSTRUCT_BOUNDS
from beam_abm.abm.state_update import compute_observed_norm_exposure


class VaccineBehaviourModelStateSnapshotMixin:
    def update_baseline_expected_obs_share_anchor(self, observed_share: np.ndarray, *, channel: str) -> None:
        """Update expected observed-share anchor during burn-in.

        Uses an online running mean over burn-in substeps so descriptive-norm
        share-hat anchors are calibrated to the model's own no-shock exposure
        manifold before evaluation starts.
        """
        observed = np.asarray(observed_share, dtype=np.float64)
        if observed.ndim != 1 or observed.size == 0:
            return
        if not np.isfinite(observed).all():
            return
        observed = np.clip(observed, 0.0, 1.0)

        if channel == "vax":
            anchor = self._baseline_expected_obs_share
            updates = int(self._baseline_expected_obs_share_updates)
            attr_anchor = "_baseline_expected_obs_share"
            attr_updates = "_baseline_expected_obs_share_updates"
        elif channel == "npi":
            anchor = self._baseline_expected_obs_share_npi
            updates = int(self._baseline_expected_obs_share_npi_updates)
            attr_anchor = "_baseline_expected_obs_share_npi"
            attr_updates = "_baseline_expected_obs_share_npi_updates"
        else:
            msg = f"Unsupported anchor channel '{channel}'."
            raise ValueError(msg)

        if anchor is None or anchor.shape != observed.shape or (not np.isfinite(anchor).all()):
            new_anchor = observed.copy()
            setattr(self, attr_anchor, new_anchor)
            setattr(self, attr_updates, 1)
            self._sync_observed_norm_anchor_bundle(channel=channel, anchor_share=new_anchor)
            return

        new_updates = updates + 1
        alpha = 1.0 / float(new_updates)
        updated_anchor = np.clip(anchor + alpha * (observed - anchor), 0.0, 1.0)
        setattr(self, attr_anchor, updated_anchor)
        setattr(self, attr_updates, new_updates)
        self._sync_observed_norm_anchor_bundle(channel=channel, anchor_share=updated_anchor)

    def _sync_observed_norm_anchor_bundle(self, *, channel: str, anchor_share: np.ndarray) -> None:
        """Keep observed-norm anchors internally consistent during burn-in.

        When ``obs_ref`` is recalibrated, the corresponding ``share_hat_0`` and
        exposure anchor must be recalibrated with the same threshold/slope to
        avoid systematic drift from anchor mismatch.
        """
        share = np.asarray(anchor_share, dtype=np.float64)
        if share.ndim != 1 or share.size == 0:
            return
        if not np.isfinite(share).all():
            return
        share = np.clip(share, 0.0, 1.0)

        n_agents = int(share.size)
        if "norm_threshold_theta" in self._population_t0.columns:
            threshold = self._population_t0["norm_threshold_theta"].cast(pl.Float64).to_numpy().astype(np.float64)
        else:
            threshold = np.full(n_agents, 0.5, dtype=np.float64)
        if threshold.shape != share.shape:
            threshold = np.full(n_agents, 0.5, dtype=np.float64)
        threshold = np.nan_to_num(threshold, nan=0.5, posinf=0.5, neginf=0.5)
        threshold = np.clip(
            threshold,
            float(self.cfg.constants.norm_threshold_min),
            float(self.cfg.constants.norm_threshold_max),
        )

        exposure = compute_observed_norm_exposure(
            observed_share=share,
            threshold=threshold,
            slope=float(self.cfg.constants.norm_response_slope),
        )
        if channel == "vax":
            self._baseline_share_hat = share.copy()
            self._baseline_norm_exposure = np.clip(exposure, 0.0, 1.0)
        elif channel == "npi":
            self._baseline_share_hat_npi = share.copy()
            self._baseline_norm_exposure_npi = np.clip(exposure, 0.0, 1.0)
        else:
            msg = f"Unsupported anchor channel '{channel}'."
            raise ValueError(msg)

    def snapshot_state(self) -> ModelStateSnapshot:
        """Capture continuation-relevant microstate for warm starts.

        Returns
        -------
        ModelStateSnapshot
            Typed snapshot of mutable model state.
        """
        random_state: dict[str, object] | None = None
        bit_generator = getattr(self.random, "bit_generator", None)
        if bit_generator is not None:
            random_state = dict(bit_generator.state)

        latest_outcomes: dict[str, np.ndarray] | None = None
        if self._latest_outcomes is not None:
            latest_outcomes = {
                name: values.astype(np.float64, copy=True) for name, values in self._latest_outcomes.items()
            }

        return ModelStateSnapshot(
            agents_df=self.agents.df.clone(),
            population_t0=self._population_t0.clone(),
            outcome_t0_means=dict(self._outcome_t0_means),
            current_tick=int(self._current_tick),
            community_incidence=dict(self._community_incidence),
            prev_community_incidence=dict(self._prev_community_incidence),
            importation_peak_reference=(
                float(self._importation_peak_reference) if self._importation_peak_reference is not None else None
            ),
            importation_peak_reference_by_community=(
                dict(self._importation_peak_reference_by_community)
                if self._importation_peak_reference_by_community is not None
                else None
            ),
            target_mask=None if self._target_mask is None else self._target_mask.astype(bool, copy=True),
            seed_mask=None if self._seed_mask is None else self._seed_mask.astype(bool, copy=True),
            graph_distance_from_seed=(
                None
                if self._graph_distance_from_seed is None
                else self._graph_distance_from_seed.astype(np.int32, copy=True)
            ),
            latest_outcomes=latest_outcomes,
            latest_social_similarity_mean=float(self._latest_social_similarity_mean),
            latest_contact_diagnostics=self.latest_contact_diagnostics,
            random_state=random_state,
        )

    def restore_state(
        self,
        state: ModelStateSnapshot | dict[str, object],
        *,
        reset_trajectories: bool = True,
    ) -> None:
        """Restore continuation-relevant microstate from snapshot.

        Parameters
        ----------
        state : ModelStateSnapshot or dict[str, object]
            Snapshot payload from :meth:`snapshot_state`.
        reset_trajectories : bool
            If True, clear trajectory accumulators and reset tick to 0.

        Raises
        ------
        ValueError
            If snapshot schema is invalid.
        """
        state_obj = state if isinstance(state, ModelStateSnapshot) else ModelStateSnapshot.model_validate(state)

        restored_agents = state_obj.agents_df.clone()
        if ("norm_share_hat_vax" not in restored_agents.columns) or (
            "social_norms_descriptive_npi" in restored_agents.columns
            and "norm_share_hat_npi" not in restored_agents.columns
        ):
            restored_agents = self._ensure_norm_share_hat_state(restored_agents)
        self.agents.df = restored_agents

        restored_t0 = state_obj.population_t0.clone()
        if ("norm_share_hat_vax" not in restored_t0.columns) or (
            "social_norms_descriptive_npi" in restored_t0.columns and "norm_share_hat_npi" not in restored_t0.columns
        ):
            restored_t0 = self._ensure_norm_share_hat_state(restored_t0)
        self._population_t0 = restored_t0
        self._refresh_state_ref_attractor()
        self._baseline_observed_norm_share = self._compute_baseline_observed_norm_share(self._population_t0)
        self._baseline_observed_norm_share_npi = self._compute_baseline_observed_norm_share_npi(self._population_t0)
        self._baseline_share_hat = self._extract_baseline_share_hat(self._population_t0)
        self._baseline_share_hat_npi = self._extract_baseline_share_hat_npi(self._population_t0)
        self._baseline_expected_obs_share = None  # will lazy-init from first tick
        self._baseline_expected_obs_share_npi = None  # will lazy-init from first tick
        self._baseline_expected_obs_share_updates = 0
        self._baseline_expected_obs_share_npi_updates = 0
        self._baseline_descriptive_norm, self._baseline_norm_exposure = self._compute_baseline_norm_anchors(
            self._population_t0
        )
        self._baseline_descriptive_norm_npi, self._baseline_norm_exposure_npi = self._compute_baseline_norm_anchors_npi(
            self._population_t0
        )
        self._outcome_t0_means = {str(k): float(v) for k, v in state_obj.outcome_t0_means.items()}
        self._community_incidence = {str(k): float(v) for k, v in state_obj.community_incidence.items()}
        self._prev_community_incidence = {str(k): float(v) for k, v in state_obj.prev_community_incidence.items()}
        self._importation_peak_reference = (
            float(state_obj.importation_peak_reference) if state_obj.importation_peak_reference is not None else None
        )
        self._importation_peak_reference_by_community = (
            {str(k): float(v) for k, v in state_obj.importation_peak_reference_by_community.items()}
            if state_obj.importation_peak_reference_by_community is not None
            else None
        )

        target_mask = state_obj.target_mask
        self._target_mask = None if target_mask is None else np.asarray(target_mask, dtype=bool)

        seed_mask = state_obj.seed_mask
        self._seed_mask = None if seed_mask is None else np.asarray(seed_mask, dtype=bool)

        graph_distance = state_obj.graph_distance_from_seed
        self._graph_distance_from_seed = None if graph_distance is None else np.asarray(graph_distance, dtype=np.int32)

        latest_outcomes_raw = state_obj.latest_outcomes
        if latest_outcomes_raw is None:
            self._latest_outcomes = None
        else:
            self._latest_outcomes = {
                str(name): np.asarray(values, dtype=np.float64) for name, values in latest_outcomes_raw.items()
            }

        self._latest_social_similarity_mean = float(state_obj.latest_social_similarity_mean)
        self.latest_contact_diagnostics = state_obj.latest_contact_diagnostics

        random_state = state_obj.random_state
        if random_state is not None:
            bit_generator = getattr(self.random, "bit_generator", None)
            if bit_generator is None:
                msg = "Model RNG does not expose bit_generator; cannot restore RNG state."
                raise ValueError(msg)
            bit_generator.state = dict(random_state)

        if reset_trajectories:
            self._current_tick = 0
            self._timing = TimingBreakdown()
            self._outcome_records.clear()
            self._mediator_records.clear()
            self._signal_records.clear()
            self._outcomes_by_group_records.clear()
            self._mediators_by_group_records.clear()
            self._latest_signals_realised = []
            self._debug_trace_records = []
            self._latest_outcomes_by_group = []
            self._latest_mediators_by_group = []
            self._saturation_records.clear()
            self._incidence_records.clear()
            self._distance_trajectory_records.clear()
            self._importation_peak_reference = None
            self._importation_peak_reference_by_community = None
        else:
            self._current_tick = int(state_obj.current_tick)

    def apply_init_modifier(
        self,
        *,
        kind: str,
        delta_norm: float = 0.0,
        target: str = "global",
        random_k: int = 0,
        group_id: str | None = None,
        random_seed: int = 42,
        channel: str = "both",
    ) -> dict[str, float | int | str]:
        """Apply tick-0 initial-condition modifier for basin tests.

        Parameters
        ----------
        kind : str
            Modifier kind, one of ``{"low", "high"}``.
        delta_norm : float
            Additive norm shift used when ``kind="high"``.
        target : str
            Targeting mode: ``global``, ``top_degree``, ``random_k``, ``group_id``.
        random_k : int
            Number of agents for ``random_k`` targeting.
        group_id : str or None
            Group identifier for ``group_id`` targeting.
        random_seed : int
            Seed for deterministic random subset selection.
        channel : str
            Which norm channel(s) to shift: ``both`` (default) or
            ``descriptive``.

        Returns
        -------
        dict[str, float | int | str]
            Summary of the modifier effect for quality checks.
        """
        if kind not in {"low", "high"}:
            msg = f"Unsupported init modifier kind '{kind}'."
            raise ValueError(msg)
        if kind == "low":
            return {
                "kind": kind,
                "target": target,
                "channel": channel,
                "n_targeted": 0,
                "delta_norm": 0.0,
                "target_mean_before": 0.0,
                "target_mean_after": 0.0,
            }

        if delta_norm <= 0.0:
            msg = "delta_norm must be > 0 for kind='high'."
            raise ValueError(msg)

        n_agents = self.agents.df.height
        if n_agents <= 0:
            msg = "Cannot apply init modifier to an empty population."
            raise ValueError(msg)

        mask: np.ndarray
        if target == "global":
            mask = np.ones(n_agents, dtype=bool)
        elif target == "top_degree":
            if self.network is None:
                msg = "top_degree targeting requires an active network."
                raise ValueError(msg)
            if self.network.core_degree.shape[0] != n_agents:
                msg = "Network degree vector length does not match agent population."
                raise ValueError(msg)
            threshold = np.percentile(self.network.core_degree.astype(np.float64), 90.0)
            mask = self.network.core_degree.astype(np.float64) >= threshold
        elif target == "random_k":
            if random_k <= 0:
                msg = "random_k targeting requires random_k > 0."
                raise ValueError(msg)
            k = int(min(random_k, n_agents))
            rng = np.random.default_rng(random_seed)
            selected = rng.choice(np.arange(n_agents, dtype=np.int64), size=k, replace=False)
            mask = np.zeros(n_agents, dtype=bool)
            mask[selected] = True
        elif target == "group_id":
            if group_id is None:
                msg = "group_id targeting requires non-null group_id."
                raise ValueError(msg)
            if "locality_id" not in self.agents.df.columns:
                msg = "group_id targeting requires 'locality_id' column in agent state."
                raise ValueError(msg)
            locality = self.agents.df["locality_id"].cast(pl.Utf8).to_numpy()
            mask = locality == str(group_id)
        else:
            msg = f"Unsupported init target '{target}'."
            raise ValueError(msg)

        n_targeted = int(np.count_nonzero(mask))
        if n_targeted <= 0:
            msg = f"Init modifier target '{target}' selected zero agents."
            raise ValueError(msg)

        if channel not in {"both", "descriptive"}:
            msg = f"Unsupported init channel '{channel}'. Expected one of {{'both','descriptive'}}."
            raise ValueError(msg)

        # social_norms_vax_avg is a derived composite recomputed from the channels,
        # so shift the channels directly for basin initialisation tests.
        channel_cols: list[str] = []
        if channel in {"both", "descriptive"}:
            channel_cols.append("social_norms_descriptive_vax")

        missing = [c for c in channel_cols if c not in self.agents.df.columns]
        if missing:
            msg = f"Init modifier requires norm channel columns in agent state: {missing}"
            raise ValueError(msg)

        before_vals: list[np.ndarray] = []
        after_vals: list[np.ndarray] = []
        updated_cols: list[pl.Series] = []
        for col in channel_cols:
            values = self.agents.df[col].cast(pl.Float64).to_numpy().astype(np.float64)
            before_vals.append(values[mask])
            lower, upper = CONSTRUCT_BOUNDS.get(col, (-np.inf, np.inf))
            shifted = values.copy()
            shifted[mask] = np.clip(shifted[mask] + float(delta_norm), float(lower), float(upper))
            after_vals.append(shifted[mask])
            updated_cols.append(pl.Series(name=col, values=shifted))

        before = float(np.mean(np.concatenate(before_vals))) if before_vals else 0.0
        after = float(np.mean(np.concatenate(after_vals))) if after_vals else 0.0

        self.agents.df = self._ensure_norm_share_hat_state(self.agents.df.with_columns(updated_cols))
        self._population_t0 = self.agents.df.clone()
        self._refresh_state_ref_attractor()
        self._baseline_observed_norm_share = self._compute_baseline_observed_norm_share(self._population_t0)
        self._baseline_observed_norm_share_npi = self._compute_baseline_observed_norm_share_npi(self._population_t0)
        self._baseline_share_hat = self._extract_baseline_share_hat(self._population_t0)
        self._baseline_share_hat_npi = self._extract_baseline_share_hat_npi(self._population_t0)
        self._baseline_expected_obs_share = None  # will lazy-init from first tick
        self._baseline_expected_obs_share_npi = None  # will lazy-init from first tick
        self._baseline_expected_obs_share_updates = 0
        self._baseline_expected_obs_share_npi_updates = 0
        self._baseline_descriptive_norm, self._baseline_norm_exposure = self._compute_baseline_norm_anchors(
            self._population_t0
        )
        self._baseline_descriptive_norm_npi, self._baseline_norm_exposure_npi = self._compute_baseline_norm_anchors_npi(
            self._population_t0
        )

        return {
            "kind": kind,
            "target": target,
            "channel": channel,
            "n_targeted": n_targeted,
            "delta_norm": float(delta_norm),
            "target_mean_before": before,
            "target_mean_after": after,
        }
