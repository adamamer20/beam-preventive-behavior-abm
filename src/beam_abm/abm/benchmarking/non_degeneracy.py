"""Non-degeneracy domain aggregation for benchmarking gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import GateCheck, GateResult, NonDegeneracyDomainSummary, NonDegeneracyReport


@dataclass(slots=True, frozen=True)
class _NonDegeneracyDomainRule:
    """Internal matcher describing one non-degeneracy diagnostic domain."""

    name: str
    description: str
    prefixes: tuple[str, ...]
    exact_names: tuple[str, ...] = ()


def _match_check_name(
    check_name: str,
    *,
    prefixes: tuple[str, ...],
    exact_names: tuple[str, ...],
) -> bool:
    if check_name in exact_names:
        return True
    return any(check_name.startswith(prefix) for prefix in prefixes)


_NON_DEGENERACY_DOMAIN_RULES: tuple[_NonDegeneracyDomainRule, ...] = (
    _NonDegeneracyDomainRule(
        name="distribution_and_dispersion",
        description=("Cross-sectional distribution checks to detect variance collapse or baseline mismatch."),
        prefixes=(
            "marginal_",
            "engine_quantiles_",
            "propensity_hist_distance_",
            "propensity_jsd_",
            "outcome_marginal_primary_",
        ),
    ),
    _NonDegeneracyDomainRule(
        name="frozen_or_inert_dynamics",
        description="Detects stuck states and unintended drift in quiet settings.",
        prefixes=("quiet_stationarity_", "quiet_no_drift_"),
        exact_names=("frozen_state_no_comm", "context_separation"),
    ),
    _NonDegeneracyDomainRule(
        name="boundedness_and_stability",
        description=("Detects blow-up, bound lock-in, or persistent instability in dynamic trajectories."),
        prefixes=(
            "bounded_",
            "saturation_",
            "anti_saturation_",
            "regularity_persistent_stable_under_stress_",
            "regularity_persistent_mediator_variance_growth_",
            "regularity_persistent_plateau_or_stable_region_",
            "regularity_shock_recovery_",
        ),
    ),
    _NonDegeneracyDomainRule(
        name="policy_sensitivity",
        description=("Detects loss of response to levers or wrong directional comparative statics."),
        prefixes=(
            "partial_gradient_",
            "pe_ref_sign_",
            "pe_ref_closeness_",
            "pe_ref_responsiveness_",
            "perturbation_ref_",
            "regularity_directional_tplus1_",
            "dynamic_directional_tplus1_",
            "dynamic_directional_response_",
            "dynamic_directional_window_",
            "dose_response_",
        ),
    ),
    _NonDegeneracyDomainRule(
        name="network_structure_and_order_sensitivity",
        description=(
            "Proxy checks for structural degeneracy and loop/order "
            "sensitivity via network geometry and jitter robustness."
        ),
        prefixes=(
            "feature_space_tie_rate",
            "nearest_neighbor_distance_distribution",
            "nn_zero_distance_mass_",
            "homophily_sanity_",
            "artifact_firewall_",
            "mixing_robustness",
            "jitter_sensitivity_",
            "network_sensitivity_jitter",
        ),
    ),
    _NonDegeneracyDomainRule(
        name="stochastic_dispersion",
        description=("Checks that stochastic runs retain dispersion without collapsing to deterministic clones."),
        prefixes=(
            "stochastic_variance_bounded_",
            "duplicate_clone_dispersion_",
            "deterministic_same_seed_identical_",
            "deterministic_seed_invariance_no_comm_",
        ),
    ),
)


def build_non_degeneracy_report(gates: tuple[GateResult, ...]) -> NonDegeneracyReport:
    """Aggregate existing gate checks into a compact non-degeneracy report."""
    from .core import NonDegeneracyDomainSummary, NonDegeneracyReport

    domain_rows: list[NonDegeneracyDomainSummary] = []
    for rule in _NON_DEGENERACY_DOMAIN_RULES:
        matched: list[tuple[str, GateCheck]] = []
        for gate in gates:
            for check in gate.checks:
                if _match_check_name(
                    check.name,
                    prefixes=rule.prefixes,
                    exact_names=rule.exact_names,
                ):
                    matched.append((gate.gate_name, check))

        n_checks = len(matched)
        n_passed = sum(1 for _, check in matched if check.passed)
        failing = [f"{gate_name}.{check.name}" for gate_name, check in matched if not check.passed]
        if n_checks == 0:
            status = "not_evaluated"
        elif n_passed == n_checks:
            status = "passed"
        else:
            status = "failed"

        domain_rows.append(
            NonDegeneracyDomainSummary(
                domain=rule.name,
                description=rule.description,
                status=status,
                n_checks=n_checks,
                n_passed=n_passed,
                failing_checks=failing,
            )
        )

    n_failed = sum(1 for row in domain_rows if row.status == "failed")
    n_incomplete = sum(1 for row in domain_rows if row.status == "not_evaluated")
    if n_failed > 0:
        status = "failed"
    elif n_incomplete > 0:
        status = "incomplete"
    else:
        status = "passed"
    return NonDegeneracyReport(
        status=status,
        n_domains=len(domain_rows),
        n_failed_domains=n_failed,
        n_incomplete_domains=n_incomplete,
        domains=domain_rows,
    )
