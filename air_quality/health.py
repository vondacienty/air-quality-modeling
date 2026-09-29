"""Health impact assessment relative to a baseline scenario."""

from __future__ import annotations

import math
from itertools import product

from .gaussian import _check_finite, _is_number

__all__ = [
    "aggregate",
    "aggregate_correlated",
    "any_receptor_probability",
    "assess",
    "audit_delta",
    "audit_reconcile",
    "audit_trend_batch_alerts",
    "audit_trend_batches",
    "audit_trend_grid",
    "audit_trend_report",
    "audit_trend_summary",
    "audit_trend_turns",
    "audit_window_report",
    "audit_window_trend",
    "at_least_count_probability",
    "exceedance_probability",
    "excess_interval",
    "excess_warning",
    "expected_excess",
    "level_probability",
    "max_level_probability",
    "max_level_quantile",
    "policy_recommend",
    "quantile",
    "receptor_count_cvar",
    "receptor_count_event_probability",
    "receptor_count_interval",
    "receptor_count_probability",
    "receptor_count_quantile",
    "receptor_count_run_policy_batch",
    "receptor_count_run_warning",
    "receptor_count_share",
    "receptor_excess",
    "receptor_excess_cvar",
    "receptor_excess_interval",
    "receptor_excess_quantile",
    "receptor_excess_warning",
    "receptor_expected_excess",
    "receptor_level_count_cvar",
    "receptor_level_count_entropy",
    "receptor_level_count_interval",
    "receptor_level_count_probability",
    "receptor_level_count_quantile",
    "receptor_level_count_rise_distribution",
    "receptor_level_count_run_distribution",
    "receptor_level_count_run_quantile",
    "receptor_level_count_share",
    "receptor_level_count_transition",
    "receptor_level_count_warning",
    "receptor_level_event_probability",
    "receptor_level_interval",
    "receptor_level_path_probability",
    "receptor_level_probability",
    "receptor_level_quantile",
    "receptor_level_rise_distribution",
    "receptor_level_run_cvar",
    "receptor_level_run_distribution",
    "receptor_level_run_entropy",
    "receptor_level_run_quantile",
    "receptor_level_run_stats",
    "receptor_level_transition",
    "receptor_mitigation_frontier",
    "receptor_mitigation_plan",
    "receptor_quantile",
    "receptor_risk_interval",
    "receptor_risk_quantile",
    "receptor_risk_share",
    "risk_contribution",
    "risk_excess_cvar",
    "risk_excess_quantile",
    "risk_interval",
    "risk_probability",
    "risk_share",
    "robust_policy_action_audit",
    "robust_policy_action_audit_chain",
    "robust_policy_action_batches",
    "robust_policy_action_checkpoint",
    "robust_policy_action_commit",
    "robust_policy_action_confirm",
    "robust_policy_action_progress",
    "robust_policy_action_restore",
    "robust_policy_action_resume",
    "robust_policy_action_waves",
    "robust_policy_actions",
    "robust_policy_change_summary",
    "robust_policy_grid",
    "robust_policy_grid_regions",
    "robust_policy_grid_turns",
    "robust_policy_joint",
    "robust_policy_lineage",
    "robust_policy_priority",
    "robust_policy_region_flow",
    "robust_policy_report",
    "robust_policy_sensitivity",
    "robust_policy_summary",
    "sensitivity",
]


def _validate_matrix(
    name: str, matrix: object, non_negative: bool = True
) -> list[list[float]]:
    if not isinstance(matrix, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(matrix).__name__}")
    if len(matrix) == 0:
        raise ValueError(f"{name} must not be empty")
    width: int | None = None
    validated = []
    for k, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                f"{name}[{k}] must be a list or tuple, got {type(row).__name__}"
            )
        if len(row) == 0:
            raise ValueError(f"{name}[{k}] must not be empty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"{name}[{k}] must have {width} elements, got {len(row)}"
            )
        values = []
        for i, item in enumerate(row):
            if not _is_number(item):
                raise TypeError(
                    f"{name}[{k}][{i}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"{name}[{k}][{i}]", value)
            if non_negative and value < 0:
                raise ValueError(f"{name}[{k}][{i}] must be >= 0, got {value!r}")
            values.append(value)
        validated.append(values)
    return validated


def _validate_vector(name: str, values: object, expected_length: int) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} length must equal receptor count {expected_length}, "
            f"got {len(values)}"
        )
    validated = []
    for i, item in enumerate(values):
        if not _is_number(item):
            raise TypeError(
                f"{name}[{i}] must be an int or float, got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"{name}[{i}]", value)
        if value < 0:
            raise ValueError(f"{name}[{i}] must be >= 0, got {value!r}")
        validated.append(value)
    return validated


def _validate_cube(name: str, cube: object) -> list[list[list[float]]]:
    if not isinstance(cube, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(cube).__name__}")
    if len(cube) == 0:
        raise ValueError(f"{name} must not be empty")
    width: int | None = None
    validated = []
    for k, matrix in enumerate(cube):
        if not isinstance(matrix, (list, tuple)):
            raise TypeError(
                f"{name}[{k}] must be a list or tuple, got {type(matrix).__name__}"
            )
        if len(matrix) == 0:
            raise ValueError(f"{name}[{k}] must not be empty")
        if width is None:
            width = len(matrix)
        elif len(matrix) != width:
            raise ValueError(
                f"{name}[{k}] must have {width} rows, got {len(matrix)}"
            )
        rows = []
        for i, row in enumerate(matrix):
            if not isinstance(row, (list, tuple)):
                raise TypeError(
                    f"{name}[{k}][{i}] must be a list or tuple, "
                    f"got {type(row).__name__}"
                )
            if len(row) != width:
                raise ValueError(
                    f"{name}[{k}][{i}] must have {width} elements, "
                    f"got {len(row)}"
                )
            values = []
            for j, item in enumerate(row):
                if not _is_number(item):
                    raise TypeError(
                        f"{name}[{k}][{i}][{j}] must be an int or float, "
                        f"got {type(item).__name__}"
                    )
                value = float(item)
                _check_finite(f"{name}[{k}][{i}][{j}]", value)
                if j > i and value != 0.0:
                    raise ValueError(
                        f"{name}[{k}][{i}][{j}] must be 0 above the diagonal, "
                        f"got {value!r}"
                    )
                values.append(value)
            rows.append(values)
        validated.append(rows)
    return validated


def assess(
    C: list[list[float]] | tuple[tuple[float, ...], ...],
    U: list[list[float]] | tuple[tuple[float, ...], ...],
    P: list[float] | tuple[float, ...],
    beta: float,
    baseline: int = 0,
) -> tuple[
    list[list[float]],
    list[float],
    list[list[float]],
    list[float],
]:
    """Assess health impacts of scenarios relative to a baseline scenario.

    C: non-empty scenario x receptor matrix of concentrations; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite and >= 0.
    U: matrix of mutually independent standard uncertainties with the same
        shape and constraints as ``C``.
    P: non-empty list or tuple of populations, one per receptor, in
        receptor order; each must be finite and >= 0.
    beta: cases per person per concentration unit; finite and >= 0.
    baseline: index of the baseline scenario; a non-bool int in
        ``[0, len(C))`` (default 0).
    Returns ``(H, S, W, Q)`` where, with
    ``D[k][i] = C[k][i] - C[baseline][i]`` and
    ``E[k][i] = sqrt(U[k][i]**2 + U[baseline][i]**2)``:

    * ``H[k][i] = P[i] * beta * D[k][i]``,
    * ``W[k][i] = P[i] * beta * E[k][i]``,
    * ``S[k] = fsum(H[k])``,
    * ``Q[k] = hypot(*W[k])``, equal to ``sqrt(fsum(W[k][i] ** 2))``
      but without intermediate overflow.

    The baseline row of ``H`` and ``W`` (and the corresponding ``S`` and
    ``Q`` entries) are exactly 0.0. All results are plain lists of floats
    in input order and are not rounded.
    """
    concentrations = _validate_matrix("C", C)
    uncertainties = _validate_matrix("U", U)

    if len(uncertainties) != len(concentrations):
        raise ValueError(
            f"C and U must have the same number of scenarios, "
            f"got {len(concentrations)} and {len(uncertainties)}"
        )
    n_receptors = len(concentrations[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"U[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    populations = _validate_vector("P", P, n_receptors)

    if not _is_number(beta):
        raise TypeError(f"beta must be an int or float, got {type(beta).__name__}")
    beta = float(beta)
    _check_finite("beta", beta)
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta!r}")

    if not isinstance(baseline, int) or isinstance(baseline, bool):
        raise TypeError(
            f"baseline must be an int, got {type(baseline).__name__}"
        )
    if not 0 <= baseline < len(concentrations):
        raise ValueError(
            f"baseline must be in [0, {len(concentrations)}), got {baseline}"
        )

    base_c = concentrations[baseline]
    base_u = uncertainties[baseline]

    H: list[list[float]] = []
    W: list[list[float]] = []
    for k in range(len(concentrations)):
        h_row: list[float] = []
        w_row: list[float] = []
        if k == baseline:
            h_row = [0.0] * n_receptors
            w_row = [0.0] * n_receptors
        else:
            for i in range(n_receptors):
                d = concentrations[k][i] - base_c[i]
                e = math.sqrt(uncertainties[k][i] ** 2 + base_u[i] ** 2)
                scale = populations[i] * beta
                h_row.append(scale * d)
                w_row.append(scale * e)
        H.append(h_row)
        W.append(w_row)

    S = [math.fsum(row) for row in H]
    Q = [math.hypot(*row) for row in W]

    return H, S, W, Q


def aggregate(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float], float, float]:
    """Aggregate scenario health impacts across scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    weights: ``None`` (the default) or a list or tuple with one finite,
        non-negative entry per scenario. With ``None`` every scenario has
        weight ``1 / K``; otherwise the weights are normalized by their
        positive sum.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(M, R, lower, upper, total, total_spread)`` where, per
    receptor ``i``:

    * ``M[i] = fsum(w[k] * H[k][i])``,
    * ``V[i] = fsum(w[k] * ((H[k][i] - M[i]) ** 2 + W[k][i] ** 2))``,
    * ``R[i] = z * sqrt(V[i])``,
    * ``lower[i] = M[i] - R[i]``, ``upper[i] = M[i] + R[i]``,

    and ``total = fsum(M)``, ``total_spread = hypot(*R)``. The first four
    results are plain N-long lists of floats and the last two are floats,
    all unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    M: list[float] = []
    V: list[float] = []
    for i in range(n_receptors):
        mean = math.fsum(w[k] * impacts[k][i] for k in range(n_scenarios))
        variance = math.fsum(
            w[k] * ((impacts[k][i] - mean) ** 2 + uncertainties[k][i] ** 2)
            for k in range(n_scenarios)
        )
        M.append(mean)
        V.append(variance)

    R = [z * math.sqrt(variance) for variance in V]
    lower = [M[i] - R[i] for i in range(n_receptors)]
    upper = [M[i] + R[i] for i in range(n_receptors)]
    total = math.fsum(M)
    total_spread = math.hypot(*R)

    return M, R, lower, upper, total, total_spread


def aggregate_correlated(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    F: list[list[list[float]]] | tuple[tuple[tuple[float, ...], ...], ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float], float, float]:
    """Aggregate scenario health impacts with correlated receptor errors.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    F: K x N x N array of lower-triangular factor matrices, one per
        scenario; each container level must be a list or tuple, every
        matrix must be non-empty, square and share one receptor count with
        ``H``; each value must be finite and entries above the diagonal
        must be exactly 0.
    weights: ``None`` (the default) or a list or tuple with one finite,
        non-negative entry per scenario. With ``None`` every scenario has
        weight ``1 / K``; otherwise the weights are normalized by their
        positive sum.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(M, R, lower, upper, T, total_spread)`` where, with
    ``Sigma[k][i][j] = fsum(F[k][i][r] * F[k][j][r] for r in range(N))``:

    * ``M[i] = fsum(w[k] * H[k][i])``,
    * ``V[i] = fsum(w[k] * ((H[k][i] - M[i]) ** 2 + Sigma[k][i][i]))``,
    * ``R[i] = z * sqrt(V[i])``,
    * ``lower[i] = M[i] - R[i]``, ``upper[i] = M[i] + R[i]``,
    * ``T = fsum(M)``,
    * ``VT = fsum(w[k] * ((fsum(H[k]) - T) ** 2
      + fsum(Sigma[k][i][j] for i in range(N) for j in range(N))))``,
    * ``total_spread = z * sqrt(VT)``.

    The first four results are plain N-long lists of floats and the last
    two are floats, all unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    factors = _validate_cube("F", F)

    if len(factors) != len(impacts):
        raise ValueError(
            f"H and F must have the same number of scenarios, "
            f"got {len(impacts)} and {len(factors)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, matrix in enumerate(factors):
        if len(matrix) != n_receptors:
            raise ValueError(
                f"F[{k}] must have {n_receptors} rows, got {len(matrix)}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    covariances: list[list[list[float]]] = []
    for k in range(n_scenarios):
        matrix = factors[k]
        sigma = [
            [
                math.fsum(matrix[i][r] * matrix[j][r] for r in range(n_receptors))
                for j in range(n_receptors)
            ]
            for i in range(n_receptors)
        ]
        covariances.append(sigma)

    M: list[float] = []
    V: list[float] = []
    for i in range(n_receptors):
        mean = math.fsum(w[k] * impacts[k][i] for k in range(n_scenarios))
        variance = math.fsum(
            w[k] * ((impacts[k][i] - mean) ** 2 + covariances[k][i][i])
            for k in range(n_scenarios)
        )
        M.append(mean)
        V.append(variance)

    R = [z * math.sqrt(variance) for variance in V]
    lower = [M[i] - R[i] for i in range(n_receptors)]
    upper = [M[i] + R[i] for i in range(n_receptors)]
    total = math.fsum(M)
    total_variance = math.fsum(
        w[k]
        * (
            (math.fsum(impacts[k]) - total) ** 2
            + math.fsum(
                covariances[k][i][j]
                for i in range(n_receptors)
                for j in range(n_receptors)
            )
        )
        for k in range(n_scenarios)
    )
    total_spread = z * math.sqrt(total_variance)

    return M, R, lower, upper, total, total_spread


def exceedance_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
) -> tuple[list[list[float]], list[float]]:
    """Compute exceedance probabilities of scenario impacts against thresholds.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    Returns ``(probabilities, expected)`` where, with
    ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * ``probabilities[k][j]`` is the probability that the total impact of
      scenario ``k`` exceeds ``thresholds[j]``:
      ``0.5 * erfc((thresholds[j] - mu[k]) / (sigma[k] * sqrt(2)))`` when
      ``sigma[k] > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu[k]``
      else ``0.0``;
    * ``expected[k] = fsum(probabilities[k])``.

    ``probabilities`` is a K x 3 list of lists of floats and ``expected``
    is a K-long list of floats, in input order and unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    probabilities: list[list[float]] = []
    for k in range(len(impacts)):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        row: list[float] = []
        for level in levels:
            if sigma > 0:
                row.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
            else:
                row.append(1.0 if level <= mu else 0.0)
        probabilities.append(row)

    expected = [math.fsum(row) for row in probabilities]

    return probabilities, expected


def risk_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], float]:
    """Aggregate scenario exceedance risks across scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(probabilities, expected)`` where, with
    ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * ``p[k][j] = 0.5 * erfc((thresholds[j] - mu[k]) / (sigma[k] * sqrt(2)))``
      when ``sigma[k] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= mu[k]`` else ``0.0``;
    * ``probabilities[j] = fsum(w[k] * p[k][j] for k in range(K))``;
    * ``expected = fsum(probabilities)``.

    ``probabilities`` is a 3-long list of floats and ``expected`` is a
    float, both unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        row: list[float] = []
        for level in levels:
            if sigma > 0:
                row.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
            else:
                row.append(1.0 if level <= mu else 0.0)
        per_scenario.append(row)

    probabilities = [
        math.fsum(w[k] * per_scenario[k][j] for k in range(n_scenarios))
        for j in range(3)
    ]
    expected = math.fsum(probabilities)

    return probabilities, expected


def expected_excess(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], list[float]]:
    """Aggregate expected threshold exceedance across scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(probabilities, excess)`` where, with
    ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * when ``sigma[k] > 0``, with ``a = (thresholds[j] - mu[k]) / sigma[k]``:
      ``p[k][j] = 0.5 * erfc(a / sqrt(2))`` and
      ``e[k][j] = sigma[k] * exp(-a * a / 2) / sqrt(2 * pi)
      + (mu[k] - thresholds[j]) * p[k][j]``;
    * when ``sigma[k] == 0``: ``p[k][j] = 1.0`` if
      ``thresholds[j] <= mu[k]`` else ``0.0``, and
      ``e[k][j] = max(mu[k] - thresholds[j], 0.0)``;
    * ``probabilities[j] = fsum(w[k] * p[k][j] for k in range(K))``;
    * ``excess[j] = fsum(w[k] * e[k][j] for k in range(K))``.

    Both results are 3-long lists of floats in threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    per_p: list[list[float]] = []
    per_e: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        p_row: list[float] = []
        e_row: list[float] = []
        for level in levels:
            if sigma > 0:
                a = (level - mu) / sigma
                p = 0.5 * math.erfc(a / math.sqrt(2))
                e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * p
            else:
                p = 1.0 if level <= mu else 0.0
                e = max(mu - level, 0.0)
            p_row.append(p)
            e_row.append(e)
        per_p.append(p_row)
        per_e.append(e_row)

    probabilities = [
        math.fsum(w[k] * per_p[k][j] for k in range(n_scenarios)) for j in range(3)
    ]
    excess = [
        math.fsum(w[k] * per_e[k][j] for k in range(n_scenarios)) for j in range(3)
    ]

    return probabilities, excess


def excess_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Weighted mean and interval for the expected threshold excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(mean, spread, lower, upper)`` where, with
    ``mu[k] = fsum(H[k])``, ``sigma[k] = hypot(*W[k])`` and
    ``d = mu[k] - thresholds[j]``:

    * when ``sigma[k] > 0``, with ``a = (thresholds[j] - mu[k]) / sigma[k]``,
      ``phi = exp(-a * a / 2) / sqrt(2 * pi)`` and
      ``p = 0.5 * erfc(a / sqrt(2))``:
      ``e[k][j] = sigma[k] * phi + d * p`` and
      ``s[k][j] = ((d * d + sigma[k] ** 2) * p + sigma[k] * d * phi)
      - e[k][j] ** 2``;
    * when ``sigma[k] == 0``: ``e[k][j] = max(d, 0.0)`` and
      ``s[k][j] = 0.0``;
    * ``mean[j] = fsum(w[k] * e[k][j] for k in range(K))``;
    * ``V[j] = fsum(w[k] * ((e[k][j] - mean[j]) ** 2 + s[k][j])
      for k in range(K))``;
    * ``spread[j] = z * sqrt(V[j])``;
    * ``lower[j] = max(0.0, mean[j] - spread[j])`` and
      ``upper[j] = mean[j] + spread[j]``.

    All four results are 3-long lists of floats in threshold order,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    per_e: list[list[float]] = []
    per_s: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        e_row: list[float] = []
        s_row: list[float] = []
        for level in levels:
            d = mu - level
            if sigma > 0:
                a = (level - mu) / sigma
                phi = math.exp(-a * a / 2.0) / sqrt_2pi
                p = 0.5 * math.erfc(a / math.sqrt(2))
                e = sigma * phi + d * p
                s = ((d * d + sigma * sigma) * p + sigma * d * phi) - e * e
            else:
                e = max(d, 0.0)
                s = 0.0
            e_row.append(e)
            s_row.append(s)
        per_e.append(e_row)
        per_s.append(s_row)

    mean = [
        math.fsum(w[k] * per_e[k][j] for k in range(n_scenarios))
        for j in range(3)
    ]
    variance = [
        math.fsum(
            w[k] * ((per_e[k][j] - mean[j]) ** 2 + per_s[k][j])
            for k in range(n_scenarios)
        )
        for j in range(3)
    ]
    spread = [z * math.sqrt(variance[j]) for j in range(3)]
    lower = [max(0.0, mean[j] - spread[j]) for j in range(3)]
    upper = [mean[j] + spread[j] for j in range(3)]

    return mean, spread, lower, upper


def excess_warning(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[int, float | None, list[float], tuple[list[float], list[float]]]:
    """Warning level derived from the expected-excess interval.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    With ``(M, R, L, U) = excess_interval(H, W, thresholds, weights, z)``,
    returns ``(level, trigger, score, interval)`` where:

    * ``level`` is the number of thresholds ``j`` with ``L[j] > 0``;
    * ``trigger`` is ``thresholds[level - 1]`` when ``level > 0``,
      otherwise ``None``;
    * ``score`` is ``M``;
    * ``interval`` is the pair ``(L, U)``.

    ``level`` is an int, ``trigger`` a float or ``None``, ``score`` a
    3-long list of floats and ``interval`` a 2-tuple of 3-long lists of
    floats, all in threshold order and unrounded.
    """
    mean, _, lower, upper = excess_interval(H, W, thresholds, weights, z)

    level = sum(1 for value in lower if value > 0)
    trigger = float(thresholds[level - 1]) if level > 0 else None

    return level, trigger, mean, (lower, upper)


def risk_contribution(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-scenario weighted exceedance probabilities and expected excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(P, E)`` where, with ``mu[k] = fsum(H[k])`` and
    ``sigma[k] = hypot(*W[k])``:

    * when ``sigma[k] > 0``, with ``a = (thresholds[j] - mu[k]) / sigma[k]``:
      ``q[k][j] = 0.5 * erfc(a / sqrt(2))`` and
      ``e[k][j] = sigma[k] * exp(-a * a / 2) / sqrt(2 * pi)
      + (mu[k] - thresholds[j]) * q[k][j]``;
    * when ``sigma[k] == 0``: ``q[k][j] = 1.0`` if
      ``thresholds[j] <= mu[k]`` else ``0.0``, and
      ``e[k][j] = max(mu[k] - thresholds[j], 0.0)``;
    * ``P[k][j] = w[k] * q[k][j]``;
    * ``E[k][j] = w[k] * e[k][j]``.

    Both results are K x 3 lists of lists of floats, in scenario then
    threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    P: list[list[float]] = []
    E: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        p_row: list[float] = []
        e_row: list[float] = []
        for level in levels:
            if sigma > 0:
                a = (level - mu) / sigma
                q = 0.5 * math.erfc(a / math.sqrt(2))
                e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * q
            else:
                q = 1.0 if level <= mu else 0.0
                e = max(mu - level, 0.0)
            p_row.append(w[k] * q)
            e_row.append(w[k] * e)
        P.append(p_row)
        E.append(e_row)

    return P, E


def risk_share(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-scenario share of aggregate exceedance risk and expected excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(PS, ES)`` where, with ``mu[k] = fsum(H[k])`` and
    ``sigma[k] = hypot(*W[k])``:

    * when ``sigma[k] > 0``, with ``a = (thresholds[j] - mu[k]) / sigma[k]``:
      ``q[k][j] = 0.5 * erfc(a / sqrt(2))`` and
      ``e[k][j] = sigma[k] * exp(-a * a / 2) / sqrt(2 * pi)
      + (mu[k] - thresholds[j]) * q[k][j]``;
    * when ``sigma[k] == 0``: ``q[k][j] = 1.0`` if
      ``thresholds[j] <= mu[k]`` else ``0.0``, and
      ``e[k][j] = max(mu[k] - thresholds[j], 0.0)``;
    * ``p[k][j] = w[k] * q[k][j]``, ``r[k][j] = w[k] * e[k][j]``;
    * ``P[j] = fsum(p[k][j] for k in range(K))`` and
      ``E[j] = fsum(r[k][j] for k in range(K))``;
    * ``PS[k][j] = p[k][j] / P[j]`` and ``ES[k][j] = r[k][j] / E[j]``,
      each taken as ``0.0`` when its denominator is ``0.0``.

    Both results are K x 3 lists of lists of floats, in scenario then
    threshold order, unrounded. The columns of each matrix sum to 1
    whenever the corresponding denominator is positive.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    p: list[list[float]] = []
    r: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        p_row: list[float] = []
        r_row: list[float] = []
        for level in levels:
            if sigma > 0:
                a = (level - mu) / sigma
                q = 0.5 * math.erfc(a / math.sqrt(2))
                e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * q
            else:
                q = 1.0 if level <= mu else 0.0
                e = max(mu - level, 0.0)
            p_row.append(w[k] * q)
            r_row.append(w[k] * e)
        p.append(p_row)
        r.append(r_row)

    totals_p = [
        math.fsum(p[k][j] for k in range(n_scenarios)) for j in range(3)
    ]
    totals_e = [
        math.fsum(r[k][j] for k in range(n_scenarios)) for j in range(3)
    ]

    PS: list[list[float]] = []
    ES: list[list[float]] = []
    for k in range(n_scenarios):
        ps_row: list[float] = []
        es_row: list[float] = []
        for j in range(3):
            if totals_p[j] == 0.0:
                ps_row.append(0.0)
            else:
                ps_row.append(p[k][j] / totals_p[j])
            if totals_e[j] == 0.0:
                es_row.append(0.0)
            else:
                es_row.append(r[k][j] / totals_e[j])
        PS.append(ps_row)
        ES.append(es_row)

    return PS, ES


def risk_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[float, float, float, float, list[float]]:
    """Aggregate scenario impacts with an interval and exceedance risks.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(M, R, lower, upper, p)`` where, with
    ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * ``M = fsum(w[k] * mu[k])``,
    * ``V = fsum(w[k] * ((mu[k] - M) ** 2 + sigma[k] ** 2))``,
    * ``R = z * sqrt(V)``,
    * ``lower = M - R``, ``upper = M + R``,
    * ``q[k][j] = 0.5 * erfc((thresholds[j] - mu[k]) / (sigma[k] * sqrt(2)))``
      when ``sigma[k] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= mu[k]`` else ``0.0``;
    * ``p[j] = fsum(w[k] * q[k][j] for k in range(K))``.

    The first four results are floats and ``p`` is a 3-long list of
    floats, all unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    mus = [math.fsum(row) for row in impacts]
    sigmas = [math.hypot(*row) for row in uncertainties]

    M = math.fsum(w[k] * mus[k] for k in range(n_scenarios))
    V = math.fsum(
        w[k] * ((mus[k] - M) ** 2 + sigmas[k] ** 2) for k in range(n_scenarios)
    )
    R = z * math.sqrt(V)

    q: list[list[float]] = []
    for k in range(n_scenarios):
        mu = mus[k]
        sigma = sigmas[k]
        row: list[float] = []
        for level in levels:
            if sigma > 0:
                row.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
            else:
                row.append(1.0 if level <= mu else 0.0)
        q.append(row)

    p = [
        math.fsum(w[k] * q[k][j] for k in range(n_scenarios)) for j in range(3)
    ]

    return M, R, M - R, M + R, p


def risk_excess_cvar(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[float]]:
    """Upper-tail conditional value at risk of aggregate threshold excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``C`` where, with ``mu[k] = fsum(H[k])``,
    ``sigma[k] = hypot(*W[k])`` and ``t = thresholds[j]``:

    * when ``sigma[k] > 0``, with ``a = (t - mu[k]) / sigma[k]``:
      ``p = 0.5 * erfc(a / sqrt(2))``,
      ``phi = exp(-a * a / 2) / sqrt(2 * pi)`` and
      ``E[k][j] = sigma[k] * phi + (mu[k] - t) * p``;
    * when ``sigma[k] == 0``: ``E[k][j] = max(mu[k] - t, 0.0)``;
    * for each threshold ``j`` and quantile ``q`` the scenarios are
      sorted by ``(E[k][j], -k)`` descending; with ``remaining = 1 - q``
      the scenarios are visited in that order, contributing
      ``take = min(w[k], remaining)`` each until ``remaining`` reaches
      0, accumulating ``n = fsum((n, take * E[k][j]))``, and
      ``C[m][j] = n / (1 - q)``.

    ``C`` is a ``len(quantiles)`` x 3 list of lists of floats, in
    ``quantiles`` then ``thresholds`` order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    excesses: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        row: list[float] = []
        for level in levels:
            if sigma > 0:
                a = (level - mu) / sigma
                p = 0.5 * math.erfc(a / math.sqrt(2))
                phi = math.exp(-a * a / 2.0) / sqrt_2pi
                e = sigma * phi + (mu - level) * p
            else:
                e = max(mu - level, 0.0)
            row.append(e)
        excesses.append(row)

    C: list[list[float]] = [[0.0] * 3 for _ in qs]
    for j in range(3):
        values = [excesses[k][j] for k in range(n_scenarios)]
        order = sorted(
            range(n_scenarios), key=lambda k: (values[k], -k), reverse=True
        )
        for m, q in enumerate(qs):
            remaining = 1.0 - q
            n = 0.0
            for k in order:
                if remaining <= 0.0:
                    break
                take = min(w[k], remaining)
                n = math.fsum((n, take * values[k]))
                remaining -= take
            C[m][j] = n / (1.0 - q)

    return C


def risk_excess_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[float]]:
    """Weighted quantiles of aggregate threshold expected excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``Q`` where, with ``mu[k] = fsum(H[k])``,
    ``sigma[k] = hypot(*W[k])`` and ``t = thresholds[j]``:

    * when ``sigma[k] > 0``, with ``a = (t - mu[k]) / sigma[k]``:
      ``E[k][j] = sigma[k] * exp(-a * a / 2) / sqrt(2 * pi)
      + (mu[k] - t) * erfc(a / sqrt(2)) / 2``;
    * when ``sigma[k] == 0``: ``E[k][j] = max(mu[k] - t, 0.0)``;
    * for each threshold ``j`` the scenarios are sorted by
      ``(E[k][j], k)`` ascending; for each requested quantile ``q`` the
      quantile scenario is the first one in that order when ``q == 0``,
      otherwise the first one whose cumulative normalized weight is
      ``>= q``;
    * ``Q[m][j]`` holds that scenario's ``E[k][j]`` value.

    ``Q`` is a ``len(quantiles)`` x 3 list of lists of floats, in
    ``quantiles`` then ``thresholds`` order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    excesses: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        row: list[float] = []
        for level in levels:
            if sigma > 0:
                a = (level - mu) / sigma
                e = (
                    sigma * math.exp(-a * a / 2.0) / sqrt_2pi
                    + (mu - level) * math.erfc(a / math.sqrt(2)) / 2.0
                )
            else:
                e = max(mu - level, 0.0)
            row.append(e)
        excesses.append(row)

    Q: list[list[float]] = [[0.0] * 3 for _ in qs]
    for j in range(3):
        values = [excesses[k][j] for k in range(n_scenarios)]
        order = sorted(range(n_scenarios), key=lambda k: (values[k], k))
        for m, q in enumerate(qs):
            if q == 0.0:
                chosen = order[0]
            else:
                cumulative = 0.0
                chosen = order[-1]
                for k in order:
                    cumulative = math.fsum([cumulative, w[k]])
                    if cumulative >= q:
                        chosen = k
                        break
            Q[m][j] = float(values[chosen])

    return Q


def level_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], float, float]:
    """Aggregate health-level probabilities across scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(probabilities, expected_level, alert_probability)`` where,
    with ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * ``q[k][j] = 0.5 * erfc((thresholds[j] - mu[k]) / (sigma[k] * sqrt(2)))``
      when ``sigma[k] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= mu[k]`` else ``0.0``;
    * ``p[k] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k``;
    * ``probabilities[j] = fsum(w[k] * p[k][j] for k in range(K))``;
    * ``expected_level = fsum(j * probabilities[j] for j in range(4))``;
    * ``alert_probability = probabilities[3]``.

    ``probabilities`` is a 4-long list of floats and ``expected_level``
    and ``alert_probability`` are floats, all unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[float]] = []
    for k in range(n_scenarios):
        mu = math.fsum(impacts[k])
        sigma = math.hypot(*uncertainties[k])
        q: list[float] = []
        for level in levels:
            if sigma > 0:
                q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
            else:
                q.append(1.0 if level <= mu else 0.0)
        per_scenario.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])

    probabilities = [
        math.fsum(w[k] * per_scenario[k][j] for k in range(n_scenarios))
        for j in range(4)
    ]
    expected_level = math.fsum(j * probabilities[j] for j in range(4))
    alert_probability = probabilities[3]

    return probabilities, expected_level, alert_probability


def max_level_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], float, float]:
    """Aggregate probabilities of the maximum per-receptor health level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(probabilities, expected_level, alert_probability)`` where,
    with ``mu = H[k][i]``, ``sigma = W[k][i]`` and ``t = thresholds[j]``:

    * ``q = 0.5 * erfc((t - mu) / (sigma * sqrt(2)))`` when ``sigma > 0``,
      otherwise ``1.0`` if ``t <= mu`` else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * receptors are treated as mutually independent, so
      ``F[k][r] = prod(fsum(p[k][i][s] for s in range(r + 1))
      for i in range(N))`` is the probability that every receptor is at
      level at most ``r`` (the CDF of the maximum level);
    * ``m[k][0] = F[k][0]`` and ``m[k][r] = F[k][r] - F[k][r - 1]`` for
      ``r`` in ``1..3`` is the probability mass of the maximum level;
    * ``probabilities[r] = fsum(w[k] * m[k][r] for k in range(K))``;
    * ``expected_level = fsum(r * probabilities[r] for r in range(4))``;
    * ``alert_probability = probabilities[3]``.

    ``probabilities`` is a 4-long list of floats and ``expected_level``
    and ``alert_probability`` are floats, all unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    mass: list[list[float]] = []
    for k in range(n_scenarios):
        cumulative = [
            math.prod(
                math.fsum(per_scenario[k][i][s] for s in range(r + 1))
                for i in range(n_receptors)
            )
            for r in range(4)
        ]
        mass.append(
            [
                cumulative[0],
                cumulative[1] - cumulative[0],
                cumulative[2] - cumulative[1],
                cumulative[3] - cumulative[2],
            ]
        )

    probabilities = [
        math.fsum(w[k] * mass[k][r] for k in range(n_scenarios)) for r in range(4)
    ]
    expected_level = math.fsum(r * probabilities[r] for r in range(4))
    alert_probability = probabilities[3]

    return probabilities, expected_level, alert_probability


def max_level_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[int]:
    """Weighted quantiles of the maximum per-receptor health level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns a list ``Q`` where, with ``mu = H[k][i]``,
    ``sigma = W[k][i]`` and ``t = thresholds[j]``:

    * ``qj = 0.5 * erfc((t - mu) / (sigma * sqrt(2)))`` when ``sigma > 0``,
      otherwise ``1.0`` if ``t <= mu`` else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * receptors are treated as mutually independent, so
      ``F[k][r] = prod(fsum(p[k][i][s] for s in range(r + 1))
      for i in range(N))`` is the probability that every receptor is at
      level at most ``r`` (the CDF of the maximum level);
    * ``m[k][0] = F[k][0]`` and ``m[k][r] = F[k][r] - F[k][r - 1]`` for
      ``r`` in ``1..3`` is the probability mass of the maximum level;
    * ``P[r] = fsum(w[k] * m[k][r] for k in range(K))`` is the weighted
      probability mass of the maximum level;
    * ``Q[m]`` is the smallest level ``r`` whose cumulative probability
      ``fsum(P[s] for s in range(r + 1))`` is ``>= quantiles[m]``
      (``0`` when ``quantiles[m] == 0``).

    ``Q`` is an M-long list of ints in ``quantiles`` input order,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1], got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    mass: list[list[float]] = []
    for k in range(n_scenarios):
        cumulative = [
            math.prod(
                math.fsum(per_scenario[k][i][s] for s in range(r + 1))
                for i in range(n_receptors)
            )
            for r in range(4)
        ]
        mass.append(
            [
                cumulative[0],
                cumulative[1] - cumulative[0],
                cumulative[2] - cumulative[1],
                cumulative[3] - cumulative[2],
            ]
        )

    P = [
        math.fsum(w[k] * mass[k][r] for k in range(n_scenarios)) for r in range(4)
    ]

    Q: list[int] = []
    for q in qs:
        if q == 0.0:
            Q.append(0)
        else:
            chosen = 3
            for r in range(4):
                if math.fsum(P[s] for s in range(r + 1)) >= q:
                    chosen = r
                    break
            Q.append(chosen)

    return Q


def receptor_expected_excess(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-receptor expected threshold exceedance, weighted over scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(probabilities, excess)`` where, with ``mu = H[k][i]``,
    ``sigma = W[k][i]`` and ``t = thresholds[j]``:

    * when ``sigma > 0``, with ``a = (t - mu) / sigma``:
      ``p = 0.5 * erfc(a / sqrt(2))`` and
      ``e = sigma * exp(-a * a / 2) / sqrt(2 * pi) + (mu - t) * p``;
    * when ``sigma == 0``: ``p = 1.0`` if ``t <= mu`` else ``0.0``, and
      ``e = max(mu - t, 0.0)``;
    * ``probabilities[i][j] = fsum(w[k] * p for k in range(K))``;
    * ``excess[i][j] = fsum(w[k] * e for k in range(K))``.

    Both results are N x 3 lists of lists of floats, in receptor then
    threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    probabilities: list[list[float]] = []
    excess: list[list[float]] = []
    for i in range(n_receptors):
        p_row: list[float] = []
        e_row: list[float] = []
        for level in levels:
            p_terms: list[float] = []
            e_terms: list[float] = []
            for k in range(n_scenarios):
                mu = impacts[k][i]
                sigma = uncertainties[k][i]
                if sigma > 0:
                    a = (level - mu) / sigma
                    p = 0.5 * math.erfc(a / math.sqrt(2))
                    e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * p
                else:
                    p = 1.0 if level <= mu else 0.0
                    e = max(mu - level, 0.0)
                p_terms.append(w[k] * p)
                e_terms.append(w[k] * e)
            p_row.append(math.fsum(p_terms))
            e_row.append(math.fsum(e_terms))
        probabilities.append(p_row)
        excess.append(e_row)

    return probabilities, excess


def receptor_excess(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], list[float]]:
    """Expected exceeding-receptor count and excess per threshold.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(count, excess)`` where, with ``mu = H[k][i]``,
    ``sigma = W[k][i]`` and ``t = thresholds[j]``:

    * when ``sigma > 0``, with ``a = (t - mu) / sigma``:
      ``q = 0.5 * erfc(a / sqrt(2))`` and
      ``e = sigma * exp(-a * a / 2) / sqrt(2 * pi) + (mu - t) * q``;
    * when ``sigma == 0``: ``q = 1.0`` if ``t <= mu`` else ``0.0``, and
      ``e = max(mu - t, 0.0)``;
    * ``count[j] = fsum(w[k] * q for k in range(K) for i in range(N))``;
    * ``excess[j] = fsum(w[k] * e for k in range(K) for i in range(N))``.

    Both results are 3-long lists of floats in threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    count: list[float] = []
    excess: list[float] = []
    for level in levels:
        q_terms: list[float] = []
        e_terms: list[float] = []
        for k in range(n_scenarios):
            for i in range(n_receptors):
                mu = impacts[k][i]
                sigma = uncertainties[k][i]
                if sigma > 0:
                    a = (level - mu) / sigma
                    q = 0.5 * math.erfc(a / math.sqrt(2))
                    e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * q
                else:
                    q = 1.0 if level <= mu else 0.0
                    e = max(mu - level, 0.0)
                q_terms.append(w[k] * q)
                e_terms.append(w[k] * e)
        count.append(math.fsum(q_terms))
        excess.append(math.fsum(e_terms))

    return count, excess


def receptor_excess_cvar(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[list[float]]]:
    """Upper-tail conditional value at risk of per-receptor expected excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``C`` where, with ``mu = H[k][i]``, ``sigma = W[k][i]`` and
    ``t = thresholds[j]``:

    * when ``sigma > 0``, with ``a = (t - mu) / sigma``:
      ``p = 0.5 * erfc(a / sqrt(2))`` and
      ``e(k, i, j) = sigma * exp(-a * a / 2) / sqrt(2 * pi)
      + (mu - t) * p``;
    * when ``sigma == 0``: ``e(k, i, j) = max(mu - t, 0.0)``;
    * for each receptor ``i``, threshold ``j`` and quantile ``q`` the
      scenarios are sorted by ``(e(k, i, j), -k)`` descending; with
      ``remaining = 1 - q`` the scenarios are visited in that order,
      contributing ``take = min(w[k], remaining)`` each until
      ``remaining`` reaches 0, and
      ``C[m][i][j] = fsum(take * e(k, i, j)) / (1 - q)``.

    ``C`` is an M x N x 3 list of lists of lists of floats, in
    ``quantiles``, receptor then threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    excesses: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            row: list[float] = []
            for level in levels:
                if sigma > 0:
                    a = (level - mu) / sigma
                    p = 0.5 * math.erfc(a / math.sqrt(2))
                    e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * p
                else:
                    e = max(mu - level, 0.0)
                row.append(e)
            rows.append(row)
        excesses.append(rows)

    C: list[list[list[float]]] = [
        [[0.0] * 3 for _ in range(n_receptors)] for _ in qs
    ]
    for i in range(n_receptors):
        for j in range(3):
            values = [excesses[k][i][j] for k in range(n_scenarios)]
            order = sorted(
                range(n_scenarios), key=lambda k: (values[k], -k), reverse=True
            )
            for m, q in enumerate(qs):
                remaining = 1.0 - q
                terms: list[float] = []
                for k in order:
                    if remaining <= 0.0:
                        break
                    take = min(w[k], remaining)
                    terms.append(take * values[k])
                    remaining -= take
                C[m][i][j] = math.fsum(terms) / (1.0 - q)

    return C


def receptor_excess_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    """Per-receptor weighted mean and interval for expected threshold excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(M, R, L, U)`` where, with ``mu = H[k][i]``,
    ``sigma = W[k][i]``, ``t = thresholds[j]`` and ``d = mu - t``:

    * when ``sigma > 0``, with ``a = (t - mu) / sigma``,
      ``phi = exp(-a * a / 2) / sqrt(2 * pi)`` and
      ``p = 0.5 * erfc(a / sqrt(2))``:
      ``e[k][i][j] = sigma * phi + d * p`` and
      ``s[k][i][j] = ((d * d + sigma ** 2) * p + sigma * d * phi)
      - e[k][i][j] ** 2``;
    * when ``sigma == 0``: ``e[k][i][j] = max(d, 0.0)`` and
      ``s[k][i][j] = 0.0``;
    * ``M[i][j] = fsum(w[k] * e[k][i][j] for k in range(K))``;
    * ``V[i][j] = fsum(w[k] * ((e[k][i][j] - M[i][j]) ** 2 + s[k][i][j])
      for k in range(K))``;
    * ``R[i][j] = z * sqrt(V[i][j])``;
    * ``L[i][j] = M[i][j] - R[i][j]`` and
      ``U[i][j] = M[i][j] + R[i][j]``.

    All four results are N x 3 lists of lists of floats in receptor then
    threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    per_e: list[list[list[float]]] = []
    per_s: list[list[list[float]]] = []
    for k in range(n_scenarios):
        e_rows: list[list[float]] = []
        s_rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            e_row: list[float] = []
            s_row: list[float] = []
            for level in levels:
                d = mu - level
                if sigma > 0:
                    a = (level - mu) / sigma
                    phi = math.exp(-a * a / 2.0) / sqrt_2pi
                    p = 0.5 * math.erfc(a / math.sqrt(2))
                    e = sigma * phi + d * p
                    s = ((d * d + sigma * sigma) * p + sigma * d * phi) - e * e
                else:
                    e = max(d, 0.0)
                    s = 0.0
                e_row.append(e)
                s_row.append(s)
            e_rows.append(e_row)
            s_rows.append(s_row)
        per_e.append(e_rows)
        per_s.append(s_rows)

    M: list[list[float]] = []
    R: list[list[float]] = []
    L: list[list[float]] = []
    U: list[list[float]] = []
    for i in range(n_receptors):
        m_row: list[float] = []
        r_row: list[float] = []
        l_row: list[float] = []
        u_row: list[float] = []
        for j in range(3):
            mean = math.fsum(w[k] * per_e[k][i][j] for k in range(n_scenarios))
            variance = math.fsum(
                w[k] * ((per_e[k][i][j] - mean) ** 2 + per_s[k][i][j])
                for k in range(n_scenarios)
            )
            spread = z * math.sqrt(variance)
            m_row.append(mean)
            r_row.append(spread)
            l_row.append(mean - spread)
            u_row.append(mean + spread)
        M.append(m_row)
        R.append(r_row)
        L.append(l_row)
        U.append(u_row)

    return M, R, L, U


def receptor_excess_warning(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[
    list[int],
    list[float | None],
    list[list[float]],
    tuple[list[list[float]], list[list[float]]],
]:
    """Per-receptor warning levels derived from the excess interval.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    With ``(M, R, L, U) = receptor_excess_interval(H, W, thresholds,
    weights, z)``, returns ``(levels, triggers, scores, interval)`` where,
    per receptor ``i``:

    * ``levels[i]`` is the number of thresholds ``j`` with ``L[i][j] > 0``;
    * ``triggers[i]`` is ``thresholds[levels[i] - 1]`` when
      ``levels[i] > 0``, otherwise ``None``;
    * ``scores`` is ``M``;
    * ``interval`` is the pair ``(L, U)``.

    ``levels`` is an N-long list of ints, ``triggers`` an N-long list of
    floats or ``None``, ``scores`` an N x 3 list of lists of floats and
    ``interval`` a 2-tuple of N x 3 lists of lists of floats, all in
    receptor then threshold order and unrounded.
    """
    M, _, L, U = receptor_excess_interval(H, W, thresholds, weights, z)

    levels = [sum(1 for value in row if value > 0) for row in L]
    triggers = [
        float(thresholds[level - 1]) if level > 0 else None for level in levels
    ]

    return levels, triggers, M, (L, U)


def receptor_excess_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[list[float]]]:
    """Weighted quantiles of per-receptor expected threshold excess.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``Q`` where, with ``mu = H[k][i]``, ``sigma = W[k][i]`` and
    ``t = thresholds[j]``:

    * when ``sigma > 0``, with ``a = (t - mu) / sigma``:
      ``p = 0.5 * erfc(a / sqrt(2))`` and
      ``e(k, i, j) = sigma * exp(-a * a / 2) / sqrt(2 * pi)
      + (mu - t) * p``;
    * when ``sigma == 0``: ``e(k, i, j) = max(mu - t, 0.0)``;
    * for each receptor ``i`` and threshold ``j`` the scenarios are
      sorted by ``(e(k, i, j), k)`` ascending and, for each requested
      quantile ``r``, the quantile scenario is the first one whose
      cumulative normalized weight is ``>= r`` (the first scenario in
      that order when ``r == 0``);
    * ``Q[m][i][j]`` holds that scenario's ``e`` value.

    ``Q`` is an M x N x 3 list of lists of lists of floats, in
    ``quantiles``, receptor then threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1], got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    excesses: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            row: list[float] = []
            for level in levels:
                if sigma > 0:
                    a = (level - mu) / sigma
                    p = 0.5 * math.erfc(a / math.sqrt(2))
                    e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * p
                else:
                    e = max(mu - level, 0.0)
                row.append(e)
            rows.append(row)
        excesses.append(rows)

    Q: list[list[list[float]]] = [
        [[0.0] * 3 for _ in range(n_receptors)] for _ in qs
    ]
    for i in range(n_receptors):
        for j in range(3):
            values = [excesses[k][i][j] for k in range(n_scenarios)]
            order = sorted(range(n_scenarios), key=lambda k: (values[k], k))
            for m, q in enumerate(qs):
                if q == 0.0:
                    chosen = order[0]
                else:
                    cumulative = 0.0
                    chosen = order[-1]
                    for k in order:
                        cumulative = math.fsum([cumulative, w[k]])
                        if cumulative >= q:
                            chosen = k
                            break
                Q[m][i][j] = float(values[chosen])

    return Q


def receptor_count_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[float]]:
    """Distribution of the number of receptors exceeding each threshold.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``p`` where:

    * ``q(k, i, j) = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``d[k][j]`` is the (N + 1)-long probability mass function of the
      number of exceeding receptors for scenario ``k`` and threshold
      ``j``: it starts as ``[1.0, 0.0, ...]`` and folds the receptors in
      one at a time with
      ``n[r] = fsum((d[r] * (1 - q), d[r - 1] * q))`` (out-of-range
      terms taken as 0);
    * ``p[j][r] = fsum(w[k] * d[k][j][r] for k in range(K))``.

    ``p`` is a 3 x (N + 1) list of lists of floats, in threshold then
    count (``r = 0..N``) order, unrounded. Each row sums to 1.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for level in levels:
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                sigma = uncertainties[k][i]
                if sigma > 0:
                    q = 0.5 * math.erfc(
                        (level - impacts[k][i]) / (sigma * math.sqrt(2))
                    )
                else:
                    q = 1.0 if level <= impacts[k][i] else 0.0
                n = [0.0] * (n_receptors + 1)
                for r in range(i + 2):
                    n[r] = math.fsum(
                        (
                            d[r] * (1.0 - q),
                            d[r - 1] * q if r > 0 else 0.0,
                        )
                    )
                d = n
            rows.append(d)
        distributions.append(rows)

    p = [
        [
            math.fsum(
                w[k] * distributions[k][j][r] for k in range(n_scenarios)
            )
            for r in range(n_receptors + 1)
        ]
        for j in range(3)
    ]

    return p


def receptor_count_event_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    minimums: list[int] | tuple[int, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], float]:
    """Probability that threshold-exceedance counts meet required minimums.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    minimums: list or tuple of exactly 3 non-bool ints, each >= 0; entry
        ``j`` is the required number of receptors exceeding
        ``thresholds[j]``.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``(A, weighted)`` where, with
    ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities of
      the four health levels at the receptor;
    * a state ``c`` is a 3-tuple whose entry ``j`` counts the receptors
      exceeding ``thresholds[j]``; the state distribution starts as
      ``{(0, 0, 0): 1.0}`` and folds the receptors in one at a time,
      visiting states in lexicographic order and levels ``l`` in ``0..3``
      with ``c'[j] = c[j] + (l > j)``, accumulated with ``fsum``;
    * ``A[k] = fsum(D[c] for c with c[j] >= minimums[j] for all j)``,
      taken as ``0.0`` when any minimum exceeds the receptor count;
    * ``weighted = fsum(w[k] * A[k] for k in range(K))``.

    ``A`` is a K-long list of floats in scenario order and ``weighted``
    is a float, both unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(minimums, (list, tuple)):
        raise TypeError(
            f"minimums must be a list or tuple, got {type(minimums).__name__}"
        )
    if len(minimums) != 3:
        raise ValueError(
            f"minimums must have exactly 3 elements, got {len(minimums)}"
        )
    mins: list[int] = []
    for j, item in enumerate(minimums):
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(
                f"minimums[{j}] must be an int, got {type(item).__name__}"
            )
        if item < 0:
            raise ValueError(f"minimums[{j}] must be >= 0, got {item!r}")
        mins.append(item)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if mins[0] > n_receptors or mins[1] > n_receptors or mins[2] > n_receptors:
        A = [0.0] * n_scenarios
    else:
        A = []
        for k in range(n_scenarios):
            per_receptor: list[list[float]] = []
            for i in range(n_receptors):
                mu = impacts[k][i]
                sigma = uncertainties[k][i]
                q: list[float] = []
                for level in levels:
                    if sigma > 0:
                        q.append(
                            0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2)))
                        )
                    else:
                        q.append(1.0 if level <= mu else 0.0)
                per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])

            distribution: dict[tuple[int, int, int], float] = {(0, 0, 0): 1.0}
            for i in range(n_receptors):
                p = per_receptor[i]
                incoming: dict[tuple[int, int, int], list[float]] = {}
                for c in sorted(distribution):
                    mass = distribution[c]
                    for l in range(4):
                        key = (c[0] + (l > 0), c[1] + (l > 1), c[2] + (l > 2))
                        incoming.setdefault(key, []).append(mass * p[l])
                distribution = {
                    key: math.fsum(terms) for key, terms in incoming.items()
                }

            A.append(
                math.fsum(
                    mass
                    for c, mass in distribution.items()
                    if c[0] >= mins[0] and c[1] >= mins[1] and c[2] >= mins[2]
                )
            )

    weighted = math.fsum(w[k] * A[k] for k in range(n_scenarios))

    return A, weighted


def receptor_count_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Weighted mean and interval for the number of exceeding receptors.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``(mean, spread, lower, upper)`` where:

    * ``q(k, i, j) = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``c[k][j] = fsum(q(k, i, j) for i in range(N))`` is the expected
      number of exceeding receptors for scenario ``k`` and threshold
      ``j`` and ``v[k][j] = fsum(q(k, i, j) * (1 - q(k, i, j))
      for i in range(N))`` is its variance;
    * ``M[j] = fsum(w[k] * c[k][j] for k in range(K))``,
    * ``V[j] = fsum(w[k] * ((c[k][j] - M[j]) ** 2 + v[k][j])
      for k in range(K))``,
    * ``R[j] = z * sqrt(V[j])``,
    * ``lower[j] = max(0, M[j] - R[j])``,
      ``upper[j] = min(N, M[j] + R[j])``.

    All four results are 3-long lists of floats in threshold order,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    counts: list[list[float]] = []
    variances: list[list[float]] = []
    for k in range(n_scenarios):
        c_row: list[float] = []
        v_row: list[float] = []
        for level in levels:
            qs = (
                0.5
                * math.erfc(
                    (level - impacts[k][i])
                    / (uncertainties[k][i] * math.sqrt(2))
                )
                if uncertainties[k][i] > 0
                else (1.0 if level <= impacts[k][i] else 0.0)
                for i in range(n_receptors)
            )
            q_list = list(qs)
            c_row.append(math.fsum(q_list))
            v_row.append(math.fsum(q * (1.0 - q) for q in q_list))
        counts.append(c_row)
        variances.append(v_row)

    mean: list[float] = []
    spread: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for j in range(3):
        m = math.fsum(w[k] * counts[k][j] for k in range(n_scenarios))
        v = math.fsum(
            w[k] * ((counts[k][j] - m) ** 2 + variances[k][j])
            for k in range(n_scenarios)
        )
        r = z * math.sqrt(v)
        mean.append(m)
        spread.append(r)
        lower.append(max(0.0, m - r))
        upper.append(min(float(n_receptors), m + r))

    return mean, spread, lower, upper


def receptor_count_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[int]]:
    """Quantiles of the number of receptors exceeding each threshold.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``Q`` where:

    * ``q(k, i, j) = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``d[k][j]`` is the (N + 1)-long probability mass function of the
      number of exceeding receptors for scenario ``k`` and threshold
      ``j``: it starts as ``[1.0, 0.0, ...]`` and folds the receptors in
      one at a time with
      ``n[r] = d[r] * (1 - q) + (d[r - 1] * q if r > 0 else 0.0)``;
    * ``p[j][r] = fsum(w[k] * d[k][j][r] for k in range(K))``;
    * ``Q[j][m]`` is the smallest count ``r`` whose cumulative
      probability ``fsum(p[j][:r + 1])`` is ``>= quantiles[m]`` (the
      smallest ``r`` when ``quantiles[m] == 0``).

    ``Q`` is a 3 x M list of lists of ints, in threshold then
    ``quantiles`` input order.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1], got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for level in levels:
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                sigma = uncertainties[k][i]
                if sigma > 0:
                    q = 0.5 * math.erfc(
                        (level - impacts[k][i]) / (sigma * math.sqrt(2))
                    )
                else:
                    q = 1.0 if level <= impacts[k][i] else 0.0
                n = [0.0] * (n_receptors + 1)
                for r in range(i + 2):
                    n[r] = d[r] * (1.0 - q) + (d[r - 1] * q if r > 0 else 0.0)
                d = n
            rows.append(d)
        distributions.append(rows)

    p = [
        [
            math.fsum(
                w[k] * distributions[k][j][r] for k in range(n_scenarios)
            )
            for r in range(n_receptors + 1)
        ]
        for j in range(3)
    ]

    Q: list[list[int]] = []
    for j in range(3):
        row: list[int] = []
        for q in qs:
            if q == 0.0:
                row.append(0)
            else:
                cumulative = 0.0
                chosen = n_receptors
                for r in range(n_receptors + 1):
                    cumulative = math.fsum([cumulative, p[j][r]])
                    if cumulative >= q:
                        chosen = r
                        break
                row.append(chosen)
        Q.append(row)

    return Q


def receptor_count_share(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Per-scenario share of the number of exceeding receptors.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``(PS, ES)`` where, with ``sigma = W[k][i]`` and
    ``t = thresholds[j]``:

    * ``q = 0.5 * erfc((t - H[k][i]) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``t <= H[k][i]`` else ``0.0``;
    * ``d[k][j]`` is the (N + 1)-long probability mass function of the
      number of exceeding receptors for scenario ``k`` and threshold
      ``j``: it starts as ``[1.0, 0.0, ...]`` and folds the receptors in
      one at a time with
      ``d'[r] = d[r] * (1 - q) + (d[r - 1] * q if r > 0 else 0.0)``;
    * ``A[k][j][r] = w[k] * d[k][j][r]``;
    * ``P[j][r] = fsum(A[k][j][r] for k in range(K))``;
    * ``PS[k][j][r] = A[k][j][r] / P[j][r]``, taken as ``0.0`` when
      ``P[j][r] == 0.0``;
    * ``B[j] = fsum(r * P[j][r] for r in range(N + 1))``;
    * ``ES[k][j][r] = A[k][j][r] * r / B[j]``, taken as ``0.0`` when
      ``B[j] == 0.0``.

    Both results are K x 3 x (N + 1) lists of lists of lists of floats,
    in scenario, threshold then count (``r = 0..N``) order, unrounded.
    Summing ``PS`` over scenarios for a fixed ``(j, r)`` yields 1 whenever
    ``P[j][r]`` is positive, and summing ``ES`` over scenarios and counts
    for a fixed ``j`` yields 1 whenever ``B[j]`` is positive.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for t in levels:
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                sigma = uncertainties[k][i]
                if sigma > 0:
                    q = 0.5 * math.erfc(
                        (t - impacts[k][i]) / (sigma * math.sqrt(2))
                    )
                else:
                    q = 1.0 if t <= impacts[k][i] else 0.0
                d = [
                    d[r] * (1.0 - q) + (d[r - 1] * q if r > 0 else 0.0)
                    for r in range(n_receptors + 1)
                ]
            rows.append(d)
        distributions.append(rows)

    A = [
        [
            [w[k] * distributions[k][j][r] for r in range(n_receptors + 1)]
            for j in range(3)
        ]
        for k in range(n_scenarios)
    ]
    P = [
        [
            math.fsum(A[k][j][r] for k in range(n_scenarios))
            for r in range(n_receptors + 1)
        ]
        for j in range(3)
    ]
    B = [
        math.fsum(r * P[j][r] for r in range(n_receptors + 1)) for j in range(3)
    ]

    PS: list[list[list[float]]] = []
    ES: list[list[list[float]]] = []
    for k in range(n_scenarios):
        ps_rows: list[list[float]] = []
        es_rows: list[list[float]] = []
        for j in range(3):
            ps_row: list[float] = []
            es_row: list[float] = []
            for r in range(n_receptors + 1):
                a = A[k][j][r]
                if P[j][r] > 0.0:
                    ps_row.append(a / P[j][r])
                else:
                    ps_row.append(0.0)
                if B[j] > 0.0:
                    es_row.append(a * r / B[j])
                else:
                    es_row.append(0.0)
            ps_rows.append(ps_row)
            es_rows.append(es_row)
        PS.append(ps_rows)
        ES.append(es_rows)

    return PS, ES


def receptor_count_cvar(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[float]]:
    """Upper-tail conditional value at risk of the exceeding-receptor count.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``C`` where:

    * ``q(k, i, j) = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``d[k][j]`` is the (N + 1)-long probability mass function of the
      number of exceeding receptors for scenario ``k`` and threshold
      ``j``: it starts as ``[1.0, 0.0, ...]`` and folds the receptors in
      one at a time with
      ``n[r] = fsum((d[r] * (1 - q(k, i, j)),
      d[r - 1] * q(k, i, j) if r > 0 else 0.0))``;
    * the atoms ``(w[k] * d[k][j][r], r, k)`` are visited in
      ``(-r, k)`` ascending order; for each quantile ``q``, with
      ``remaining = 1 - q``, each atom contributes
      ``take = min(w[k] * d[k][j][r], remaining)`` and
      ``num = fsum((num, take * r))`` until ``remaining`` reaches 0, and
      ``C[m][j] = num / (1 - q)``.

    ``C`` is an M x 3 list of lists of floats, in ``quantiles`` then
    threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for level in levels:
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                sigma = uncertainties[k][i]
                if sigma > 0:
                    q = 0.5 * math.erfc(
                        (level - impacts[k][i]) / (sigma * math.sqrt(2))
                    )
                else:
                    q = 1.0 if level <= impacts[k][i] else 0.0
                n = [0.0] * (n_receptors + 1)
                for r in range(i + 2):
                    n[r] = math.fsum(
                        (
                            d[r] * (1.0 - q),
                            d[r - 1] * q if r > 0 else 0.0,
                        )
                    )
                d = n
            rows.append(d)
        distributions.append(rows)

    C: list[list[float]] = [[0.0] * 3 for _ in qs]
    for j in range(3):
        atoms = [
            (w[k] * distributions[k][j][r], r, k)
            for r in range(n_receptors + 1)
            for k in range(n_scenarios)
        ]
        atoms.sort(key=lambda atom: (-atom[1], atom[2]))
        for m, q in enumerate(qs):
            remaining = 1.0 - q
            num = 0.0
            for probability, r, _ in atoms:
                if remaining <= 0.0:
                    break
                take = min(probability, remaining)
                num = math.fsum((num, take * r))
                remaining -= take
            C[m][j] = num / (1.0 - q)

    return C


def receptor_level_count_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Weighted mean and interval for the number of receptors at each level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``(mean, spread, lower, upper)`` where:

    * ``q[k][i][j] = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``c[k][r] = fsum(p[k][i][r] for i in range(N))`` is the expected
      number of receptors at level ``r`` for scenario ``k`` and
      ``v[k][r] = fsum(p[k][i][r] * (1 - p[k][i][r]) for i in range(N))``
      is its variance;
    * ``M[r] = fsum(w[k] * c[k][r] for k in range(K))``,
    * ``V[r] = fsum(w[k] * ((c[k][r] - M[r]) ** 2 + v[k][r])
      for k in range(K))``,
    * ``R[r] = z * sqrt(V[r])``,
    * ``lower[r] = max(0, M[r] - R[r])``,
      ``upper[r] = min(N, M[r] + R[r])``.

    All four results are 4-long lists of floats in level (``0..3``)
    order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    counts: list[list[float]] = []
    variances: list[list[float]] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        c_row = [
            math.fsum(per_receptor[i][r] for i in range(n_receptors))
            for r in range(4)
        ]
        v_row = [
            math.fsum(
                per_receptor[i][r] * (1.0 - per_receptor[i][r])
                for i in range(n_receptors)
            )
            for r in range(4)
        ]
        counts.append(c_row)
        variances.append(v_row)

    mean: list[float] = []
    spread: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for r in range(4):
        m = math.fsum(w[k] * counts[k][r] for k in range(n_scenarios))
        v = math.fsum(
            w[k] * ((counts[k][r] - m) ** 2 + variances[k][r])
            for k in range(n_scenarios)
        )
        rad = z * math.sqrt(v)
        mean.append(m)
        spread.append(rad)
        lower.append(max(0.0, m - rad))
        upper.append(min(float(n_receptors), m + rad))

    return mean, spread, lower, upper


def receptor_level_count_warning(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[int, float | None, list[float], tuple[list[float], list[float]]]:
    """Warning level derived from the receptor level-count interval.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    With ``(M, R, L, U) = receptor_level_count_interval(H, W, thresholds,
    weights, z)``, returns ``(level, trigger, scores, interval)`` where:

    * ``level`` is the largest ``r`` in ``1..3`` with ``L[r] > 0``, or
      ``0`` when no such ``r`` exists;
    * ``trigger`` is ``thresholds[level - 1]`` when ``level > 0``,
      otherwise ``None``;
    * ``scores`` is ``M``;
    * ``interval`` is the pair ``(L, U)``.

    ``level`` is an int, ``trigger`` a float or ``None``, ``scores`` a
    4-long list of floats and ``interval`` a 2-tuple of 4-long lists of
    floats, all in level (``0..3``) order and unrounded.
    """
    M, _, L, U = receptor_level_count_interval(H, W, thresholds, weights, z)

    level = max((r for r in range(1, 4) if L[r] > 0), default=0)
    trigger = float(thresholds[level - 1]) if level > 0 else None

    return level, trigger, M, (L, U)


def receptor_level_count_entropy(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    """Shannon entropy (natural log) of the receptor-count law per level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``E`` where, with ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``d[k][l]`` is the (N + 1)-long probability mass function of the
      number of receptors at level ``l`` for scenario ``k``: it starts as
      ``[1.0, 0.0, ...]`` and folds the receptors in one at a time with
      ``d'[r] = fsum([d[r] * (1 - p[k][i][l]), d[r - 1] * p[k][i][l] if
      r > 0 else 0.0])``;
    * ``P[l][r] = fsum(w[k] * d[k][l][r] for k in range(K))``;
    * ``E[l] = -fsum(P[l][r] * log(P[l][r]) for r in range(N + 1) if
      P[l][r] > 0)``.

    Returns a 4-long list of floats in level (``l = 0..3``) order,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        rows: list[list[float]] = []
        for level_index in range(4):
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                p_l = per_receptor[i][level_index]
                d = [
                    math.fsum(
                        (
                            d[r] * (1.0 - p_l),
                            d[r - 1] * p_l if r > 0 else 0.0,
                        )
                    )
                    for r in range(n_receptors + 1)
                ]
            rows.append(d)
        distributions.append(rows)

    P = [
        [
            math.fsum(
                w[k] * distributions[k][level_index][r]
                for k in range(n_scenarios)
            )
            for r in range(n_receptors + 1)
        ]
        for level_index in range(4)
    ]

    return [
        -math.fsum(
            P[l][r] * math.log(P[l][r])
            for r in range(n_receptors + 1)
            if P[l][r] > 0
        )
        for l in range(4)
    ]


def receptor_level_count_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[float]]:
    """Distribution of the number of receptors at each health level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``P`` where, with ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``d[k][l]`` is the (N + 1)-long probability mass function of the
      number of receptors at level ``l`` for scenario ``k``: it starts as
      ``[1.0, 0.0, ...]`` and folds the receptors in one at a time with
      ``d'[r] = d[r] * (1 - p[k][i][l]) + (d[r - 1] * p[k][i][l] if r > 0
      else 0.0)``;
    * ``P[l][r] = fsum(w[k] * d[k][l][r] for k in range(K))``.

    ``P`` is a 4 x (N + 1) list of lists of floats, in level then count
    (``r = 0..N``) order, unrounded. Each row sums to 1.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        rows: list[list[float]] = []
        for level_index in range(4):
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                p = per_receptor[i][level_index]
                d = [
                    d[r] * (1.0 - p) + (d[r - 1] * p if r > 0 else 0.0)
                    for r in range(n_receptors + 1)
                ]
            rows.append(d)
        distributions.append(rows)

    P = [
        [
            math.fsum(
                w[k] * distributions[k][level_index][r]
                for k in range(n_scenarios)
            )
            for r in range(n_receptors + 1)
        ]
        for level_index in range(4)
    ]

    return P


def receptor_level_count_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[list[int]]]:
    """Quantiles of the number of receptors at each health level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``Q`` where, with ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``d[k][l]`` is the (N + 1)-long probability mass function of the
      number of receptors at level ``l`` for scenario ``k``: it starts as
      ``[1.0, 0.0, ...]`` and folds the receptors in one at a time with
      ``d'[r] = d[r] * (1 - p[k][i][l]) + (d[r - 1] * p[k][i][l] if r > 0
      else 0.0)``;
    * ``P[l][r] = fsum(w[k] * d[k][l][r] for k in range(K))``;
    * ``Q[m][l]`` is the (N + 1)-long one-hot vector whose entry is ``1``
      at the count ``r`` that is ``0`` when ``quantiles[m] == 0`` and
      otherwise the smallest count whose cumulative probability
      ``fsum(P[l][s] for s in range(r + 1))`` is ``>= quantiles[m]``,
      and ``0`` at every other count.

    ``Q`` is an M x 4 x (N + 1) list of lists of lists of ints, in
    ``quantiles`` then level (``l = 0..3``) then count (``r = 0..N``)
    order.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1], got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        rows: list[list[float]] = []
        for level_index in range(4):
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                p = per_receptor[i][level_index]
                d = [
                    d[r] * (1.0 - p) + (d[r - 1] * p if r > 0 else 0.0)
                    for r in range(n_receptors + 1)
                ]
            rows.append(d)
        distributions.append(rows)

    P = [
        [
            math.fsum(
                w[k] * distributions[k][level_index][r]
                for k in range(n_scenarios)
            )
            for r in range(n_receptors + 1)
        ]
        for level_index in range(4)
    ]

    Q: list[list[list[int]]] = []
    for quantile in qs:
        row: list[list[int]] = []
        for level_index in range(4):
            if quantile == 0.0:
                chosen = 0
            else:
                chosen = n_receptors
                for r in range(n_receptors + 1):
                    if (
                        math.fsum(P[level_index][s] for s in range(r + 1))
                        >= quantile
                    ):
                        chosen = r
                        break
            row.append(
                [1 if r == chosen else 0 for r in range(n_receptors + 1)]
            )
        Q.append(row)

    return Q


def receptor_level_count_cvar(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[float]]:
    """Upper-tail conditional value at risk of the level-receptor count.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``C`` where:

    * with ``mu = H[k][i]`` and ``sigma = W[k][i]``,
      ``q_j = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``d[k][l]`` is the (N + 1)-long probability mass function of the
      number of receptors at level ``l`` for scenario ``k``: it starts as
      ``[1.0, 0.0, ...]`` and folds the receptors in one at a time with
      ``d'[r] = fsum((d[r] * (1 - p[k][i][l]),
      d[r - 1] * p[k][i][l] if r > 0 else 0.0))``;
    * the atoms ``(w[k] * d[k][l][r], r, k)`` are visited in
      ``(-r, k)`` ascending order; for each quantile ``q``, with
      ``remaining = 1 - q``, each atom contributes
      ``take = min(w[k] * d[k][l][r], remaining)`` and
      ``num = fsum((num, take * r))`` until ``remaining`` reaches 0, and
      ``C[m][l] = num / (1 - q)``.

    ``C`` is an M x 4 list of lists of floats, in ``quantiles`` then
    level (``l = 0..3``) order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        rows: list[list[float]] = []
        for level_index in range(4):
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                p = per_receptor[i][level_index]
                n = [0.0] * (n_receptors + 1)
                for r in range(i + 2):
                    n[r] = math.fsum(
                        (
                            d[r] * (1.0 - p),
                            d[r - 1] * p if r > 0 else 0.0,
                        )
                    )
                d = n
            rows.append(d)
        distributions.append(rows)

    C: list[list[float]] = [[0.0] * 4 for _ in qs]
    for level_index in range(4):
        atoms = [
            (w[k] * distributions[k][level_index][r], r, k)
            for r in range(n_receptors + 1)
            for k in range(n_scenarios)
        ]
        atoms.sort(key=lambda atom: (-atom[1], atom[2]))
        for m, q in enumerate(qs):
            remaining = 1.0 - q
            num = 0.0
            for probability, r, _ in atoms:
                if remaining <= 0.0:
                    break
                take = min(probability, remaining)
                num = math.fsum((num, take * r))
                remaining -= take
            C[m][level_index] = num / (1.0 - q)

    return C


def receptor_level_count_share(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Per-scenario share of receptors at each health level, by count.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``(PS, ES)`` where, with ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``d[k][l]`` is the (N + 1)-long probability mass function of the
      number of receptors at level ``l`` for scenario ``k``: it starts as
      ``[1.0, 0.0, ...]`` and folds the receptors in one at a time with
      ``d'[r] = d[r] * (1 - p[k][i][l]) + (d[r - 1] * p[k][i][l] if r > 0
      else 0.0)``;
    * ``P[l][r] = fsum(w[k] * d[k][l][r] for k in range(K))``;
    * ``A[k][l][r] = w[k] * d[k][l][r]``;
    * ``PS[k][l][r] = A[k][l][r] / P[l][r]``, taken as ``0.0`` when
      ``P[l][r] == 0.0``;
    * ``B[l] = fsum(r * P[l][r] for r in range(N + 1))``;
    * ``ES[k][l][r] = A[k][l][r] * r / B[l]``, taken as ``0.0`` when
      ``B[l] == 0.0``.

    Both results are K x 4 x (N + 1) lists of lists of lists of floats,
    in scenario, level (``l = 0..3``) then count (``r = 0..N``) order,
    unrounded. Summing ``PS`` over scenarios for a fixed ``(l, r)``
    yields 1 whenever ``P[l][r]`` is positive, and summing ``ES`` over
    scenarios and counts for a fixed ``l`` yields 1 whenever ``B[l]`` is
    positive.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    distributions: list[list[list[float]]] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        rows: list[list[float]] = []
        for level_index in range(4):
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                p = per_receptor[i][level_index]
                d = [
                    d[r] * (1.0 - p) + (d[r - 1] * p if r > 0 else 0.0)
                    for r in range(n_receptors + 1)
                ]
            rows.append(d)
        distributions.append(rows)

    P = [
        [
            math.fsum(
                w[k] * distributions[k][level_index][r]
                for k in range(n_scenarios)
            )
            for r in range(n_receptors + 1)
        ]
        for level_index in range(4)
    ]
    B = [
        math.fsum(r * P[level_index][r] for r in range(n_receptors + 1))
        for level_index in range(4)
    ]

    PS: list[list[list[float]]] = []
    ES: list[list[list[float]]] = []
    for k in range(n_scenarios):
        ps_rows: list[list[float]] = []
        es_rows: list[list[float]] = []
        for level_index in range(4):
            ps_row: list[float] = []
            es_row: list[float] = []
            for r in range(n_receptors + 1):
                a = w[k] * distributions[k][level_index][r]
                if P[level_index][r] > 0.0:
                    ps_row.append(a / P[level_index][r])
                else:
                    ps_row.append(0.0)
                if B[level_index] > 0.0:
                    es_row.append(a * r / B[level_index])
                else:
                    es_row.append(0.0)
            ps_rows.append(ps_row)
            es_rows.append(es_row)
        PS.append(ps_rows)
        ES.append(es_rows)

    return PS, ES


def receptor_level_count_transition(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    level: int = 3,
) -> list[list[list[float]]]:
    """Joint count distribution of a health level across scenario pairs.

    Scenario errors are treated as independent and receptors are folded in
    their given order. For each pair of consecutive scenarios the state
    ``D[(c, d)]`` is the probability that exactly ``c`` receptors are at
    ``level`` in the earlier scenario and exactly ``d`` in the following
    one.

    H: K x N (K >= 2, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    level: integer health level to track, ``0 <= level <= 3`` (default 3).
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each transition ``t`` let ``u[i] = p[t][i][level]`` and
    ``v[i] = p[t + 1][i][level]``. The recursion starts from
    ``D[(0, 0)] = 1.0`` and folds the receptors in one at a time; for each
    receptor every state key ``(c, d)`` (visited in sorted order) spreads
    its mass over the four targets ``(c, d)``, ``(c + 1, d)``,
    ``(c, d + 1)``, ``(c + 1, d + 1)`` with probabilities
    ``(1 - u)(1 - v)``, ``u(1 - v)``, ``(1 - u)v``, ``uv``, contributions
    to the same target being merged with ``math.fsum``.

    Returns a ``(K - 1) x (N + 1) x (N + 1)`` ``list[list[list[float]]]``
    indexed by transition, then count in the earlier scenario, then count
    in the following scenario, unrounded. Each ``T[t]`` sums to 1.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(impacts) < 2:
        raise ValueError(f"H must have at least 2 scenarios, got {len(impacts)}")
    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError(f"level must be an int, got {type(level).__name__}")
    if not 0 <= level <= 3:
        raise ValueError(f"level must be between 0 and 3, got {level}")

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for threshold in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((threshold - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if threshold <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    transitions: list[list[list[float]]] = []
    for t in range(n_scenarios - 1):
        state: dict[tuple[int, int], float] = {(0, 0): 1.0}
        for i in range(n_receptors):
            u = per_scenario[t][i][level]
            v = per_scenario[t + 1][i][level]
            moves = (
                ((0, 0), (1.0 - u) * (1.0 - v)),
                ((1, 0), u * (1.0 - v)),
                ((0, 1), (1.0 - u) * v),
                ((1, 1), u * v),
            )
            updated: dict[tuple[int, int], list[float]] = {}
            for (c, d) in sorted(state):
                mass = state[(c, d)]
                for (a, b), probability in moves:
                    updated.setdefault((c + a, d + b), []).append(
                        mass * probability
                    )
            state = {
                target: math.fsum(terms) for target, terms in updated.items()
            }
        transitions.append(
            [
                [state.get((c, d), 0.0) for d in range(n_receptors + 1)]
                for c in range(n_receptors + 1)
            ]
        )

    return transitions


def receptor_level_count_rise_distribution(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    level: int = 3,
) -> list[float]:
    """Distribution of the number of rises of a health-level receptor count.

    Scenario errors are treated as independent. For each scenario ``k`` the
    ``N`` receptors are folded in their given order, taking ``p`` to be the
    probability that the receptor is at ``level``, yielding the count PMF
    ``Ck[c]`` over the number ``c`` of receptors at ``level``. Across the
    scenario sequence a "rise" happens at scenario ``k`` when its count is
    strictly greater than the previous scenario's count.

    H: K x N (K >= 2, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    level: non-bool integer health level to track, ``0 <= level <= 3``
        (default 3).
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``p = p[k][i][level]`` is the probability of the tracked level.

    For each scenario ``k`` the count PMF starts as ``d = [1] + [0] * N``
    and folds the receptors one at a time with
    ``d'[r] = fsum((d[r] * (1 - p), d[r - 1] * p if r > 0))``, giving
    ``Ck = d``. The cross-scenario state ``D[(c, r)]`` holds the probability
    that the previous scenario had count ``c`` with ``r`` rises so far; it
    starts as ``{(c, 0): C0[c] for c}``. For ``k = 1..K - 1`` each state
    spreads onto every target count ``c'`` (states and targets visited in
    lexicographic order) as
    ``D'[(c', r + (c' > c))] += D[(c, r)] * Ck[c']``, contributions to the
    same target being merged with ``math.fsum``.

    Returns the K-long ``list[float]``
    ``R[r] = fsum(D[(c, r)] for c in range(N + 1))`` for ``r = 0..K - 1``,
    the distribution over the number of rises, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(impacts) < 2:
        raise ValueError(f"H must have at least 2 scenarios, got {len(impacts)}")
    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError(f"level must be an int, got {type(level).__name__}")
    if not 0 <= level <= 3:
        raise ValueError(f"level must be between 0 and 3, got {level}")

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for threshold in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((threshold - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if threshold <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    count_pmfs: list[list[float]] = []
    for k in range(n_scenarios):
        d = [1.0] + [0.0] * n_receptors
        for i in range(n_receptors):
            p = per_scenario[k][i][level]
            updated = [0.0] * (n_receptors + 1)
            for r in range(n_receptors + 1):
                terms = [d[r] * (1.0 - p)]
                if r > 0:
                    terms.append(d[r - 1] * p)
                updated[r] = math.fsum(terms)
            d = updated
        count_pmfs.append(d)

    state: dict[tuple[int, int], float] = {
        (c, 0): count_pmfs[0][c] for c in range(n_receptors + 1)
    }
    for k in range(1, n_scenarios):
        updated: dict[tuple[int, int], list[float]] = {}
        for c, r in sorted(state):
            mass = state[(c, r)]
            for c_next in range(n_receptors + 1):
                target = (c_next, r + (1 if c_next > c else 0))
                updated.setdefault(target, []).append(mass * count_pmfs[k][c_next])
        state = {target: math.fsum(terms) for target, terms in updated.items()}

    return [
        math.fsum(state.get((c, r), 0.0) for c in range(n_receptors + 1))
        for r in range(n_scenarios)
    ]


def _count_run_event_probs(
    impacts: list[list[float]],
    uncertainties: list[list[float]],
    levels: list[float],
    level: int,
    minimum_count: int,
) -> list[float]:
    """Per-scenario probabilities of the level-count event.

    ``impacts`` and ``uncertainties`` are the validated K x N matrices and
    ``levels`` the 3 validated threshold values. Returns the K-long list
    ``a[k] = fsum(Ck[minimum_count:])`` where ``Ck`` is the PMF of the
    number of receptors at ``level`` in scenario ``k``.
    """
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for threshold in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((threshold - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if threshold <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    event_probs: list[float] = []
    for k in range(n_scenarios):
        d = [1.0] + [0.0] * n_receptors
        for i in range(n_receptors):
            p = per_scenario[k][i][level]
            updated = [0.0] * (n_receptors + 1)
            for r in range(n_receptors + 1):
                updated[r] = math.fsum(
                    (d[r] * (1.0 - p), d[r - 1] * p if r > 0 else 0.0)
                )
            d = updated
        event_probs.append(math.fsum(d[minimum_count:]))
    return event_probs


def receptor_level_count_run_distribution(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    level: int = 3,
    minimum_count: int = 1,
) -> list[float]:
    """Distribution of the longest run of a health-level receptor count event.

    Scenario errors are treated as independent. For each scenario ``k`` the
    ``N`` receptors are folded in their given order, taking ``p`` to be the
    probability that the receptor is at ``level``, yielding the count PMF
    ``Ck[c]`` over the number ``c`` of receptors at ``level``. The event for
    scenario ``k`` is "at least ``minimum_count`` receptors at ``level``",
    with probability ``ak``; a "run" is a maximal stretch of consecutive
    scenarios where the event happens.

    H: K x N (K >= 2, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    level: non-bool integer health level to track, ``0 <= level <= 3``
        (default 3).
    minimum_count: non-bool integer event threshold, ``minimum_count >= 1``
        (default 1).
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``p = p[k][i][level]`` is the probability of the tracked level.

    For each scenario ``k`` the count PMF starts as ``d = [1] + [0] * N``
    and folds the receptors one at a time with
    ``d'[r] = fsum((d[r] * (1 - p), d[r - 1] * p if r > 0 else 0.0))`` for
    ``r = 0..N``, giving ``Ck = d``. The event probability is
    ``ak = fsum(Ck[minimum_count:])`` (``0.0`` when ``minimum_count > N``).
    The cross-scenario state ``D[(r, m)]`` holds the probability that the
    current run length is ``r`` and the longest run so far is ``m``; it
    starts as ``{(0, 0): 1}``. For each scenario in order, every state
    (states visited in ascending ``(r, m)`` order) first spreads onto the
    non-event target ``(0, m)`` with weight ``1 - ak`` and then onto the
    event target ``(min(r + 1, K), max(m, r + 1))`` with weight ``ak``,
    contributions to the same target being merged with ``math.fsum``.

    Returns the ``(K + 1)``-long ``list[float]``
    ``R[m] = fsum(D[(r, m)] for r in range(K + 1))`` for ``m = 0..K``, the
    distribution over the longest run length, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(impacts) < 2:
        raise ValueError(f"H must have at least 2 scenarios, got {len(impacts)}")
    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError(f"level must be an int, got {type(level).__name__}")
    if not 0 <= level <= 3:
        raise ValueError(f"level must be between 0 and 3, got {level}")

    if not isinstance(minimum_count, int) or isinstance(minimum_count, bool):
        raise TypeError(
            f"minimum_count must be an int, got {type(minimum_count).__name__}"
        )
    if minimum_count < 1:
        raise ValueError(f"minimum_count must be >= 1, got {minimum_count}")

    event_probs = _count_run_event_probs(
        impacts, uncertainties, levels, level, minimum_count
    )

    state: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for k in range(n_scenarios):
        a = event_probs[k]
        updated: dict[tuple[int, int], list[float]] = {}
        for r, m in sorted(state):
            mass = state[(r, m)]
            updated.setdefault((0, m), []).append(mass * (1.0 - a))
            target = (min(r + 1, n_scenarios), max(m, r + 1))
            updated.setdefault(target, []).append(mass * a)
        state = {target: math.fsum(terms) for target, terms in updated.items()}

    return [
        math.fsum(state.get((r, m), 0.0) for r in range(n_scenarios + 1))
        for m in range(n_scenarios + 1)
    ]


def receptor_level_count_run_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    level: int = 3,
    minimum_count: int = 1,
) -> list[int]:
    """Quantiles of the longest run of a health-level receptor count event.

    The run-length distribution is taken verbatim from
    :func:`receptor_level_count_run_distribution`; see that function for the
    full scenario folding and error contract. Scenario errors are treated as
    independent: each scenario ``k`` folds its ``N`` receptors into the count
    PMF ``Ck``, the event is "at least ``minimum_count`` receptors at
    ``level``" with probability ``ak``, and states ``(r, m)`` track the
    current run length ``r`` and longest run so far ``m`` across scenarios in
    their given order.

    H: K x N (K >= 2, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    level: non-bool integer health level to track, ``0 <= level <= 3``
        (default 3).
    minimum_count: non-bool integer event threshold, ``minimum_count >= 1``
        (default 1).

    Let ``R`` be the ``(K + 1)``-long result returned by
    ``receptor_level_count_run_distribution(H, W, thresholds, level,
    minimum_count)``, the distribution over the longest run length. For
    quantile ``q`` the result is ``0`` when ``q == 0``, otherwise the
    smallest ``m`` such that ``fsum(R[:m + 1]) >= q``.

    Returns a ``list[int]`` in ``quantiles`` order, unrounded.
    """
    R = receptor_level_count_run_distribution(
        H, W, thresholds, level=level, minimum_count=minimum_count
    )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    result: list[int] = [0] * len(qs)
    for m_index, quantile in enumerate(qs):
        if quantile == 0.0:
            continue
        cumulative_terms: list[float] = []
        for m in range(len(R)):
            cumulative_terms.append(R[m])
            if math.fsum(cumulative_terms) >= quantile:
                result[m_index] = m
                break

    return result


def receptor_count_run_warning(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    durations: list[int] | tuple[int, ...],
    level: int = 3,
    minimum_count: int = 1,
) -> tuple[
    list[int], list[int], list[int | None], list[list[float]], list[list[float]]
]:
    """Warning levels from run-length quantiles of a receptor count event.

    ``H``, ``W``, ``thresholds``, ``quantiles``, ``level`` and
    ``minimum_count`` share the contract of
    :func:`receptor_level_count_run_quantile` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim); ``runs`` below is exactly that
    function's result. ``durations`` is a list or tuple of exactly 3
    strictly increasing non-bool positive ints, each ``<= K``; container
    or element type errors raise ``TypeError``, while length, value or
    order errors raise ``ValueError``.

    With ``runs[m]`` the longest-run quantile for ``quantiles[m]``,
    ``levels[m]`` counts the durations ``d`` with ``d <= runs[m]`` and
    ``triggers[m]`` is ``None`` when ``levels[m] == 0``, otherwise the
    largest such ``d``.

    The per-scenario event probabilities follow the existing
    distribution: ``a[k] = fsum(Ck[minimum_count:])`` with ``Ck`` the
    receptor-count PMF of scenario ``k`` as in
    :func:`receptor_level_count_run_distribution`. For each duration
    ``d = durations[j]`` a state vector ``s`` of length ``d`` holds the
    probability of a current untriggered streak of length ``0..d - 1``
    and starts as ``[1, 0, ...]``. Iterating ``k = 0..K - 1``:
    ``first[k][j] = s[d - 1] * a[k]``,
    ``s'[0] = fsum(s) * (1 - a[k])``,
    ``s'[r] = s[r - 1] * a[k]`` for ``1 <= r < d``,
    ``cumulative[k][j] = 1 - fsum(s')``, then ``s = s'``.

    Returns ``(runs, levels, triggers, first, cumulative)``: the first
    three are M-long ``list[int]``, ``list[int]`` and
    ``list[int | None]``; the last two are K x 3
    ``list[list[float]]`` in scenario x duration order, unrounded.
    """
    runs = receptor_level_count_run_quantile(
        H, W, thresholds, quantiles, level=level, minimum_count=minimum_count
    )

    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)
    n_scenarios = len(impacts)

    if not isinstance(durations, (list, tuple)):
        raise TypeError(
            f"durations must be a list or tuple, got {type(durations).__name__}"
        )
    if len(durations) != 3:
        raise ValueError(
            f"durations must have exactly 3 elements, got {len(durations)}"
        )
    ds: list[int] = []
    for j, item in enumerate(durations):
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(
                f"durations[{j}] must be an int, got {type(item).__name__}"
            )
        if item < 1:
            raise ValueError(f"durations[{j}] must be >= 1, got {item!r}")
        if item > n_scenarios:
            raise ValueError(
                f"durations[{j}] must be <= scenario count {n_scenarios}, "
                f"got {item!r}"
            )
        if j > 0 and item <= ds[j - 1]:
            raise ValueError(
                f"durations must be strictly increasing, got {[*ds, item]!r}"
            )
        ds.append(item)

    threshold_values = [float(item) for item in thresholds]
    event_probs = _count_run_event_probs(
        impacts, uncertainties, threshold_values, level, minimum_count
    )

    warning_levels: list[int] = []
    triggers: list[int | None] = []
    for run in runs:
        count = 0
        trigger: int | None = None
        for d in ds:
            if d <= run:
                count += 1
                trigger = d
        warning_levels.append(count)
        triggers.append(trigger)

    first = [[0.0] * 3 for _ in range(n_scenarios)]
    cumulative = [[0.0] * 3 for _ in range(n_scenarios)]
    states = [[1.0] + [0.0] * (d - 1) for d in ds]
    for k in range(n_scenarios):
        a = event_probs[k]
        for j, d in enumerate(ds):
            s = states[j]
            first[k][j] = s[d - 1] * a
            updated = [0.0] * d
            updated[0] = math.fsum(s) * (1.0 - a)
            for r in range(1, d):
                updated[r] = s[r - 1] * a
            cumulative[k][j] = 1.0 - math.fsum(updated)
            states[j] = updated

    return runs, warning_levels, triggers, first, cumulative


def receptor_count_run_policy_batch(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
) -> tuple[list[tuple], list[float], list[int]]:
    """Batch-evaluate run-warning policies and rank them.

    ``H``, ``W``, ``thresholds`` and ``quantiles`` share the contract of
    :func:`receptor_count_run_warning` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim). ``policies`` is a non-empty
    list or tuple; each entry must be a list or tuple of exactly five
    values ``(level, minimum_count, d1, d2, d3)``, all five non-bool
    ints. A wrong container, entry or element type raises ``TypeError``;
    an empty ``policies``, an entry of length other than 5, a ``level``
    outside ``0..3``, a ``minimum_count < 1``, or durations that are not
    positive, not strictly increasing or not ``<= K`` raise
    ``ValueError``.

    For each policy ``p`` the durations ``(d1, d2, d3)`` are passed as
    ``durations`` to :func:`receptor_count_run_warning` with the
    policy's ``level`` and ``minimum_count``. ``results[p]`` stores that
    call's ``(runs, levels, triggers, first, cumulative)`` tuple and
    ``scores[p] = fsum(cumulative[K - 1])`` over the three duration
    columns.

    Returns ``(results, scores, order)`` where ``results`` is a
    P-long ``list[tuple]``, ``scores`` a P-long ``list[float]`` and
    ``order`` a P-long ``list[int]`` of 0-based policy indices sorted
    ascending by ``(-max(levels), -scores[p], p)``. Inner shapes follow
    :func:`receptor_count_run_warning` and values are not rounded.
    """
    if not isinstance(policies, (list, tuple)):
        raise TypeError(
            f"policies must be a list or tuple, got {type(policies).__name__}"
        )
    if len(policies) == 0:
        raise ValueError("policies must not be empty")

    results: list[tuple] = []
    scores: list[float] = []
    peaks: list[int] = []
    for p, policy in enumerate(policies):
        if not isinstance(policy, (list, tuple)):
            raise TypeError(
                f"policies[{p}] must be a list or tuple, "
                f"got {type(policy).__name__}"
            )
        if len(policy) != 5:
            raise ValueError(
                f"policies[{p}] must have exactly 5 elements, "
                f"got {len(policy)}"
            )
        values: list[int] = []
        for e, item in enumerate(policy):
            if not isinstance(item, int) or isinstance(item, bool):
                raise TypeError(
                    f"policies[{p}][{e}] must be an int, "
                    f"got {type(item).__name__}"
                )
            values.append(item)
        level, minimum_count, d1, d2, d3 = values
        if not 0 <= level <= 3:
            raise ValueError(
                f"policies[{p}][0] level must be between 0 and 3, got {level}"
            )
        if minimum_count < 1:
            raise ValueError(
                f"policies[{p}][1] minimum_count must be >= 1, "
                f"got {minimum_count}"
            )
        durations = (d1, d2, d3)
        for j, d in enumerate(durations):
            if d < 1:
                raise ValueError(
                    f"policies[{p}][{j + 2}] must be >= 1, got {d!r}"
                )
            if j > 0 and d <= durations[j - 1]:
                raise ValueError(
                    f"policies[{p}] durations must be strictly increasing, "
                    f"got {list(durations)!r}"
                )

        outcome = receptor_count_run_warning(
            H,
            W,
            thresholds,
            quantiles,
            durations,
            level=level,
            minimum_count=minimum_count,
        )
        runs, run_levels, triggers, first, cumulative = outcome
        results.append(outcome)
        scores.append(math.fsum(cumulative[-1]))
        peaks.append(max(run_levels))

    order = sorted(
        range(len(results)),
        key=lambda p: (-peaks[p], -scores[p], p),
    )

    return results, scores, order


def receptor_level_event_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    minimums: list[int] | tuple[int, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[float], float]:
    """Probability that per-level receptor counts meet required minimums.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, strictly increasing
        numbers.
    minimums: list or tuple of exactly 4 non-bool ints, each >= 0; entry
        ``l`` is the required number of receptors at health level ``l``.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``(A, weighted)`` where, with
    ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities of
      the four health levels at the receptor;
    * a state ``c`` is a 4-tuple of receptor counts per level; the state
      distribution starts as ``{(0, 0, 0, 0): 1.0}`` and folds the
      receptors in one at a time, visiting states in lexicographic order
      and levels ``l`` in ``0..3`` with
      ``D'[c + e_l] += D[c] * p[l]``, accumulated with ``fsum``;
    * ``A[k] = fsum(D[c] for c with c[l] >= minimums[l] for all l)``;
    * ``weighted = fsum(w[k] * A[k] for k in range(K))``.

    ``A`` is a K-long list of floats in scenario order and ``weighted``
    is a float, both unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(minimums, (list, tuple)):
        raise TypeError(
            f"minimums must be a list or tuple, got {type(minimums).__name__}"
        )
    if len(minimums) != 4:
        raise ValueError(
            f"minimums must have exactly 4 elements, got {len(minimums)}"
        )
    mins: list[int] = []
    for l, item in enumerate(minimums):
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(
                f"minimums[{l}] must be an int, got {type(item).__name__}"
            )
        if item < 0:
            raise ValueError(f"minimums[{l}] must be >= 0, got {item!r}")
        mins.append(item)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    A: list[float] = []
    for k in range(n_scenarios):
        per_receptor: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            per_receptor.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])

        distribution: dict[tuple[int, int, int, int], float] = {
            (0, 0, 0, 0): 1.0
        }
        for i in range(n_receptors):
            p = per_receptor[i]
            incoming: dict[tuple[int, int, int, int], list[float]] = {}
            for c in sorted(distribution):
                mass = distribution[c]
                for l in range(4):
                    nxt = list(c)
                    nxt[l] += 1
                    key = (nxt[0], nxt[1], nxt[2], nxt[3])
                    incoming.setdefault(key, []).append(mass * p[l])
            distribution = {
                key: math.fsum(terms) for key, terms in incoming.items()
            }

        A.append(
            math.fsum(
                mass
                for c, mass in distribution.items()
                if c[0] >= mins[0]
                and c[1] >= mins[1]
                and c[2] >= mins[2]
                and c[3] >= mins[3]
            )
        )

    weighted = math.fsum(w[k] * A[k] for k in range(n_scenarios))

    return A, weighted


def receptor_level_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Per-receptor interval for the expected health level, weighted over scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(mean, spread, lower, upper)`` where, with ``mu = H[k][i]``
    and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities of
      the four health levels for scenario ``k`` at receptor ``i``;
    * ``e[k][i] = fsum(r * p[r] for r in range(4))`` and
      ``v[k][i] = fsum(r * r * p[r] for r in range(4)) - e[k][i] ** 2``;
    * ``mean[i] = fsum(w[k] * e[k][i] for k in range(K))``;
    * ``V[i] = fsum(w[k] * ((e[k][i] - mean[i]) ** 2 + v[k][i])
      for k in range(K))``;
    * ``spread[i] = z * sqrt(V[i])``;
    * ``lower[i] = mean[i] - spread[i]``, ``upper[i] = mean[i] + spread[i]``.

    All four results are N-long lists of floats in receptor order,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    e: list[list[float]] = []
    v: list[list[float]] = []
    for k in range(n_scenarios):
        e_row: list[float] = []
        v_row: list[float] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            p = [1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]]
            expected = math.fsum(r * p[r] for r in range(4))
            variance = math.fsum(r * r * p[r] for r in range(4)) - expected ** 2
            e_row.append(expected)
            v_row.append(variance)
        e.append(e_row)
        v.append(v_row)

    mean: list[float] = []
    spread: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for i in range(n_receptors):
        m = math.fsum(w[k] * e[k][i] for k in range(n_scenarios))
        variance = math.fsum(
            w[k] * ((e[k][i] - m) ** 2 + v[k][i]) for k in range(n_scenarios)
        )
        r = z * math.sqrt(variance)
        mean.append(m)
        spread.append(r)
        lower.append(m - r)
        upper.append(m + r)

    return mean, spread, lower, upper


def receptor_level_path_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    paths: list[list[int]] | tuple[tuple[int, ...], ...],
) -> list[list[float]]:
    """Probabilities of given per-scenario health-level paths at each receptor.

    Scenario errors are treated as independent.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    paths: non-empty list or tuple; each ``paths[m]`` must be a list or
        tuple of length K whose entries are non-bool ints with
        ``0 <= level <= 3``.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``result[m][i] = prod(p[k][i][paths[m][k]] for k in range(K))``.

    Returns an M x N ``list[list[float]]`` with ``M = len(paths)``, in
    path then receptor order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(paths, (list, tuple)):
        raise TypeError(
            f"paths must be a list or tuple, got {type(paths).__name__}"
        )
    if len(paths) == 0:
        raise ValueError("paths must not be empty")
    validated_paths: list[list[int]] = []
    for m, path in enumerate(paths):
        if not isinstance(path, (list, tuple)):
            raise TypeError(
                f"paths[{m}] must be a list or tuple, got {type(path).__name__}"
            )
        if len(path) != n_scenarios:
            raise ValueError(
                f"paths[{m}] must have {n_scenarios} elements, got {len(path)}"
            )
        validated = []
        for k, item in enumerate(path):
            if not isinstance(item, int) or isinstance(item, bool):
                raise TypeError(
                    f"paths[{m}][{k}] must be an int, "
                    f"got {type(item).__name__}"
                )
            if not 0 <= item <= 3:
                raise ValueError(
                    f"paths[{m}][{k}] must be between 0 and 3, got {item!r}"
                )
            validated.append(item)
        validated_paths.append(validated)

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    return [
        [
            math.prod(
                per_scenario[k][i][validated_paths[m][k]]
                for k in range(n_scenarios)
            )
            for i in range(n_receptors)
        ]
        for m in range(len(validated_paths))
    ]


def receptor_level_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Per-receptor health-level probabilities, equally weighted over scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    Returns ``(probabilities, expected_level, alert_probability)`` where,
    with ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``probabilities[i][j] = fsum(p[k][i][j] for k in range(K)) / K``;
    * ``expected_level[i] = fsum(j * probabilities[i][j] for j in range(4))``;
    * ``alert_probability[i] = probabilities[i][3]``.

    ``probabilities`` is an N x 4 list of lists of floats and
    ``expected_level`` and ``alert_probability`` are N-long lists of
    floats, all in receptor order and unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    probabilities: list[list[float]] = []
    for i in range(n_receptors):
        probabilities.append(
            [
                math.fsum(per_scenario[k][i][j] for k in range(n_scenarios))
                / n_scenarios
                for j in range(4)
            ]
        )
    expected_level = [
        math.fsum(j * probabilities[i][j] for j in range(4))
        for i in range(n_receptors)
    ]
    alert_probability = [probabilities[i][3] for i in range(n_receptors)]

    return probabilities, expected_level, alert_probability


def quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float]]:
    """Weighted quantiles of scenario total health impacts.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(Q, L, U)`` where, with
    ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * ``v[k] = mu[k]``, ``l[k] = mu[k] - z * sigma[k]``,
      ``u[k] = mu[k] + z * sigma[k]``;
    * for each requested quantile ``q`` the scenarios are sorted by
      ``(v[k], k)`` ascending and the quantile scenario is the first one
      whose cumulative normalized weight is ``>= q`` (the first scenario
      in that order when ``q == 0``);
    * ``Q``, ``L`` and ``U`` hold that scenario's ``v``, ``l`` and ``u``
      values respectively.

    Each result is an M-long list of floats in ``quantiles`` input order,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for j, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{j}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{j}] must be in [0, 1], got {value!r}")
        if j > 0 and value < qs[j - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    mus = [math.fsum(row) for row in impacts]
    sigmas = [math.hypot(*row) for row in uncertainties]
    lowers = [mus[k] - z * sigmas[k] for k in range(n_scenarios)]
    uppers = [mus[k] + z * sigmas[k] for k in range(n_scenarios)]

    order = sorted(range(n_scenarios), key=lambda k: (mus[k], k))
    Q: list[float] = []
    L: list[float] = []
    U: list[float] = []
    for q in qs:
        if q == 0.0:
            chosen = order[0]
        else:
            cumulative = 0.0
            chosen = order[-1]
            for k in order:
                cumulative = math.fsum([cumulative, w[k]])
                if cumulative >= q:
                    chosen = k
                    break
        Q.append(float(mus[chosen]))
        L.append(float(lowers[chosen]))
        U.append(float(uppers[chosen]))

    return Q, L, U


def receptor_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Weighted per-receptor quantiles of scenario health impacts.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(Q, L, U)`` where, for each receptor ``i`` and scenario
    ``k``:

    * ``v[k] = H[k][i]``, ``l[k] = H[k][i] - z * W[k][i]``,
      ``u[k] = H[k][i] + z * W[k][i]``;
    * for each requested quantile ``q`` the scenarios are sorted by
      ``(value, k)`` ascending for each of the three value sets and the
      quantile scenario is the first one whose cumulative normalized
      weight is ``>= q`` (the first scenario in that order when
      ``q == 0``);
    * ``Q[j][i]``, ``L[j][i]`` and ``U[j][i]`` hold that scenario's
      ``v``, ``l`` and ``u`` values respectively.

    Each result is an M x N list of lists of floats, in ``quantiles``
    then receptor order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for j, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{j}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{j}] must be in [0, 1], got {value!r}")
        if j > 0 and value < qs[j - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    Q: list[list[float]] = [[0.0] * n_receptors for _ in qs]
    L: list[list[float]] = [[0.0] * n_receptors for _ in qs]
    U: list[list[float]] = [[0.0] * n_receptors for _ in qs]

    for i in range(n_receptors):
        values = [impacts[k][i] for k in range(n_scenarios)]
        lowers = [values[k] - z * uncertainties[k][i] for k in range(n_scenarios)]
        uppers = [values[k] + z * uncertainties[k][i] for k in range(n_scenarios)]

        orders = [
            sorted(range(n_scenarios), key=lambda k, src=src: (src[k], k))
            for src in (values, lowers, uppers)
        ]
        chosen_per_q: list[tuple[int, int, int]] = []
        for q in qs:
            picks = []
            for order in orders:
                if q == 0.0:
                    chosen = order[0]
                else:
                    cumulative = 0.0
                    chosen = order[-1]
                    for k in order:
                        cumulative = math.fsum([cumulative, w[k]])
                        if cumulative >= q:
                            chosen = k
                            break
                picks.append(chosen)
            chosen_per_q.append((picks[0], picks[1], picks[2]))

        for j, (qk, lk, uk) in enumerate(chosen_per_q):
            Q[j][i] = float(values[qk])
            L[j][i] = float(lowers[lk])
            U[j][i] = float(uppers[uk])

    return Q, L, U


def receptor_risk_interval(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float], list[list[float]]]:
    """Per-receptor interval and exceedance risks, weighted over scenarios.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(M, R, lower, upper, probabilities)`` where, per receptor
    ``i``:

    * ``M[i] = fsum(w[k] * H[k][i])``,
    * ``V[i] = fsum(w[k] * ((H[k][i] - M[i]) ** 2 + W[k][i] ** 2))``,
    * ``R[i] = z * sqrt(V[i])``,
    * ``lower[i] = M[i] - R[i]``, ``upper[i] = M[i] + R[i]``,
    * ``q(k, i, j) = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``probabilities[i][j] = fsum(w[k] * q(k, i, j) for k in range(K))``.

    The first four results are N-long lists of floats and
    ``probabilities`` is an N x 3 list of lists of floats, in receptor
    then threshold order, all unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    M: list[float] = []
    V: list[float] = []
    for i in range(n_receptors):
        mean = math.fsum(w[k] * impacts[k][i] for k in range(n_scenarios))
        variance = math.fsum(
            w[k] * ((impacts[k][i] - mean) ** 2 + uncertainties[k][i] ** 2)
            for k in range(n_scenarios)
        )
        M.append(mean)
        V.append(variance)

    R = [z * math.sqrt(variance) for variance in V]
    lower = [M[i] - R[i] for i in range(n_receptors)]
    upper = [M[i] + R[i] for i in range(n_receptors)]

    probabilities: list[list[float]] = []
    for i in range(n_receptors):
        row: list[float] = []
        for j, level in enumerate(levels):
            terms: list[float] = []
            for k in range(n_scenarios):
                mu = impacts[k][i]
                sigma = uncertainties[k][i]
                if sigma > 0:
                    q = 0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2)))
                else:
                    q = 1.0 if level <= mu else 0.0
                terms.append(w[k] * q)
            row.append(math.fsum(terms))
        probabilities.append(row)

    return M, R, lower, upper, probabilities


def receptor_risk_share(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Per-scenario, per-receptor share of aggregate exceedance risk.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``(PS, ES)`` where, with ``mu = H[k][i]``,
    ``sigma = W[k][i]`` and ``t = thresholds[j]``:

    * when ``sigma > 0``, with ``a = (t - mu) / sigma``:
      ``q = 0.5 * erfc(a / sqrt(2))`` and
      ``e = sigma * exp(-a * a / 2) / sqrt(2 * pi) + (mu - t) * q``;
    * when ``sigma == 0``: ``q = 1.0`` if ``t <= mu`` else ``0.0``, and
      ``e = max(mu - t, 0.0)``;
    * ``p[k][i][j] = w[k] * q``, ``r[k][i][j] = w[k] * e``;
    * ``P[j] = fsum(p[k][i][j] for k in range(K) for i in range(N))``
      and ``E[j] = fsum(r[k][i][j] for k in range(K) for i in range(N))``;
    * ``PS[k][i][j] = p[k][i][j] / P[j]`` and
      ``ES[k][i][j] = r[k][i][j] / E[j]``, each taken as ``0.0`` when its
      denominator is ``0.0``.

    Both results are K x N x 3 lists of lists of lists of floats, in
    scenario, receptor then threshold order, unrounded. Summing either
    result over ``k`` and ``i`` yields 1 for each threshold whose
    denominator is positive.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    sqrt_2pi = math.sqrt(2.0 * math.pi)
    p: list[list[list[float]]] = []
    r: list[list[list[float]]] = []
    for k in range(n_scenarios):
        p_rows: list[list[float]] = []
        r_rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            p_row: list[float] = []
            r_row: list[float] = []
            for level in levels:
                if sigma > 0:
                    a = (level - mu) / sigma
                    q = 0.5 * math.erfc(a / math.sqrt(2))
                    e = sigma * math.exp(-a * a / 2.0) / sqrt_2pi + (mu - level) * q
                else:
                    q = 1.0 if level <= mu else 0.0
                    e = max(mu - level, 0.0)
                p_row.append(w[k] * q)
                r_row.append(w[k] * e)
            p_rows.append(p_row)
            r_rows.append(r_row)
        p.append(p_rows)
        r.append(r_rows)

    totals_p = [
        math.fsum(
            p[k][i][j]
            for k in range(n_scenarios)
            for i in range(n_receptors)
        )
        for j in range(3)
    ]
    totals_e = [
        math.fsum(
            r[k][i][j]
            for k in range(n_scenarios)
            for i in range(n_receptors)
        )
        for j in range(3)
    ]

    PS: list[list[list[float]]] = []
    ES: list[list[list[float]]] = []
    for k in range(n_scenarios):
        ps_rows: list[list[float]] = []
        es_rows: list[list[float]] = []
        for i in range(n_receptors):
            ps_row: list[float] = []
            es_row: list[float] = []
            for j in range(3):
                if totals_p[j] == 0.0:
                    ps_row.append(0.0)
                else:
                    ps_row.append(p[k][i][j] / totals_p[j])
                if totals_e[j] == 0.0:
                    es_row.append(0.0)
                else:
                    es_row.append(r[k][i][j] / totals_e[j])
            ps_rows.append(ps_row)
            es_rows.append(es_row)
        PS.append(ps_rows)
        ES.append(es_rows)

    return PS, ES


def receptor_risk_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[list[float]]]:
    """Weighted quantiles of per-receptor exceedance probabilities.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``Q`` where, with ``mu = H[k][i]``, ``sigma = W[k][i]`` and
    ``t = thresholds[j]``:

    * ``v(k, i, j) = 0.5 * erfc((t - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``t <= mu`` else ``0.0``;
    * for each receptor ``i`` and threshold ``j`` the scenarios are
      sorted by ``(v(k, i, j), k)`` ascending and, for each requested
      quantile ``r``, the quantile scenario is the first one whose
      cumulative normalized weight is ``>= r`` (the first scenario in
      that order when ``r == 0``);
    * ``Q[m][i][j]`` holds that scenario's ``v`` value.

    ``Q`` is an M x N x 3 list of lists of lists of floats, in
    ``quantiles``, receptor then threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1], got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    probabilities: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            row: list[float] = []
            for level in levels:
                if sigma > 0:
                    row.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    row.append(1.0 if level <= mu else 0.0)
            rows.append(row)
        probabilities.append(rows)

    Q: list[list[list[float]]] = [
        [[0.0] * 3 for _ in range(n_receptors)] for _ in qs
    ]
    for i in range(n_receptors):
        for j in range(3):
            values = [probabilities[k][i][j] for k in range(n_scenarios)]
            order = sorted(range(n_scenarios), key=lambda k: (values[k], k))
            for m, q in enumerate(qs):
                if q == 0.0:
                    chosen = order[0]
                else:
                    cumulative = 0.0
                    chosen = order[-1]
                    for k in order:
                        cumulative = math.fsum([cumulative, w[k]])
                        if cumulative >= q:
                            chosen = k
                            break
                Q[m][i][j] = float(values[chosen])

    return Q


def sensitivity(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[
    list[float],
    list[float],
    list[list[float]],
    list[list[float]],
]:
    """Per-receptor mean, spread and per-scenario sensitivity contributions.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(M, R, C, S)`` where, per receptor ``i``:

    * ``M[i] = fsum(w[k] * H[k][i])``,
    * ``V[i] = fsum(w[k] * ((H[k][i] - M[i]) ** 2 + W[k][i] ** 2))``,
    * ``R[i] = z * sqrt(V[i])``,
    * ``C[k][i] = w[k] * (H[k][i] - M[i])``,
    * ``S[k][i] = 0.0`` when ``V[i] == 0``, otherwise
      ``w[k] * W[k][i] ** 2 / V[i]``.

    ``M`` and ``R`` are N-long lists of floats and ``C`` and ``S`` are
    K x N lists of lists of floats, in scenario then receptor order, all
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    M: list[float] = []
    V: list[float] = []
    for i in range(n_receptors):
        mean = math.fsum(w[k] * impacts[k][i] for k in range(n_scenarios))
        variance = math.fsum(
            w[k] * ((impacts[k][i] - mean) ** 2 + uncertainties[k][i] ** 2)
            for k in range(n_scenarios)
        )
        M.append(mean)
        V.append(variance)

    R = [z * math.sqrt(variance) for variance in V]

    C: list[list[float]] = []
    S: list[list[float]] = []
    for k in range(n_scenarios):
        c_row: list[float] = []
        s_row: list[float] = []
        for i in range(n_receptors):
            c_row.append(w[k] * (impacts[k][i] - M[i]))
            if V[i] == 0:
                s_row.append(0.0)
            else:
                s_row.append(w[k] * uncertainties[k][i] ** 2 / V[i])
        C.append(c_row)
        S.append(s_row)

    return M, R, C, S


def any_receptor_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    """Probability that at least one receptor exceeds each threshold.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``p`` where:

    * ``q(k, j, i) = 0.5 * erfc((thresholds[j] - H[k][i])
      / (W[k][i] * sqrt(2)))`` when ``W[k][i] > 0``, otherwise ``1.0`` if
      ``thresholds[j] <= H[k][i]`` else ``0.0``;
    * ``a[k][j] = 1 - prod(1 - q(k, j, i) for i in range(N))`` is the
      probability that at least one receptor of scenario ``k`` exceeds
      ``thresholds[j]``;
    * ``p[j] = fsum(w[k] * a[k][j] for k in range(K))``.

    ``p`` is a 3-long list of floats in threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[float]] = []
    for k in range(n_scenarios):
        row: list[float] = []
        for level in levels:
            q = (
                0.5
                * math.erfc(
                    (level - impacts[k][i])
                    / (uncertainties[k][i] * math.sqrt(2))
                )
                if uncertainties[k][i] > 0
                else (1.0 if level <= impacts[k][i] else 0.0)
                for i in range(n_receptors)
            )
            row.append(1.0 - math.prod(1.0 - value for value in q))
        per_scenario.append(row)

    p = [
        math.fsum(w[k] * per_scenario[k][j] for k in range(n_scenarios))
        for j in range(3)
    ]

    return p


def at_least_count_probability(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    min_count: int = 1,
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[float]:
    """Probability that at least ``min_count`` receptors exceed each threshold.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    min_count: minimum number of exceeding receptors; a non-bool int
        >= 1 (default 1).
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Assuming the receptor errors within one scenario are mutually
    independent, returns ``p`` where, with ``t = thresholds[j]``:

    * ``q(k, i, j) = 0.5 * erfc((t - H[k][i]) / (W[k][i] * sqrt(2)))``
      when ``W[k][i] > 0``, otherwise ``1.0`` if ``t <= H[k][i]`` else
      ``0.0``;
    * ``d`` is the (N + 1)-long probability mass function of the number
      of exceeding receptors for scenario ``k`` and threshold ``j``: it
      starts as ``[1.0, 0.0, ...]`` and folds the receptors in one at a
      time with ``n[r] = d[r] * (1 - q) + (d[r - 1] * q if r > 0 else
      0.0)``;
    * ``a[k][j] = fsum(d[min_count:])`` is the probability that at least
      ``min_count`` receptors of scenario ``k`` exceed ``thresholds[j]``
      (``0.0`` when ``min_count > N``);
    * ``p[j] = fsum(w[k] * a[k][j] for k in range(K))``.

    ``p`` is a 3-long list of floats in threshold order, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(min_count, int) or isinstance(min_count, bool):
        raise TypeError(
            f"min_count must be an int, got {type(min_count).__name__}"
        )
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[float]] = []
    for k in range(n_scenarios):
        row: list[float] = []
        for level in levels:
            d = [1.0] + [0.0] * n_receptors
            for i in range(n_receptors):
                sigma = uncertainties[k][i]
                if sigma > 0:
                    q = 0.5 * math.erfc(
                        (level - impacts[k][i]) / (sigma * math.sqrt(2))
                    )
                else:
                    q = 1.0 if level <= impacts[k][i] else 0.0
                n = [0.0] * (n_receptors + 1)
                for r in range(i + 2):
                    n[r] = d[r] * (1.0 - q) + (d[r - 1] * q if r > 0 else 0.0)
                d = n
            row.append(math.fsum(d[min_count:]))
        per_scenario.append(row)

    p = [
        math.fsum(w[k] * per_scenario[k][j] for k in range(n_scenarios))
        for j in range(3)
    ]

    return p


def receptor_mitigation_plan(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    options: list | tuple,
    budget: float,
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[
    list[int],
    float,
    float,
    tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]],
]:
    """Choose one mitigation option per receptor under a budget.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    options: list or tuple of length N (one entry per receptor); each
        ``options[i]`` must be a non-empty list or tuple of pairs, and each
        pair must be a 2-element list or tuple ``(r, c)`` with ``r`` the
        finite, non-negative reduction applied to receptor ``i`` and ``c``
        the finite, non-negative cost of choosing that option.
    budget: finite, non-negative total cost limit; only combinations whose
        ``fsum`` of chosen costs does not exceed the budget are feasible.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    For every feasible choice ``choice`` of one option per receptor, with
    ``(r_i, c_i) = options[i][choice[i]]`` and
    ``total_cost = fsum(c_i for i in range(N))``, the reduced impact
    matrix ``H'`` has ``H'[k][i] = H[k][i] - r_i`` and
    ``(M, R, L, U) = receptor_excess_interval(H', W, thresholds, weights,
    z)``. The plan scores ``score = fsum(max(U[i][j], 0.0) for i in
    range(N) for j in range(3))`` and
    ``trigger = sum(1 for i in range(N) for j in range(3) if L[i][j] >
    0)``. The feasible plan minimizing the tuple
    ``(score, trigger, total_cost, choice)`` in lexicographic order is
    selected.
    Returns ``(choice, total_cost, score, interval)`` where ``choice`` is
    an N-long list of 0-based option indices, ``total_cost`` and ``score``
    are floats and ``interval`` is the ``(M, R, L, U)`` tuple of the
    selected plan, all unrounded. Raises ``ValueError`` when no
    combination fits the budget.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    n_receptors = len(impacts[0])

    if not isinstance(options, (list, tuple)):
        raise TypeError(
            f"options must be a list or tuple, got {type(options).__name__}"
        )
    if len(options) != n_receptors:
        raise ValueError(
            f"options length must equal receptor count {n_receptors}, "
            f"got {len(options)}"
        )
    reductions: list[list[float]] = []
    costs: list[list[float]] = []
    for i, group in enumerate(options):
        if not isinstance(group, (list, tuple)):
            raise TypeError(
                f"options[{i}] must be a list or tuple, "
                f"got {type(group).__name__}"
            )
        if len(group) == 0:
            raise ValueError(f"options[{i}] must not be empty")
        group_reductions: list[float] = []
        group_costs: list[float] = []
        for q, item in enumerate(group):
            if not isinstance(item, (list, tuple)):
                raise TypeError(
                    f"options[{i}][{q}] must be a list or tuple, "
                    f"got {type(item).__name__}"
                )
            if len(item) != 2:
                raise ValueError(
                    f"options[{i}][{q}] must have exactly 2 elements, "
                    f"got {len(item)}"
                )
            r_raw, c_raw = item
            if not _is_number(r_raw):
                raise TypeError(
                    f"options[{i}][{q}][0] must be an int or float, "
                    f"got {type(r_raw).__name__}"
                )
            if not _is_number(c_raw):
                raise TypeError(
                    f"options[{i}][{q}][1] must be an int or float, "
                    f"got {type(c_raw).__name__}"
                )
            r = float(r_raw)
            c = float(c_raw)
            _check_finite(f"options[{i}][{q}][0]", r)
            _check_finite(f"options[{i}][{q}][1]", c)
            if r < 0:
                raise ValueError(
                    f"options[{i}][{q}][0] must be >= 0, got {r!r}"
                )
            if c < 0:
                raise ValueError(
                    f"options[{i}][{q}][1] must be >= 0, got {c!r}"
                )
            group_reductions.append(r)
            group_costs.append(c)
        reductions.append(group_reductions)
        costs.append(group_costs)

    if not _is_number(budget):
        raise TypeError(f"budget must be an int or float, got {type(budget).__name__}")
    budget = float(budget)
    _check_finite("budget", budget)
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget!r}")

    # Validate W, thresholds, weights and z under the same contract as
    # receptor_excess_interval before enumerating any combination.
    receptor_excess_interval(impacts, W, thresholds, weights, z)

    best_key: tuple[float, int, float, tuple[int, ...]] | None = None
    best_choice: list[int] = []
    best_total_cost = 0.0
    best_score = 0.0
    best_interval: tuple[
        list[list[float]], list[list[float]], list[list[float]], list[list[float]]
    ] | None = None

    for choice_tuple in product(*(range(len(group)) for group in costs)):
        total_cost = math.fsum(
            costs[i][choice_tuple[i]] for i in range(n_receptors)
        )
        if total_cost > budget:
            continue
        adjusted = [
            [
                impacts[k][i] - reductions[i][choice_tuple[i]]
                for i in range(n_receptors)
            ]
            for k in range(len(impacts))
        ]
        M, R, L, U = receptor_excess_interval(
            adjusted, W, thresholds, weights, z
        )
        score = math.fsum(
            max(U[i][j], 0.0) for i in range(n_receptors) for j in range(3)
        )
        trigger = sum(
            1 for i in range(n_receptors) for j in range(3) if L[i][j] > 0
        )
        key = (score, trigger, total_cost, choice_tuple)
        if best_key is None or key < best_key:
            best_key = key
            best_choice = list(choice_tuple)
            best_total_cost = total_cost
            best_score = score
            best_interval = (M, R, L, U)

    if best_key is None:
        raise ValueError(f"no mitigation combination fits the budget {budget!r}")

    return best_choice, best_total_cost, best_score, best_interval


def receptor_mitigation_frontier(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    options: list | tuple,
    budget: float,
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> list[
    tuple[
        list[int],
        float,
        float,
        int,
        float,
        tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]],
    ]
]:
    """Non-dominated mitigation plans over excess, spread, triggers and cost.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    options: list or tuple of length N (one entry per receptor); each
        ``options[i]`` must be a non-empty list or tuple of pairs, and each
        pair must be a 2-element list or tuple ``(r, c)`` with ``r`` the
        finite, non-negative reduction applied to receptor ``i`` and ``c``
        the finite, non-negative cost of choosing that option.
    budget: finite, non-negative total cost limit; only combinations whose
        ``fsum`` of chosen costs does not exceed the budget are feasible.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    For every feasible choice ``choice`` of one option per receptor, with
    ``(r_i, c_i) = options[i][choice[i]]`` and
    ``total_cost = fsum(c_i for i in range(N)) <= budget``, the reduced
    impact matrix ``H'`` has ``H'[k][i] = H[k][i] - r_i`` and
    ``(M, R, L, U) = receptor_excess_interval(H', W, thresholds, weights,
    z)``. The plan scores:

    * ``excess = fsum(max(U[i][j], 0.0) for i in range(N) for j in
      range(3))``;
    * ``spread = fsum(R[i][j] for i in range(N) for j in range(3))``;
    * ``triggers = sum(1 for i in range(N) for j in range(3) if
      L[i][j] > 0)``.

    A feasible plan is kept only when no other feasible plan is no worse
    in every one of ``(excess, spread, triggers, total_cost)`` and strictly
    better in at least one. The retained plans are sorted ascending by
    ``(excess, spread, triggers, total_cost, choice)``.
    Returns a list of ``(choice, total_cost, excess, spread, triggers,
    interval)`` tuples where ``choice`` is an N-long list of 0-based option
    indices, ``total_cost``, ``excess`` and ``spread`` are floats,
    ``triggers`` is an int and ``interval`` is the ``(M, R, L, U)`` tuple
    of that plan, all unrounded. Raises ``ValueError`` when no combination
    fits the budget.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    n_receptors = len(impacts[0])

    if not isinstance(options, (list, tuple)):
        raise TypeError(
            f"options must be a list or tuple, got {type(options).__name__}"
        )
    if len(options) != n_receptors:
        raise ValueError(
            f"options length must equal receptor count {n_receptors}, "
            f"got {len(options)}"
        )
    reductions: list[list[float]] = []
    costs: list[list[float]] = []
    for i, group in enumerate(options):
        if not isinstance(group, (list, tuple)):
            raise TypeError(
                f"options[{i}] must be a list or tuple, "
                f"got {type(group).__name__}"
            )
        if len(group) == 0:
            raise ValueError(f"options[{i}] must not be empty")
        group_reductions: list[float] = []
        group_costs: list[float] = []
        for q, item in enumerate(group):
            if not isinstance(item, (list, tuple)):
                raise TypeError(
                    f"options[{i}][{q}] must be a list or tuple, "
                    f"got {type(item).__name__}"
                )
            if len(item) != 2:
                raise ValueError(
                    f"options[{i}][{q}] must have exactly 2 elements, "
                    f"got {len(item)}"
                )
            r_raw, c_raw = item
            if not _is_number(r_raw):
                raise TypeError(
                    f"options[{i}][{q}][0] must be an int or float, "
                    f"got {type(r_raw).__name__}"
                )
            if not _is_number(c_raw):
                raise TypeError(
                    f"options[{i}][{q}][1] must be an int or float, "
                    f"got {type(c_raw).__name__}"
                )
            r = float(r_raw)
            c = float(c_raw)
            _check_finite(f"options[{i}][{q}][0]", r)
            _check_finite(f"options[{i}][{q}][1]", c)
            if r < 0:
                raise ValueError(
                    f"options[{i}][{q}][0] must be >= 0, got {r!r}"
                )
            if c < 0:
                raise ValueError(
                    f"options[{i}][{q}][1] must be >= 0, got {c!r}"
                )
            group_reductions.append(r)
            group_costs.append(c)
        reductions.append(group_reductions)
        costs.append(group_costs)

    if not _is_number(budget):
        raise TypeError(f"budget must be an int or float, got {type(budget).__name__}")
    budget = float(budget)
    _check_finite("budget", budget)
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget!r}")

    # Validate W, thresholds, weights and z under the same contract as
    # receptor_excess_interval before enumerating any combination.
    receptor_excess_interval(impacts, W, thresholds, weights, z)

    # Each entry: (excess, spread, triggers, total_cost, choice_tuple,
    # interval).
    candidates: list[
        tuple[float, float, int, float, tuple[int, ...], tuple]
    ] = []

    for choice_tuple in product(*(range(len(group)) for group in costs)):
        total_cost = math.fsum(
            costs[i][choice_tuple[i]] for i in range(n_receptors)
        )
        if total_cost > budget:
            continue
        adjusted = [
            [
                impacts[k][i] - reductions[i][choice_tuple[i]]
                for i in range(n_receptors)
            ]
            for k in range(len(impacts))
        ]
        M, R, L, U = receptor_excess_interval(
            adjusted, W, thresholds, weights, z
        )
        excess = math.fsum(
            max(U[i][j], 0.0) for i in range(n_receptors) for j in range(3)
        )
        spread = math.fsum(
            R[i][j] for i in range(n_receptors) for j in range(3)
        )
        triggers = sum(
            1 for i in range(n_receptors) for j in range(3) if L[i][j] > 0
        )
        candidates.append(
            (excess, spread, triggers, total_cost, choice_tuple, (M, R, L, U))
        )

    if not candidates:
        raise ValueError(f"no mitigation combination fits the budget {budget!r}")

    frontier: list[
        tuple[float, float, int, float, tuple[int, ...], tuple]
    ] = []
    for candidate in candidates:
        s_excess, s_spread, s_triggers, s_cost, _, _ = candidate
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            o_excess, o_spread, o_triggers, o_cost, _, _ = other
            if (
                o_excess <= s_excess
                and o_spread <= s_spread
                and o_triggers <= s_triggers
                and o_cost <= s_cost
                and (
                    o_excess < s_excess
                    or o_spread < s_spread
                    or o_triggers < s_triggers
                    or o_cost < s_cost
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))

    return [
        (
            list(choice_tuple),
            total_cost,
            excess,
            spread,
            triggers,
            interval,
        )
        for excess, spread, triggers, total_cost, choice_tuple, interval in frontier
    ]


def policy_recommend(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budget: float,
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> list[tuple]:
    """Pair each non-dominated mitigation plan with its best run policy.

    ``H``, ``W``, ``thresholds``, ``options``, ``budget``, ``weights`` and
    ``z`` share the contract of :func:`receptor_mitigation_frontier`
    (every ``TypeError`` and ``ValueError`` is inherited verbatim);
    ``quantiles`` and ``policies`` share the contract of
    :func:`receptor_count_run_policy_batch`, which in particular requires
    ``K >= 2`` scenarios.

    Let ``F = receptor_mitigation_frontier(H, W, thresholds, options,
    budget, weights, z)``. For each frontier entry
    ``(choice, C, S, D, G, interval)`` the reduced impact matrix ``H'``
    with ``H'[k][i] = H[k][i] - options[i][choice[i]][0]`` replaces ``H``
    in a call to :func:`receptor_count_run_policy_batch` returning
    ``(results, scores, order)``; with ``p = order[0]``,
    ``o = results[p]``, ``a = max(o[1])`` and ``s = scores[p]`` the entry
    contributes ``(choice, p, C, S, D, G, a, s, interval, o)``.

    Returns the list of such tuples sorted ascending by
    ``(S, D, G, C, -a, -s, choice, p)``. ``choice`` is an N-long list of
    ints, ``p`` is an int, ``C``, ``S``, ``D`` and ``s`` are floats, ``G``
    and ``a`` are ints, ``interval`` is the ``(M, R, L, U)`` tuple from
    the frontier and ``o`` is the winning policy's
    ``(runs, levels, triggers, first, cumulative)`` tuple, all unrounded.
    """
    frontier = receptor_mitigation_frontier(
        H, W, thresholds, options, budget, weights, z
    )

    impacts = _validate_matrix("H", H, non_negative=False)
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    reductions = [[float(pair[0]) for pair in group] for group in options]

    rows: list[tuple] = []
    for choice, C, S, D, G, interval in frontier:
        adjusted = [
            [
                impacts[k][i] - reductions[i][choice[i]]
                for i in range(n_receptors)
            ]
            for k in range(n_scenarios)
        ]
        results, scores, order = receptor_count_run_policy_batch(
            adjusted, W, thresholds, quantiles, policies
        )
        p = order[0]
        o = results[p]
        a = max(o[1])
        s = scores[p]
        rows.append((choice, p, C, S, D, G, a, s, interval, o))

    rows.sort(
        key=lambda item: (
            item[3],
            item[4],
            item[5],
            item[2],
            -item[6],
            -item[7],
            item[0],
            item[1],
        )
    )

    return rows


def robust_policy_summary(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budget: float,
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    z: float = 1.96,
) -> tuple[
    list[int],
    int,
    list[int],
    list[tuple[float, float, int, float, int, float]],
    list[
        tuple[
            list[list[float]],
            list[list[float]],
            list[list[float]],
            list[list[float]],
        ]
    ],
    tuple[list[int], list[int], list[int | None], list[list[float]], list[list[float]]],
]:
    """Pick a budget-feasible plan and run policy robust across weight sets.

    ``H``, ``W``, ``thresholds``, ``quantiles``, ``policies``, ``options``,
    ``budget`` and ``z`` share the contract of :func:`policy_recommend`
    (every ``TypeError`` and ``ValueError`` is inherited verbatim).
    ``weight_sets`` is a non-empty L x K list or tuple matrix of non-bool,
    finite, non-negative numbers whose rows each have a positive ``fsum``;
    every row is normalized by its ``fsum`` before use. A wrong container,
    row or element type raises ``TypeError``; an empty ``weight_sets``, a
    ragged or wrong-length row, a non-finite or negative entry or a
    zero-sum row raises ``ValueError``.

    Every budget-feasible ``(choice, p)`` pair is enumerated: ``choice``
    selects one option per receptor with ``C = fsum`` of the chosen costs
    ``<= budget`` and ``p`` ranges over all policies. With
    ``H'[k][i] = H[k][i] - options[i][choice[i]][0]``, each normalized
    weight row ``w`` yields ``I = (M, R, L, U) =
    receptor_excess_interval(H', W, thresholds, w, z)`` and policy ``p``
    yields ``o = receptor_count_run_warning(H', W, thresholds, quantiles,
    durations, level=level, minimum_count=minimum_count)`` with the
    policy's own parameters. Let ``S = fsum(max(U[i][j], 0.0))``,
    ``D = fsum(R[i][j])`` and ``G = sum(1 if L[i][j] > 0)`` over
    ``i < N, j < 3``, ``a = max(o[1])`` and ``s = fsum(o[4][K - 1])``.
    For each weight row the pairs are assigned 0-based ranks by ascending
    ``(S, D, G, C, -a, -s, choice, p)``; ``ranks`` collects the L ranks of
    a pair. The pair minimizing ``(max(ranks), fsum(ranks), choice, p)``
    wins.

    Returns ``(choice, p, ranks, metrics, intervals, o)`` where ``choice``
    is an N-long ``list[int]`` of 0-based option indices, ``p`` is an int,
    ``ranks`` is an L-long ``list[int]``, ``metrics`` holds the L
    ``(S, D, G, C, a, s)`` tuples, ``intervals`` holds the L
    ``(M, R, L, U)`` tuples and ``o`` is the winning policy's
    ``(runs, levels, triggers, first, cumulative)`` tuple, all unrounded.
    Raises ``ValueError`` when no combination fits the budget.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])

    if not isinstance(weight_sets, (list, tuple)):
        raise TypeError(
            f"weight_sets must be a list or tuple, "
            f"got {type(weight_sets).__name__}"
        )
    if len(weight_sets) == 0:
        raise ValueError("weight_sets must not be empty")
    normalized_sets: list[list[float]] = []
    for l, row in enumerate(weight_sets):
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                f"weight_sets[{l}] must be a list or tuple, "
                f"got {type(row).__name__}"
            )
        if len(row) != n_scenarios:
            raise ValueError(
                f"weight_sets[{l}] length must equal scenario count "
                f"{n_scenarios}, got {len(row)}"
            )
        raw: list[float] = []
        for k, item in enumerate(row):
            if not _is_number(item):
                raise TypeError(
                    f"weight_sets[{l}][{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weight_sets[{l}][{k}]", value)
            if value < 0:
                raise ValueError(
                    f"weight_sets[{l}][{k}] must be >= 0, got {value!r}"
                )
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(
                f"weight_sets[{l}] sum must be > 0, got {total_weight!r}"
            )
        normalized_sets.append([value / total_weight for value in raw])

    if not isinstance(options, (list, tuple)):
        raise TypeError(
            f"options must be a list or tuple, got {type(options).__name__}"
        )
    if len(options) != n_receptors:
        raise ValueError(
            f"options length must equal receptor count {n_receptors}, "
            f"got {len(options)}"
        )
    reductions: list[list[float]] = []
    costs: list[list[float]] = []
    for i, group in enumerate(options):
        if not isinstance(group, (list, tuple)):
            raise TypeError(
                f"options[{i}] must be a list or tuple, "
                f"got {type(group).__name__}"
            )
        if len(group) == 0:
            raise ValueError(f"options[{i}] must not be empty")
        group_reductions: list[float] = []
        group_costs: list[float] = []
        for q, item in enumerate(group):
            if not isinstance(item, (list, tuple)):
                raise TypeError(
                    f"options[{i}][{q}] must be a list or tuple, "
                    f"got {type(item).__name__}"
                )
            if len(item) != 2:
                raise ValueError(
                    f"options[{i}][{q}] must have exactly 2 elements, "
                    f"got {len(item)}"
                )
            r_raw, c_raw = item
            if not _is_number(r_raw):
                raise TypeError(
                    f"options[{i}][{q}][0] must be an int or float, "
                    f"got {type(r_raw).__name__}"
                )
            if not _is_number(c_raw):
                raise TypeError(
                    f"options[{i}][{q}][1] must be an int or float, "
                    f"got {type(c_raw).__name__}"
                )
            r = float(r_raw)
            c = float(c_raw)
            _check_finite(f"options[{i}][{q}][0]", r)
            _check_finite(f"options[{i}][{q}][1]", c)
            if r < 0:
                raise ValueError(
                    f"options[{i}][{q}][0] must be >= 0, got {r!r}"
                )
            if c < 0:
                raise ValueError(
                    f"options[{i}][{q}][1] must be >= 0, got {c!r}"
                )
            group_reductions.append(r)
            group_costs.append(c)
        reductions.append(group_reductions)
        costs.append(group_costs)

    if not _is_number(budget):
        raise TypeError(f"budget must be an int or float, got {type(budget).__name__}")
    budget = float(budget)
    _check_finite("budget", budget)
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget!r}")

    # Validate W, thresholds and z under the same contract as
    # receptor_excess_interval before enumerating any combination.
    receptor_excess_interval(impacts, W, thresholds, None, z)

    # Each entry: (choice, p, total_cost, a, s, o, intervals, row_metrics)
    # where intervals and row_metrics hold one entry per weight row.
    entries: list[tuple] = []
    for choice_tuple in product(*(range(len(group)) for group in costs)):
        total_cost = math.fsum(
            costs[i][choice_tuple[i]] for i in range(n_receptors)
        )
        if total_cost > budget:
            continue
        adjusted = [
            [
                impacts[k][i] - reductions[i][choice_tuple[i]]
                for i in range(n_receptors)
            ]
            for k in range(n_scenarios)
        ]
        results, _, _ = receptor_count_run_policy_batch(
            adjusted, W, thresholds, quantiles, policies
        )
        intervals: list[tuple] = []
        row_metrics: list[tuple[float, float, int]] = []
        for w in normalized_sets:
            M, R, L, U = receptor_excess_interval(adjusted, W, thresholds, w, z)
            excess = math.fsum(
                max(U[i][j], 0.0) for i in range(n_receptors) for j in range(3)
            )
            spread = math.fsum(
                R[i][j] for i in range(n_receptors) for j in range(3)
            )
            triggers = sum(
                1 for i in range(n_receptors) for j in range(3) if L[i][j] > 0
            )
            intervals.append((M, R, L, U))
            row_metrics.append((excess, spread, triggers))
        for p in range(len(results)):
            o = results[p]
            a = max(o[1])
            s = math.fsum(o[4][-1])
            entries.append(
                (list(choice_tuple), p, total_cost, a, s, o, intervals, row_metrics)
            )

    if not entries:
        raise ValueError(f"no mitigation combination fits the budget {budget!r}")

    n_rows = len(normalized_sets)
    ranks: list[list[int]] = [[0] * n_rows for _ in entries]
    for l in range(n_rows):
        order = sorted(
            range(len(entries)),
            key=lambda c: (
                entries[c][7][l][0],
                entries[c][7][l][1],
                entries[c][7][l][2],
                entries[c][2],
                -entries[c][3],
                -entries[c][4],
                entries[c][0],
                entries[c][1],
            ),
        )
        for rank, c in enumerate(order):
            ranks[c][l] = rank

    best = min(
        range(len(entries)),
        key=lambda c: (
            max(ranks[c]),
            math.fsum(ranks[c]),
            entries[c][0],
            entries[c][1],
        ),
    )
    choice, p, total_cost, a, s, o, intervals, row_metrics = entries[best]
    metrics = [
        (row_metrics[l][0], row_metrics[l][1], row_metrics[l][2], total_cost, a, s)
        for l in range(n_rows)
    ]

    return choice, p, ranks[best], metrics, intervals, o


def robust_policy_report(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budget: float,
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    z: float = 1.96,
) -> dict:
    """JSON-compatible summary of the robust plan and policy pick.

    Every parameter constraint and every ``TypeError``/``ValueError``
    condition is inherited verbatim from :func:`robust_policy_summary`;
    exceptions propagate unchanged. The function calls
    :func:`robust_policy_summary` exactly once, takes
    ``(choice, p, ranks, metrics, intervals, o)`` and returns a dict
    whose keys in order are ``choice``, ``policy``, ``robustness``,
    ``weights`` and ``warning``. ``choice`` is a list copy of the
    winning choice and ``policy`` is ``p``. ``robustness`` is
    ``[max(ranks), fsum(ranks), list(ranks)]``. ``weights`` follows the
    ``weight_sets`` order; entry ``l`` is ``[S, D, G, C, a, s, M, R, L,
    U]`` with the first six values taken from ``metrics[l]`` and the
    last four from ``intervals[l]`` copied as 2-D lists. ``warning``
    follows the ``[runs, levels, triggers, first, cumulative]`` order of
    ``o`` with every inner container recursively converted to a list and
    ``None`` preserved. No value is rounded.
    """
    choice, p, ranks, metrics, intervals, o = robust_policy_summary(
        H, W, thresholds, quantiles, policies, options, budget, weight_sets, z
    )

    def _as_lists(value: object) -> object:
        if isinstance(value, (list, tuple)):
            return [_as_lists(item) for item in value]
        return value

    weights: list[list] = []
    for l in range(len(metrics)):
        excess, spread, triggers, total_cost, a, s = metrics[l]
        M, R, L, U = intervals[l]
        weights.append(
            [
                excess,
                spread,
                triggers,
                total_cost,
                a,
                s,
                [list(row) for row in M],
                [list(row) for row in R],
                [list(row) for row in L],
                [list(row) for row in U],
            ]
        )

    runs, levels, warning_triggers, first, cumulative = o
    return {
        "choice": list(choice),
        "policy": p,
        "robustness": [max(ranks), math.fsum(ranks), list(ranks)],
        "weights": weights,
        "warning": [
            _as_lists(item)
            for item in (runs, levels, warning_triggers, first, cumulative)
        ],
    }


def _validate_strictly_increasing(name: str, values: object) -> list[float]:
    """Validate a non-empty, strictly increasing non-negative scalar axis."""
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"{name} must be a list or tuple, got {type(values).__name__}"
        )
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    result: list[float] = []
    for r, item in enumerate(values):
        if not _is_number(item):
            raise TypeError(
                f"{name}[{r}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        try:
            value = float(item)
        except OverflowError:
            raise ValueError(
                f"{name}[{r}] must be finite, got {item!r}"
            ) from None
        _check_finite(f"{name}[{r}]", value)
        if value < 0:
            raise ValueError(f"{name}[{r}] must be >= 0, got {value!r}")
        if r > 0 and value <= result[r - 1]:
            raise ValueError(
                f"{name} must be strictly increasing, got {[*result, value]!r}"
            )
        result.append(value)
    return result


def robust_policy_sensitivity(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budget: float,
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Sensitivity of the robust plan and policy pick to the interval width.

    Every parameter except ``zs`` shares the contract of
    :func:`robust_policy_report` (every ``TypeError`` and ``ValueError``
    is inherited verbatim; exceptions propagate unchanged). ``zs`` is a
    non-empty list or tuple of non-bool, finite, non-negative int/float
    values in strictly increasing order; a wrong container or element
    type raises ``TypeError``, while an empty ``zs``, a non-finite or
    negative entry, an int too large to convert to float or a
    non-increasing pair raises ``ValueError``.

    For each ``z`` in ``zs`` order the function calls
    :func:`robust_policy_report` exactly once with every other argument
    unchanged and collects the results as ``R``. Reports with an
    identical ``(choice, policy)`` pair are grouped; ``S`` holds, in
    first-occurrence order, one ``[choice, policy, count]`` entry per
    group with ``choice`` a list copy. Row ``r`` of ``D`` is the
    element-wise difference of ``R[r + 1]["robustness"][2]`` and
    ``R[r]["robustness"][2]``; row ``r`` of ``G`` is the element-wise
    difference of ``R[r + 1]["warning"][1]`` and ``R[r]["warning"][1]``.
    With a single ``z`` both ``D`` and ``G`` are empty.

    Returns a dict whose keys in order are ``z``, ``reports``,
    ``selection_counts``, ``rank_changes`` and ``level_changes``, with
    values ``zs`` converted to floats, ``R``, ``S``, ``D`` and ``G``.
    Everything follows the ``zs`` order and the original
    ``weight_sets``/``quantiles`` order; all containers are
    JSON-compatible, ``None`` is preserved and no value is rounded.
    """
    z_values = _validate_strictly_increasing("zs", zs)

    reports = [
        robust_policy_report(
            H, W, thresholds, quantiles, policies, options, budget,
            weight_sets, z,
        )
        for z in z_values
    ]

    selection_counts: list[list] = []
    group_index: dict[tuple, int] = {}
    for report in reports:
        key = (tuple(report["choice"]), report["policy"])
        index = group_index.get(key)
        if index is None:
            group_index[key] = len(selection_counts)
            selection_counts.append(
                [list(report["choice"]), report["policy"], 1]
            )
        else:
            selection_counts[index][2] += 1

    rank_changes: list[list[float]] = [
        [
            after - before
            for before, after in zip(
                reports[r]["robustness"][2], reports[r + 1]["robustness"][2]
            )
        ]
        for r in range(len(reports) - 1)
    ]
    level_changes: list[list[float]] = [
        [
            after - before
            for before, after in zip(
                reports[r]["warning"][1], reports[r + 1]["warning"][1]
            )
        ]
        for r in range(len(reports) - 1)
    ]

    return {
        "z": z_values,
        "reports": reports,
        "selection_counts": selection_counts,
        "rank_changes": rank_changes,
        "level_changes": level_changes,
    }


def robust_policy_grid(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Grid of robust plan and policy picks over budgets and interval widths.

    Every parameter except ``budgets`` and ``zs`` shares the contract of
    :func:`robust_policy_report` (every ``TypeError`` and ``ValueError``
    is inherited verbatim; exceptions propagate unchanged). ``budgets``
    and ``zs`` each follow the ``zs`` contract of
    :func:`robust_policy_sensitivity`: a non-empty list or tuple of
    non-bool, finite, non-negative int/float values in strictly
    increasing order; a wrong container or element type raises
    ``TypeError``, while an empty axis, a non-finite or negative entry,
    an int too large to convert to float or a non-increasing pair raises
    ``ValueError``.

    With ``budgets`` as the outer axis and ``zs`` as the inner axis the
    function calls :func:`robust_policy_report` exactly once per
    ``(budget, z)`` cell with every other argument unchanged and
    collects the results as the B x Z matrix ``R``. ``stability[b]``
    holds, in ``z`` order, the maximal consecutive segments of reports
    with an identical ``(choice, policy)`` pair, one
    ``[first_z, last_z, choice, policy]`` entry per segment with
    ``choice`` a list copy. ``rank_changes`` and ``level_changes`` are
    dicts whose keys in order are ``z`` and ``budget``: the ``z`` value
    is the B x (Z - 1) matrix of element-wise differences of
    ``R[b][r + 1]`` minus ``R[b][r]`` over ``robustness[2]`` and
    ``warning[1]`` respectively, and the ``budget`` value is the
    (B - 1) x Z matrix of the same differences of ``R[b + 1][r]`` minus
    ``R[b][r]``; an axis with a single point yields an empty dimension.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``reports``, ``stability``, ``rank_changes`` and ``level_changes``,
    with values ``budgets`` and ``zs`` converted to floats, ``R``, the
    per-budget segment lists and the two change dicts. Everything
    follows the ``budgets``/``zs`` order and the original
    ``weight_sets``/``quantiles`` order; all containers are
    JSON-compatible, ``None`` is preserved and no value is rounded.
    """
    budget_values = _validate_strictly_increasing("budgets", budgets)
    z_values = _validate_strictly_increasing("zs", zs)

    reports = [
        [
            robust_policy_report(
                H, W, thresholds, quantiles, policies, options,
                budget, weight_sets, z,
            )
            for z in z_values
        ]
        for budget in budget_values
    ]

    stability: list[list[list]] = []
    for row in reports:
        segments: list[list] = []
        for z_value, report in zip(z_values, row):
            if (
                segments
                and segments[-1][2] == report["choice"]
                and segments[-1][3] == report["policy"]
            ):
                segments[-1][1] = z_value
            else:
                segments.append(
                    [z_value, z_value, list(report["choice"]), report["policy"]]
                )
        stability.append(segments)

    def _changes(select) -> dict:
        return {
            "z": [
                [
                    [
                        after - before
                        for before, after in zip(
                            select(row[r]), select(row[r + 1])
                        )
                    ]
                    for r in range(len(z_values) - 1)
                ]
                for row in reports
            ],
            "budget": [
                [
                    [
                        after - before
                        for before, after in zip(
                            select(reports[b][r]), select(reports[b + 1][r])
                        )
                    ]
                    for r in range(len(z_values))
                ]
                for b in range(len(budget_values) - 1)
            ],
        }

    def _robustness_ranks(report: dict) -> list:
        return report["robustness"][2]

    def _warning_levels(report: dict) -> list:
        return report["warning"][1]

    rank_changes = _changes(_robustness_ranks)
    level_changes = _changes(_warning_levels)

    return {
        "budgets": budget_values,
        "z": z_values,
        "reports": reports,
        "stability": stability,
        "rank_changes": rank_changes,
        "level_changes": level_changes,
    }


def robust_policy_grid_turns(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Turning points of the robust grid state along both axes.

    Every parameter shares the contract of :func:`robust_policy_grid`
    (every ``TypeError`` and ``ValueError`` is inherited verbatim;
    exceptions propagate unchanged). :func:`robust_policy_grid` is
    called exactly once to obtain the grid ``G``.

    The state ``S[b][r]`` is ``[choice copy, policy, robustness[2]
    copy, warning[1] copy]`` taken from ``G["reports"][b][r]``: the
    plan, the policy, the per-weight-set ranks and the per-quantile
    warning levels. Adjacent states along the ``z`` axis are compared
    in ascending ``b`` then ``r = 1..Z - 1`` order, and adjacent states
    along the ``budget`` axis in ``b = 1..B - 1`` then ascending ``r``
    order. Whenever two adjacent states differ, an event
    ``[[before_b, before_r], [after_b, after_r], before_state,
    after_state, fields]`` is recorded, where ``fields`` names the
    changed components in the order ``choice``, ``policy``, ``ranks``,
    ``levels``.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``states``, ``z_changes`` and ``budget_changes``: the first two are
    float list copies of the same-named entries of ``G``, ``states`` is
    the B x Z state matrix and the last two are the event lists in the
    scan orders above, empty when nothing changes or the axis has a
    single point; every change is kept. All containers are
    JSON-compatible and no value is rounded.
    """
    grid = robust_policy_grid(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )
    reports = grid["reports"]
    n_budgets = len(grid["budgets"])
    n_zs = len(grid["z"])

    states: list[list[list]] = [
        [
            [
                list(report["choice"]),
                report["policy"],
                list(report["robustness"][2]),
                list(report["warning"][1]),
            ]
            for report in row
        ]
        for row in reports
    ]

    field_names = ("choice", "policy", "ranks", "levels")

    def _turn(
        before: list, after: list, b0: int, r0: int, b1: int, r1: int
    ) -> list | None:
        fields = [
            name
            for name, old, new in zip(field_names, before, after)
            if old != new
        ]
        if not fields:
            return None
        return [
            [b0, r0],
            [b1, r1],
            [list(before[0]), before[1], list(before[2]), list(before[3])],
            [list(after[0]), after[1], list(after[2]), list(after[3])],
            fields,
        ]

    z_changes: list[list] = []
    for b in range(n_budgets):
        for r in range(1, n_zs):
            event = _turn(states[b][r - 1], states[b][r], b, r - 1, b, r)
            if event is not None:
                z_changes.append(event)

    budget_changes: list[list] = []
    for b in range(1, n_budgets):
        for r in range(n_zs):
            event = _turn(states[b - 1][r], states[b][r], b - 1, r, b, r)
            if event is not None:
                budget_changes.append(event)

    return {
        "budgets": list(grid["budgets"]),
        "z": list(grid["z"]),
        "states": states,
        "z_changes": z_changes,
        "budget_changes": budget_changes,
    }


def robust_policy_grid_regions(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Orthogonally connected regions of equal states on the robust grid.

    Every parameter shares the contract of
    :func:`robust_policy_grid_turns` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_grid_turns` is called exactly once
    to obtain the grid ``G``.

    A cell ``[b, r]`` is adjacent only to its up, down, left and right
    neighbours, and two cells share a state when the corresponding
    entries of ``G["states"]`` compare equal. Cells are scanned with
    ``b`` outer and ``r`` inner, skipping cells already assigned; each
    flood fill gathers every orthogonally reachable cell with the same
    state into one region. Region ids increase from ``0`` in scan
    order, and each region's members are sorted by ``(b, r)``. Each
    region is a dict whose keys in order are ``id``, ``cells``,
    ``budget_bounds``, ``z_bounds`` and ``state``: ``cells`` lists the
    ``[b, r]`` members, ``budget_bounds`` is the two-element float list
    of the budgets at the extreme member ``b`` indices, ``z_bounds`` is
    the analogous list of ``z`` values for the extreme member ``r``
    indices, and ``state`` is a recursive list copy of the region's
    state.

    Two distinct regions with any orthogonally adjacent members form an
    edge ``[smaller id, larger id]``; edges are deduplicated and sorted
    by the two ids.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``states``, ``regions`` and ``adjacency``: the first three are
    recursive list copies of the same-named entries of ``G``,
    ``regions`` is ordered by id and ``adjacency`` is the edge list.
    All containers are JSON-compatible, ``None`` is preserved and no
    value is rounded.
    """
    grid = robust_policy_grid_turns(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )
    grid_states = grid["states"]
    budget_values = grid["budgets"]
    z_values = grid["z"]
    n_budgets = len(budget_values)
    n_zs = len(z_values)

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    region_of: dict[tuple[int, int], int] = {}
    regions: list[dict] = []
    for b0 in range(n_budgets):
        for r0 in range(n_zs):
            if (b0, r0) in region_of:
                continue
            region_id = len(regions)
            state = grid_states[b0][r0]
            cells: list[list[int]] = []
            pending = [(b0, r0)]
            region_of[(b0, r0)] = region_id
            while pending:
                b, r = pending.pop()
                cells.append([b, r])
                for nb, nr in (
                    (b - 1, r), (b + 1, r), (b, r - 1), (b, r + 1)
                ):
                    if not (0 <= nb < n_budgets and 0 <= nr < n_zs):
                        continue
                    if (nb, nr) in region_of:
                        continue
                    if grid_states[nb][nr] != state:
                        continue
                    region_of[(nb, nr)] = region_id
                    pending.append((nb, nr))
            cells.sort(key=lambda cell: (cell[0], cell[1]))
            b_min = cells[0][0]
            b_max = cells[-1][0]
            r_min = min(cell[1] for cell in cells)
            r_max = max(cell[1] for cell in cells)
            regions.append(
                {
                    "id": region_id,
                    "cells": cells,
                    "budget_bounds": [
                        float(budget_values[b_min]),
                        float(budget_values[b_max]),
                    ],
                    "z_bounds": [
                        float(z_values[r_min]),
                        float(z_values[r_max]),
                    ],
                    "state": _copy(state),
                }
            )

    edge_set: set[tuple[int, int]] = set()
    for b in range(n_budgets):
        for r in range(n_zs):
            current = region_of[(b, r)]
            if b + 1 < n_budgets:
                other = region_of[(b + 1, r)]
                if other != current:
                    edge_set.add(
                        (current, other) if current < other else (other, current)
                    )
            if r + 1 < n_zs:
                other = region_of[(b, r + 1)]
                if other != current:
                    edge_set.add(
                        (current, other) if current < other else (other, current)
                    )
    adjacency = [[small, large] for small, large in sorted(edge_set)]

    return {
        "budgets": _copy(budget_values),
        "z": _copy(z_values),
        "states": _copy(grid_states),
        "regions": regions,
        "adjacency": adjacency,
    }


def robust_policy_region_flow(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Flow of equal-state regions across the budget and z slices.

    Every parameter shares the contract of
    :func:`robust_policy_grid_regions` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_grid_regions` is called exactly
    once to obtain the grid ``G``; its regions give every cell
    ``[b, r]`` an id, and those ids form a B x Z mesh.

    A fixed-budget slice (one row, swept over ``r``) and a fixed-z slice
    (one column, swept over ``b``) are each compressed into the maximal
    runs of consecutive cells with one id; a run is
    ``[first, last, id, state_copy]`` with ``first`` and ``last`` the
    sweep indices at the run's ends and ``state_copy`` a recursive list
    copy of that region's state. Within one slice every run gets one
    ascending index in sweep order.

    Two runs from adjacent slices are joined when they share an id and
    their index intervals intersect (their shared cells touch across
    the slice boundary). The join graph between two ordered run lists
    is bipartite; its connected components are taken in the order of
    their earliest run, isolated runs included. A component with
    ``n_prev`` runs on the earlier slice and ``n_next`` runs on the
    later one yields one event ``[kind, from, to, overlap]``: ``from``
    and ``to`` are ascending run-index lists on the respective slices,
    ``overlap`` is the sorted union without duplicates of the edge-wise
    intersection coordinates (the ``r`` indices for a budget slice and
    the ``b`` indices for a z slice), and ``kind`` is one of ``enter``
    (0 -> many), ``exit`` (many -> 0), ``continue`` (1 -> 1), ``split``
    (1 -> many), ``merge`` (many -> 1) and ``merge_split``
    (many -> many).

    Within a slice the adjacent different-id cell pairs form a
    deduplicated, ascending edge set. Each pair of adjacent slices is
    recorded as ``[earlier, later, added_edges, removed_edges]`` with
    the two edge sets differenced.

    Returns a dict whose keys in order are ``budgets``, ``z`` and
    ``axes``: the first two are copies of the same-named entries of
    ``G``, and ``axes`` is a dict whose keys in order are ``budget``
    and ``z``. Each axis value is a dict whose keys in order are
    ``slices``, ``events`` and ``adjacency_changes``: ``slices`` lists,
    per slice, the run lists above; ``events`` lists, per pair of
    adjacent slices, the component events; and ``adjacency_changes``
    lists the per-pair edge-set differences. All containers are
    JSON-compatible, ``None`` is preserved and no value is rounded.
    """
    grid = robust_policy_grid_regions(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )
    budget_values = grid["budgets"]
    z_values = grid["z"]
    n_budgets = len(budget_values)
    n_zs = len(z_values)

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    # Id mesh rebuilt from G's regions: each region's cells share an id.
    id_mesh: list[list[int]] = [
        [0 for _r in range(n_zs)] for _b in range(n_budgets)
    ]
    state_of: dict[int, object] = {}
    for region in grid["regions"]:
        region_id = region["id"]
        state_of[region_id] = region["state"]
        for b, r in region["cells"]:
            id_mesh[b][r] = region_id

    def _runs(line: list[int]) -> list[list]:
        runs: list[list] = []
        for index, region_id in enumerate(line):
            if runs and runs[-1][2] == region_id:
                runs[-1][1] = index
            else:
                runs.append(
                    [
                        index,
                        index,
                        region_id,
                        _copy(state_of[region_id]),
                    ]
                )
        return runs

    def _edges(line: list[int]) -> list[list[int]]:
        edge_set: set[tuple[int, int]] = set()
        for index in range(len(line) - 1):
            left = line[index]
            right = line[index + 1]
            if left != right:
                edge_set.add(
                    (left, right) if left < right else (right, left)
                )
        return [[small, large] for small, large in sorted(edge_set)]

    kind_by_shape = {
        (0, 1): "enter",
        (1, 0): "exit",
        (1, 1): "continue",
        (1, 2): "split",
        (2, 1): "merge",
        (2, 2): "merge_split",
    }

    def _events(
        prev_runs: list[list],
        next_runs: list[list],
    ) -> list[list]:
        # Bipartite joins: edge (prev run index, next run index) when the
        # two runs share an id and their index intervals intersect.
        links: list[set[int]] = [set() for _ in prev_runs]
        back_links: list[set[int]] = [set() for _ in next_runs]
        overlap_of: dict[tuple[int, int], list[int]] = {}
        for pi, run in enumerate(prev_runs):
            p_first, p_last, p_id, _ = run
            for ni, other in enumerate(next_runs):
                n_first, n_last, n_id, _ = other
                if p_id != n_id:
                    continue
                lo = max(p_first, n_first)
                hi = min(p_last, n_last)
                if lo > hi:
                    continue
                links[pi].add(ni)
                back_links[ni].add(pi)
                overlap_of[(pi, ni)] = list(range(lo, hi + 1))

        # Connected components of the bipartite join graph.
        seen_prev: set[int] = set()
        seen_next: set[int] = set()
        seeds: list[tuple[str, int]] = []
        for pi in range(len(prev_runs)):
            seeds.append(("p", pi))
        for ni in range(len(next_runs)):
            seeds.append(("n", ni))

        events: list[list] = []
        for side, seed in seeds:
            if side == "p" and seed in seen_prev:
                continue
            if side == "n" and seed in seen_next:
                continue
            prev_members: set[int] = set()
            next_members: set[int] = set()
            pending: list[tuple[str, int]] = [(side, seed)]
            if side == "p":
                seen_prev.add(seed)
            else:
                seen_next.add(seed)
            while pending:
                cur_side, index = pending.pop()
                if cur_side == "p":
                    prev_members.add(index)
                    for ni in links[index]:
                        if ni not in seen_next:
                            seen_next.add(ni)
                            pending.append(("n", ni))
                else:
                    next_members.add(index)
                    for pi in back_links[index]:
                        if pi not in seen_prev:
                            seen_prev.add(pi)
                            pending.append(("p", pi))

            from_idx = sorted(prev_members)
            to_idx = sorted(next_members)

            def _side(count: int) -> int:
                return 0 if count == 0 else (1 if count == 1 else 2)

            kind = kind_by_shape[(_side(len(from_idx)), _side(len(to_idx)))]

            overlap_values: set[int] = set()
            for pi in from_idx:
                for ni in to_idx:
                    overlap_values.update(overlap_of.get((pi, ni), ()))
            overlap = sorted(overlap_values)
            events.append([kind, from_idx, to_idx, overlap])

        return events

    def _axis(
        slices: list[list[int]],
    ) -> tuple[list[list[list]], list[list[list]], list[list]]:
        run_lists = [_runs(line) for line in slices]
        edge_lists = [_edges(line) for line in slices]

        events_per_pair: list[list[list]] = []
        adjacency_changes: list[list] = []
        for index in range(len(slices) - 1):
            events_per_pair.append(
                _events(run_lists[index], run_lists[index + 1])
            )
            before = {
                (edge[0], edge[1]) for edge in edge_lists[index]
            }
            after = {
                (edge[0], edge[1]) for edge in edge_lists[index + 1]
            }
            added = [
                [small, large]
                for small, large in sorted(after - before)
            ]
            removed = [
                [small, large]
                for small, large in sorted(before - after)
            ]
            adjacency_changes.append([index, index + 1, added, removed])

        return run_lists, events_per_pair, adjacency_changes

    # Budget slices: fixed b, swept over r.
    budget_slice_lines = [list(id_mesh[b]) for b in range(n_budgets)]
    budget_slices, budget_events, budget_changes = _axis(budget_slice_lines)

    # z slices: fixed r, swept over b.
    z_slice_lines = [
        [id_mesh[b][r] for b in range(n_budgets)] for r in range(n_zs)
    ]
    z_slices, z_events, z_changes = _axis(z_slice_lines)

    return {
        "budgets": list(budget_values),
        "z": list(z_values),
        "axes": {
            "budget": {
                "slices": budget_slices,
                "events": budget_events,
                "adjacency_changes": budget_changes,
            },
            "z": {
                "slices": z_slices,
                "events": z_events,
                "adjacency_changes": z_changes,
            },
        },
    }


def robust_policy_lineage(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Region lineages traced through the budget and z slice flows.

    Every parameter shares the contract of
    :func:`robust_policy_region_flow` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_region_flow` is called exactly
    once to obtain the flow ``F``.

    Both axes flatten their slices in order (budget axis: slices in
    budget order; z axis: slices in z order) and, within each slice,
    the runs in sweep order. Every flattened run becomes one node in
    this enumeration order, so node ids ascend globally from ``0``; a
    node is ``[id, slice, run, first, last, region, state]`` with
    ``slice`` the slice index, ``run`` the run index within the slice,
    ``first`` and ``last`` that run's sweep-index ends, ``region`` its
    region id and ``state`` a recursive list copy of the run's state.

    Each event of ``F`` has its ``from`` and ``to`` run indices
    replaced by the corresponding node ids. The transitions of a slice
    pair are stored per pair as ``[kind, from, to, overlap]`` lists in
    event order: ``from`` and ``to`` are the ascending node-id lists
    and ``overlap`` is unchanged. A Cartesian from x to pair is an
    edge exactly when its two nodes share a region and their sweep
    intervals intersect.

    Edges ignore direction when connected components are taken:
    components are numbered by their smallest node id, ties resolved in
    that order. A lineage is
    ``[id, nodes, entries, exits, paths, changes]``: ``nodes`` lists
    the component's node ids ascending; ``entries`` and ``exits`` are
    the component nodes with in-degree zero and out-degree zero;
    ``paths`` are the maximal directed paths that cover every edge and
    whose interior nodes have in-degree and out-degree one each, with
    an isolated node listed on its own; the paths are ordered
    lexicographically. ``changes`` holds the non-empty adjacency
    change records of ``F`` (the ``[earlier, later, added, removed]``
    entries) for which at least one edge endpoint belongs to a region
    touching the component.

    Returns a dict whose keys in order are ``budgets``, ``z`` and
    ``axes``: the first two are the same-named entries of ``F``, and
    ``axes`` is a dict whose keys in order are ``budget`` and ``z``.
    Each axis value is a dict whose keys in order are ``nodes``,
    ``transitions`` and ``lineages``. All containers are
    JSON-compatible, ``None`` is preserved and no value is rounded.
    """
    flow = robust_policy_region_flow(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    def _lineage_axis(axis: dict) -> dict:
        slices = axis["slices"]
        events_per_pair = axis["events"]
        changes_per_pair = axis["adjacency_changes"]

        # Flatten slices then runs; node ids ascend in this order.
        nodes: list[list] = []
        node_of: dict[tuple[int, int], int] = {}
        info_of: dict[int, tuple[int, int, int, int, int]] = {}
        for slice_index, runs in enumerate(slices):
            for run_index, run in enumerate(runs):
                first, last, region_id, state = run
                node_id = len(nodes)
                node_of[(slice_index, run_index)] = node_id
                info_of[node_id] = (
                    slice_index, run_index, first, last, region_id
                )
                nodes.append(
                    [
                        node_id,
                        slice_index,
                        run_index,
                        first,
                        last,
                        region_id,
                        _copy(state),
                    ]
                )

        # Edges: Cartesian from x to with equal region and intersecting
        # sweep intervals. Transition records mirror the flow events but
        # reference node ids; the outer list stays per slice pair.
        transitions: list[list[list]] = []
        out_sets: dict[int, set[int]] = {}
        in_sets: dict[int, set[int]] = {}

        def _add_edge(source: int, target: int) -> None:
            out_sets.setdefault(source, set()).add(target)
            in_sets.setdefault(target, set()).add(source)

        for pair_index, events in enumerate(events_per_pair):
            pair_transitions: list[list] = []
            for kind, from_idx, to_idx, overlap in events:
                from_nodes = sorted(
                    node_of[(pair_index, run_index)]
                    for run_index in from_idx
                )
                to_nodes = sorted(
                    node_of[(pair_index + 1, run_index)]
                    for run_index in to_idx
                )
                pair_transitions.append(
                    [kind, from_nodes, to_nodes, list(overlap)]
                )
                for source in from_nodes:
                    _, _, s_first, s_last, s_region = info_of[source]
                    for target in to_nodes:
                        _, _, t_first, t_last, t_region = info_of[target]
                        if s_region != t_region:
                            continue
                        if max(s_first, t_first) > min(s_last, t_last):
                            continue
                        _add_edge(source, target)
            transitions.append(pair_transitions)

        # Undirected connected components, numbered by smallest node id.
        undirected: dict[int, set[int]] = {}
        for source, targets in out_sets.items():
            undirected.setdefault(source, set()).update(targets)
            for target in targets:
                undirected.setdefault(target, set()).add(source)
        for node_id in range(len(nodes)):
            undirected.setdefault(node_id, set())

        assigned: dict[int, int] = {}
        components: list[list[int]] = []
        for seed in range(len(nodes)):
            if seed in assigned:
                continue
            component_id = len(components)
            members: list[int] = []
            pending = [seed]
            assigned[seed] = component_id
            while pending:
                current = pending.pop()
                members.append(current)
                for neighbour in undirected[current]:
                    if neighbour not in assigned:
                        assigned[neighbour] = component_id
                        pending.append(neighbour)
            members.sort()
            components.append(members)
        components.sort(key=lambda members: members[0])

        # Regions touched by each component (its nodes' regions).
        regions_of_component: list[set[int]] = [
            {info_of[node_id][4] for node_id in members}
            for members in components
        ]
        component_by_region: dict[int, set[int]] = {}
        for component_id, region_ids in enumerate(
            regions_of_component
        ):
            for region_id in region_ids:
                component_by_region.setdefault(
                    region_id, set()
                ).add(component_id)

        # Non-empty adjacency change records with an endpoint region
        # touching the component.
        changes_by_component: dict[int, list[list]] = {}
        for record in changes_per_pair:
            earlier, later, added, removed = record
            if not added and not removed:
                continue
            entry = [
                earlier,
                later,
                [list(edge) for edge in added],
                [list(edge) for edge in removed],
            ]
            touched: set[int] = set()
            for edge in added:
                for region_id in edge:
                    touched.update(
                        component_by_region.get(region_id, ())
                    )
            for edge in removed:
                for region_id in edge:
                    touched.update(
                        component_by_region.get(region_id, ())
                    )
            for component_id in touched:
                changes_by_component.setdefault(
                    component_id, []
                ).append(entry)

        lineages: list[list] = []
        for component_id, members in enumerate(components):
            member_set = set(members)
            entries = sorted(
                node_id
                for node_id in members
                if not (in_sets.get(node_id, set()) & member_set)
            )
            exits = sorted(
                node_id
                for node_id in members
                if not (out_sets.get(node_id, set()) & member_set)
            )

            # Edges only join adjacent slices, so the directed graph is
            # acyclic. Maximal paths start at every outgoing edge of a
            # node that is not a degree-(1, 1) interior node, then follow
            # the unique successor while the interior condition holds.
            def _out(node_id: int) -> list[int]:
                return sorted(
                    out_sets.get(node_id, set()) & member_set
                )

            def _in(node_id: int) -> list[int]:
                return sorted(
                    in_sets.get(node_id, set()) & member_set
                )

            paths: list[list[int]] = []
            if len(members) == 1 and not _out(members[0]):
                paths.append([members[0]])
            else:
                for source in members:
                    targets = _out(source)
                    if not targets:
                        continue
                    if len(_in(source)) == 1 and len(targets) == 1:
                        continue
                    for target in targets:
                        path = [source, target]
                        current = target
                        while True:
                            nxt = _out(current)
                            if len(_in(current)) != 1 or len(nxt) != 1:
                                break
                            current = nxt[0]
                            path.append(current)
                        paths.append(path)
            paths.sort()
            lineages.append(
                [
                    component_id,
                    members,
                    entries,
                    exits,
                    paths,
                    changes_by_component.get(component_id, []),
                ]
            )

        return {
            "nodes": nodes,
            "transitions": transitions,
            "lineages": lineages,
        }

    return {
        "budgets": flow["budgets"],
        "z": flow["z"],
        "axes": {
            "budget": _lineage_axis(flow["axes"]["budget"]),
            "z": _lineage_axis(flow["axes"]["z"]),
        },
    }


def robust_policy_joint(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Per-region joint lineages across the budget and z axes.

    Every parameter shares the contract of
    :func:`robust_policy_lineage` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_lineage` is called exactly once to
    obtain the lineage result ``L``.

    A node's region is its ``region`` entry and its state is its
    ``state`` entry; the state lists rank the per-weight-set values at
    index 2 (``ranks``) and the per-quantile values at index 3
    (``levels``).

    ``regions`` lists, in ascending region order, one
    ``[region, state, B, Z]`` entry per region present on either axis:
    ``state`` is a recursive list copy of the state of the axis node on
    the region whose slice is the first on the budget axis (the node
    with the smallest id on that axis carrying the region, hence its
    lowest-budget slice), and ``B`` and ``Z`` are the ascending,
    deduplicated lineage ids (component ids) of that axis's lineages
    that contain the region.

    ``branches`` gathers, over axes in the order budget then z, slice
    pairs in order and events in order, every transition event whose
    ``kind`` is ``split``, ``merge`` or ``merge_split``, as
    ``[axis, pair, kind, from, to, overlap]``: ``axis`` is ``"budget"``
    or ``"z"``, ``pair`` is the two slice indices ``[earlier, later]``,
    ``from`` and ``to`` are the event's node-id lists and ``overlap`` is
    its overlap list.

    ``adjacency_changes`` unfolds, over axes in the order budget then z,
    lineage ids ascending and each lineage's ``changes`` entries in
    order, every change record's added edges first and removed edges
    next. An edge of region ids ``[r0, r1]`` is orientated with the
    smaller region as ``small``; it yields one
    ``[axis, earlier, later, action, small, large, rank_diff,
    level_diff]`` entry per edge, deduplicated across lineages:
    ``action`` is ``"add"`` for an added edge and ``"remove"`` for a
    removed one, and ``rank_diff`` and ``level_diff`` are the lists of
    the large region's state ranks and levels minus the small region's,
    term by term. Duplicate unfoldings (the same axis, pair, action and
    region pair produced by several lineages) are emitted once.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``regions``, ``branches`` and ``adjacency_changes``: the first two
    are copies of the same-named entries of ``L``. All containers are
    JSON-compatible, ``None`` is preserved and no value is rounded.
    """
    lineage = robust_policy_lineage(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    axis_names = ("budget", "z")

    # Per axis: node -> region, lineage id -> regions, region -> lineage
    # ids, and the smallest-id node (lowest slice) carrying each region.
    region_of_node: dict[str, dict[int, int]] = {}
    lineages_of_region: dict[str, dict[int, list[int]]] = {}
    first_node_of_region: dict[str, dict[int, int]] = {}
    regions_by_axis: dict[str, set[int]] = {}
    for axis_name in axis_names:
        axis = lineage["axes"][axis_name]
        node_regions: dict[int, int] = {}
        region_lineages: dict[int, list[int]] = {}
        first_node: dict[int, int] = {}
        region_set: set[int] = set()
        for node in axis["nodes"]:
            node_id = node[0]
            region_id = node[5]
            node_regions[node_id] = region_id
            region_set.add(region_id)
            previous = first_node.get(region_id)
            if previous is None or node_id < previous:
                first_node[region_id] = node_id
        for lin in axis["lineages"]:
            lineage_id = lin[0]
            touched = {node_regions[node_id] for node_id in lin[1]}
            for region_id in touched:
                region_lineages.setdefault(region_id, []).append(lineage_id)
        for region_id in region_lineages:
            region_lineages[region_id].sort()
        region_of_node[axis_name] = node_regions
        lineages_of_region[axis_name] = region_lineages
        first_node_of_region[axis_name] = first_node
        regions_by_axis[axis_name] = region_set

    all_regions = sorted(
        regions_by_axis["budget"] | regions_by_axis["z"]
    )

    # Representative state: budget-axis node on the region's first slice.
    budget_nodes = lineage["axes"]["budget"]["nodes"]
    state_by_node = {node[0]: node[6] for node in budget_nodes}
    regions: list[list] = []
    for region_id in all_regions:
        state = state_by_node[first_node_of_region["budget"][region_id]]
        regions.append(
            [
                region_id,
                _copy(state),
                list(lineages_of_region["budget"].get(region_id, ())),
                list(lineages_of_region["z"].get(region_id, ())),
            ]
        )

    branches: list[list] = []
    for axis_name in axis_names:
        axis = lineage["axes"][axis_name]
        for pair_index, pair_transitions in enumerate(axis["transitions"]):
            for kind, from_nodes, to_nodes, overlap in pair_transitions:
                if kind not in ("split", "merge", "merge_split"):
                    continue
                branches.append(
                    [
                        axis_name,
                        [pair_index, pair_index + 1],
                        kind,
                        list(from_nodes),
                        list(to_nodes),
                        list(overlap),
                    ]
                )

    # Representative state per region, preferring the budget axis and
    # falling back to the z axis (every region sits on at least one axis).
    representative_state: dict[int, list] = {}
    for axis_name in axis_names:
        axis = lineage["axes"][axis_name]
        for node in axis["nodes"]:
            region_id = node[5]
            if region_id not in representative_state:
                representative_state[region_id] = node[6]

    seen_edges: set[tuple] = set()
    adjacency_changes: list[list] = []
    for axis_name in axis_names:
        for lin in lineage["axes"][axis_name]["lineages"]:
            for record in lin[5]:
                earlier, later, added, removed = record
                for action, edges in (("add", added), ("remove", removed)):
                    for edge in edges:
                        r0, r1 = edge[0], edge[1]
                        small = r0 if r0 < r1 else r1
                        large = r1 if r0 < r1 else r0
                        dedup_key = (
                            axis_name, earlier, later, action, small, large
                        )
                        if dedup_key in seen_edges:
                            continue
                        seen_edges.add(dedup_key)
                        small_state = representative_state[small]
                        large_state = representative_state[large]
                        rank_diff = [
                            big - tiny
                            for tiny, big in zip(
                                small_state[2], large_state[2]
                            )
                        ]
                        level_diff = [
                            big - tiny
                            for tiny, big in zip(
                                small_state[3], large_state[3]
                            )
                        ]
                        adjacency_changes.append(
                            [
                                axis_name,
                                earlier,
                                later,
                                action,
                                small,
                                large,
                                rank_diff,
                                level_diff,
                            ]
                        )

    return {
        "budgets": _copy(lineage["budgets"]),
        "z": _copy(lineage["z"]),
        "regions": regions,
        "branches": branches,
        "adjacency_changes": adjacency_changes,
    }


def robust_policy_change_summary(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Per-region branch-event counts and cross-axis change agreement.

    Every parameter shares the contract of
    :func:`robust_policy_joint` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_joint` is called exactly once to
    obtain the joint result ``J`` and :func:`robust_policy_lineage`
    exactly once to obtain the lineage result ``L``.

    The ``nodes`` of each axis of ``L`` map that axis's node ids to
    their regions; the regions touched by a branch of ``J`` are the
    union of the mapped ``from`` and ``to`` node ids, so one branch
    counts at most once per region and axis/kind slot.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``regions``, ``consistent`` and ``conflicts``: the first two are
    recursive list copies of the same-named entries of ``J``.

    ``regions`` lists, in ascending region order (the same region set
    as ``J``), one ``[region, B, RI, LI]`` entry per region. ``B``
    holds six branch-event counts ordered by axis (budget then z) and
    within an axis by ``split``, ``merge`` and ``merge_split``. ``RI``
    and ``LI`` are ``math.fsum`` totals of, respectively, the absolute
    rank-difference and level-difference elements of the
    ``adjacency_changes`` entries of ``J`` whose region pair has the
    region as an endpoint.

    Every unordered region pair appearing among the
    ``adjacency_changes`` of both axes collects, per axis and in the
    original order of ``J``, ``[action, rank_diff_copy,
    level_diff_copy]`` triples. Pairs whose two axis lists compare
    equal go to ``consistent`` and the rest to ``conflicts``; each
    entry is ``[small_region, large_region, budget_list, z_list]`` and
    the lists are ordered by ascending region pair.

    All containers are JSON-compatible and no value is rounded.
    """
    joint = robust_policy_joint(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )
    lineage = robust_policy_lineage(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    axis_names = ("budget", "z")
    kind_slot = {"split": 0, "merge": 1, "merge_split": 2}

    # Same-axis L nodes map a branch's node ids to their regions.
    region_of_node: dict[str, dict[int, int]] = {
        axis_name: {
            node[0]: node[5]
            for node in lineage["axes"][axis_name]["nodes"]
        }
        for axis_name in axis_names
    }

    # Six branch-event counters per region: axis (budget, z) times
    # kind (split, merge, merge_split); one branch counts once per
    # region in its from/to region union.
    counts: dict[int, list[int]] = {}
    for branch in joint["branches"]:
        axis_name = branch[0]
        kind = branch[2]
        touched = {
            region_of_node[axis_name][node_id]
            for node_id in (branch[3] + branch[4])
        }
        slot = (0 if axis_name == "budget" else 3) + kind_slot[kind]
        for region_id in touched:
            bag = counts.setdefault(region_id, [0, 0, 0, 0, 0, 0])
            bag[slot] += 1

    # Absolute rank/level difference elements at region endpoints, and
    # per-axis [action, rank_diff copy, level_diff copy] unfoldings for
    # every unordered region pair, all in the original order of J.
    rank_abs: dict[int, list[float]] = {}
    level_abs: dict[int, list[float]] = {}
    pair_entries: dict[tuple[int, int], dict[str, list[list]]] = {}
    for record in joint["adjacency_changes"]:
        (
            axis_name,
            _earlier,
            _later,
            action,
            small,
            large,
            rank_diff,
            level_diff,
        ) = record
        for region_id in {small, large}:
            rank_abs.setdefault(region_id, []).extend(
                abs(item) for item in rank_diff
            )
            level_abs.setdefault(region_id, []).extend(
                abs(item) for item in level_diff
            )
        entries = pair_entries.setdefault(
            (small, large), {"budget": [], "z": []}
        )
        entries[axis_name].append(
            [action, list(rank_diff), list(level_diff)]
        )

    regions: list[list] = []
    for entry in joint["regions"]:
        region_id = entry[0]
        regions.append(
            [
                region_id,
                list(counts.get(region_id, (0, 0, 0, 0, 0, 0))),
                math.fsum(rank_abs.get(region_id, ())),
                math.fsum(level_abs.get(region_id, ())),
            ]
        )

    consistent: list[list] = []
    conflicts: list[list] = []
    for small, large in sorted(pair_entries):
        per_axis = pair_entries[(small, large)]
        budget_list = per_axis["budget"]
        z_list = per_axis["z"]
        if not budget_list or not z_list:
            continue
        item = [small, large, budget_list, z_list]
        if budget_list == z_list:
            consistent.append(item)
        else:
            conflicts.append(item)

    return {
        "budgets": _copy(joint["budgets"]),
        "z": _copy(joint["z"]),
        "regions": regions,
        "consistent": consistent,
        "conflicts": conflicts,
    }


def robust_policy_priority(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Conflict-prioritized ranking of regions and region pairs.

    Every parameter shares the contract of
    :func:`robust_policy_change_summary` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_change_summary` is called exactly
    once to obtain the summary ``S``.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``regions`` and ``pairs``: the first two are recursive list copies
    of the same-named entries of ``S``.

    For every ``[a, b, BA, ZA]`` entry of ``S["conflicts"]``, with
    ``C = len(BA) + len(ZA)``, ``RI`` the ``math.fsum`` of the absolute
    rank-difference elements and ``LI`` the ``math.fsum`` of the
    absolute level-difference elements across both axis lists, the
    ``pairs`` entry is ``[a, b, C, RI, LI, BA, ZA]`` (the axis lists
    recursively copied), sorted ascending by
    ``(-LI, -RI, -C, a, b)``.

    For every ``[r, B, RI, LI]`` entry of ``S["regions"]``, ``details``
    collects the conflicts incident to ``r`` as
    ``[other_end, BA, ZA]`` (axis lists recursively copied), sorted
    ascending by the other end. The ``regions`` entry is
    ``[r, B, fsum(B), RI, LI, len(details), details]`` and the list is
    sorted ascending by ``(-len(details), -LI, -RI, -fsum(B), r)``.

    All containers are JSON-compatible, ``None`` is preserved and no
    value is rounded.
    """
    summary = robust_policy_change_summary(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    pairs: list[list] = []
    incidents: dict[int, list[list]] = {}
    for a, b, budget_list, z_list in summary["conflicts"]:
        rank_total = math.fsum(
            abs(x) for e in budget_list + z_list for x in e[1]
        )
        level_total = math.fsum(
            abs(x) for e in budget_list + z_list for x in e[2]
        )
        count = len(budget_list) + len(z_list)
        pairs.append(
            [
                a,
                b,
                count,
                rank_total,
                level_total,
                _copy(budget_list),
                _copy(z_list),
            ]
        )
        for region_id, other in ((a, b), (b, a)):
            incidents.setdefault(region_id, []).append(
                [other, _copy(budget_list), _copy(z_list)]
            )
    pairs.sort(key=lambda item: (-item[4], -item[3], -item[2], item[0], item[1]))

    regions: list[list] = []
    for region_id, counts, rank_total, level_total in summary["regions"]:
        details = incidents.get(region_id, [])
        details = sorted(details, key=lambda item: item[0])
        regions.append(
            [
                region_id,
                _copy(counts),
                math.fsum(counts),
                rank_total,
                level_total,
                len(details),
                details,
            ]
        )
    regions.sort(
        key=lambda item: (
            -item[5],
            -item[4],
            -item[3],
            -item[2],
            item[0],
        )
    )

    return {
        "budgets": _copy(summary["budgets"]),
        "z": _copy(summary["z"]),
        "regions": regions,
        "pairs": pairs,
    }


def robust_policy_actions(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
) -> dict:
    """Conflict-priority action queue over regions and region pairs.

    Every parameter shares the contract of :func:`robust_policy_priority`
    (every ``TypeError`` and ``ValueError`` is inherited verbatim;
    exceptions propagate unchanged). :func:`robust_policy_priority` is
    called exactly once to obtain ``P``.

    Let ``BR = [P["budgets"][0], P["budgets"][-1]]`` and
    ``ZR = [P["z"][0], P["z"][-1]]``. Each
    ``[r, B, T, RI, LI, n, D]`` entry of ``P["regions"]`` becomes, at
    its 0-based rank ``i`` in ``P["regions"]`` order,
    ``[i, "region", r, BR, ZR, B, T, RI, LI, n, D]``. Each
    ``[a, b, C, RI, LI, BA, ZA]`` entry of ``P["pairs"]`` becomes, at
    its 0-based rank ``i`` in ``P["pairs"]`` order,
    ``[i, "pair", [a, b], BR, ZR, C, RI, LI,
    [["budget", BA], ["z", ZA]]]``.

    Returns a dict whose keys in order are ``budgets``, ``z`` and
    ``queue``: the first two are recursive list copies of the
    same-named entries of ``P``, and ``queue`` lists the region actions
    first and then the pair actions, each group keeping ``P``'s
    priority order. Every list is recursively copied; all containers
    are JSON-compatible, ``None`` is preserved and no value is rounded.
    """
    priority = robust_policy_priority(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    budget_bounds = [priority["budgets"][0], priority["budgets"][-1]]
    z_bounds = [priority["z"][0], priority["z"][-1]]

    queue: list[list] = []
    for i, entry in enumerate(priority["regions"]):
        region_id, counts, total, rank_total, level_total, n_details, details = entry
        queue.append(
            [
                i,
                "region",
                region_id,
                _copy(budget_bounds),
                _copy(z_bounds),
                _copy(counts),
                total,
                rank_total,
                level_total,
                n_details,
                _copy(details),
            ]
        )
    for i, entry in enumerate(priority["pairs"]):
        a, b, count, rank_total, level_total, budget_list, z_list = entry
        queue.append(
            [
                i,
                "pair",
                [a, b],
                _copy(budget_bounds),
                _copy(z_bounds),
                count,
                rank_total,
                level_total,
                [["budget", _copy(budget_list)], ["z", _copy(z_list)]],
            ]
        )

    return {
        "budgets": _copy(priority["budgets"]),
        "z": _copy(priority["z"]),
        "queue": queue,
    }


def robust_policy_action_batches(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
    capacity: int = 10,
) -> dict:
    """Conflict-priority action queue split into fixed-size batches.

    Every parameter except ``capacity`` shares the contract of
    :func:`robust_policy_actions` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_actions` is called exactly once to
    obtain ``A``. ``capacity`` must be a non-bool int >= 1; a wrong type
    raises ``TypeError`` and a value below 1 raises ``ValueError``.

    ``A["queue"]`` is sliced in its original order, ``capacity`` actions
    per batch (the last batch may be shorter). For a ``"region"`` action
    its 0-based entries 7 and 8 are the rank and level impact; for a
    ``"pair"`` action its 0-based entries 6 and 7 are. Within each batch
    those impacts are summed with ``math.fsum`` to give ``R`` and ``L``;
    ``CR`` and ``CL`` are the ``math.fsum`` of every ``R`` and ``L`` up
    to and including that batch.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``capacity`` and ``batches``: the first two values copy the
    same-named values of ``A``, and each batch is a dict whose keys in
    order are ``id``, ``range``, ``items``, ``rank``, ``level`` and
    ``cumulative``, with values the 0-based batch id, the half-open
    interval ``[start, stop]`` of queue indices, recursive copies of the
    actions, ``R``, ``L`` and ``[CR, CL]``. An empty queue gives empty
    ``batches``. Every container is JSON-compatible, ``None`` is
    preserved and no value is rounded.
    """
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise TypeError(
            f"capacity must be an int, got {type(capacity).__name__}"
        )
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")

    actions = robust_policy_actions(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    queue = actions["queue"]
    batches: list[dict] = []
    cumulative_rank: list[float] = []
    cumulative_level: list[float] = []
    for batch_id, start in enumerate(range(0, len(queue), capacity)):
        stop = min(start + capacity, len(queue))
        items = queue[start:stop]
        rank_terms: list[float] = []
        level_terms: list[float] = []
        for action in items:
            if action[1] == "region":
                rank_terms.append(action[7])
                level_terms.append(action[8])
            else:
                rank_terms.append(action[6])
                level_terms.append(action[7])
        rank_total = math.fsum(rank_terms)
        level_total = math.fsum(level_terms)
        cumulative_rank.append(rank_total)
        cumulative_level.append(level_total)
        batches.append(
            {
                "id": batch_id,
                "range": [start, stop],
                "items": [_copy(action) for action in items],
                "rank": rank_total,
                "level": level_total,
                "cumulative": [
                    math.fsum(cumulative_rank),
                    math.fsum(cumulative_level),
                ],
            }
        )

    return {
        "budgets": _copy(actions["budgets"]),
        "z": _copy(actions["z"]),
        "capacity": capacity,
        "batches": batches,
    }


def _validate_non_negative_threshold(name: str, value: object) -> object:
    """Validate a non-bool finite int or float >= 0.

    Ints (including ints too large to convert to float) are kept as
    ints — they are intrinsically finite and never converted.
    """
    if not _is_number(value):
        raise TypeError(
            f"{name} must be an int or float, got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def robust_policy_action_progress(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
    capacity: int = 10,
    rank_threshold: float = 0.0,
    level_threshold: float = 0.0,
) -> dict:
    """Cumulative rank/level progress of the batched action queue.

    Every parameter except ``rank_threshold`` and ``level_threshold``
    shares the contract of :func:`robust_policy_action_batches` (every
    ``TypeError`` and ``ValueError`` is inherited verbatim; exceptions
    propagate unchanged). :func:`robust_policy_action_batches` is called
    exactly once to obtain ``A``. Each of ``rank_threshold`` and
    ``level_threshold`` must be a non-bool int or finite float >= 0; a
    wrong type raises ``TypeError`` and a non-finite float or negative
    value raises ``ValueError``. Ints — however large — are accepted
    without conversion to float.

    Let ``Q`` be the total number of actions. For each batch of
    ``A["batches"]`` (in order), with ``e = batch["range"][1]`` and
    ``[CR, CL] = batch["cumulative"]``, ``R`` and ``P`` are the 0-based
    ranks (entry 0) of the batch's ``"region"`` and ``"pair"`` actions
    respectively, each in batch order. The progress entry is
    ``[id, e, CR, CL, CR >= rank_threshold,
    CL >= level_threshold, Q - e, R, P]``. ``fr`` and ``fl`` are the
    batch ids of the first entries whose rank and level flags are true,
    or ``None`` when no flag is.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``capacity``, ``progress`` and ``summary``: the first three values
    recursively copy the same-named values of ``A``, and ``summary`` is
    ``[Q, number of batches, last batch CR, last batch CL, fr, fl]``;
    for an empty queue the last-batch cumulative values are ``0.0``.
    Every container is JSON-compatible, ``None`` is preserved and no
    value is rounded.
    """
    rank_limit = _validate_non_negative_threshold(
        "rank_threshold", rank_threshold
    )
    level_limit = _validate_non_negative_threshold(
        "level_threshold", level_threshold
    )

    batched = robust_policy_action_batches(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs, capacity,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    batches = batched["batches"]
    total_actions = sum(batch["range"][1] - batch["range"][0] for batch in batches)

    progress: list[list] = []
    first_rank: int | None = None
    first_level: int | None = None
    for batch in batches:
        batch_id = batch["id"]
        end = batch["range"][1]
        cumulative_rank, cumulative_level = batch["cumulative"]
        region_ranks = [
            action[0] for action in batch["items"] if action[1] == "region"
        ]
        pair_ranks = [
            action[0] for action in batch["items"] if action[1] == "pair"
        ]
        rank_done = cumulative_rank >= rank_limit
        level_done = cumulative_level >= level_limit
        if rank_done and first_rank is None:
            first_rank = batch_id
        if level_done and first_level is None:
            first_level = batch_id
        progress.append(
            [
                batch_id,
                end,
                cumulative_rank,
                cumulative_level,
                rank_done,
                level_done,
                total_actions - end,
                region_ranks,
                pair_ranks,
            ]
        )

    if batches:
        last_rank = batches[-1]["cumulative"][0]
        last_level = batches[-1]["cumulative"][1]
    else:
        last_rank = 0.0
        last_level = 0.0

    return {
        "budgets": _copy(batched["budgets"]),
        "z": _copy(batched["z"]),
        "capacity": batched["capacity"],
        "progress": progress,
        "summary": [
            total_actions,
            len(batches),
            last_rank,
            last_level,
            first_rank,
            first_level,
        ],
    }


def robust_policy_action_waves(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
    capacity: int = 10,
    rank_threshold: float = 0.0,
    level_threshold: float = 0.0,
    cursor: int = 0,
) -> dict:
    """Progress entries split into done/pending waves at a batch cursor.

    Every parameter except ``cursor`` shares the contract of
    :func:`robust_policy_action_progress` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`robust_policy_action_progress` is called exactly
    once to obtain ``P``; ``Q`` and ``B`` are the first two entries of
    ``P["summary"]`` (the total action count and the batch count).
    ``cursor`` must be a non-bool int with ``0 <= cursor <= B``; a
    wrong type raises ``TypeError`` and an out-of-range value raises
    ``ValueError``.

    For each progress entry ``x`` of ``P["progress"]`` (in order), the
    wave entry is ``[status] + copy(x) + [x[0] + 1]``, where ``copy`` is
    a recursive list copy and ``status`` is ``"done"`` when
    ``x[0] < cursor`` and ``"pending"`` otherwise. ``hits`` is ordered
    by the rank then the level flag: for each flag (entries 4 and 5 of
    ``x``) it takes the first ``x`` in progress order whose flag is
    true and stores recursive copies of entries ``x[0]``, ``x[1]``,
    ``x[7]`` and ``x[8]``, or ``None`` when no flag is. ``done`` is 0
    when ``cursor == 0``, otherwise ``x[1]`` of the previous batch
    (``P["progress"][cursor - 1][1]``).

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``capacity``, ``cursor``, ``waves``, ``hits`` and ``summary``: the
    first three values recursively copy the same-named values of ``P``,
    and ``summary`` is ``[Q, B, done, Q - done]``. Every list is
    recursively copied, every container is JSON-compatible and no value
    is rounded.
    """
    if not isinstance(cursor, int) or isinstance(cursor, bool):
        raise TypeError(
            f"cursor must be an int, got {type(cursor).__name__}"
        )

    progress_result = robust_policy_action_progress(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs, capacity,
        rank_threshold, level_threshold,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    entries = progress_result["progress"]
    total_actions, batch_count = progress_result["summary"][0:2]
    if cursor < 0 or cursor > batch_count:
        raise ValueError(
            f"cursor must satisfy 0 <= cursor <= {batch_count}, got {cursor}"
        )

    waves: list[list] = []
    for entry in entries:
        status = "done" if entry[0] < cursor else "pending"
        waves.append([status, *_copy(entry), entry[0] + 1])

    hits: list[list | None] = []
    for flag_index in (4, 5):
        hit: list | None = None
        for entry in entries:
            if entry[flag_index]:
                hit = [
                    _copy(entry[0]),
                    _copy(entry[1]),
                    _copy(entry[7]),
                    _copy(entry[8]),
                ]
                break
        hits.append(hit)

    if cursor == 0:
        done = 0
    else:
        done = entries[cursor - 1][1]

    return {
        "budgets": _copy(progress_result["budgets"]),
        "z": _copy(progress_result["z"]),
        "capacity": progress_result["capacity"],
        "cursor": cursor,
        "waves": waves,
        "hits": hits,
        "summary": [total_actions, batch_count, done, total_actions - done],
    }


def robust_policy_action_confirm(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
    capacity: int = 10,
    rank_threshold: float = 0.0,
    level_threshold: float = 0.0,
    cursor: int = 0,
    *,
    target_cursor: int,
) -> dict:
    """Confirm waves between a batch cursor and a target cursor.

    The signature is :func:`robust_policy_action_waves` with a required
    keyword-only ``target_cursor`` appended. Every parameter other than
    ``target_cursor`` shares the contract of
    :func:`robust_policy_action_waves` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). ``target_cursor`` must be a non-bool int or a
    ``TypeError`` is raised.

    :func:`robust_policy_action_waves` is called exactly once with
    ``target_cursor`` in place of ``cursor`` to obtain ``W``; ``V`` is
    ``W["waves"]`` and ``Q`` and ``B`` are the first two entries of
    ``W["summary"]``. The cursors must satisfy
    ``0 <= cursor <= target_cursor <= B`` (the bounds on
    ``target_cursor`` alone are enforced by the waves call); any
    violation raises ``ValueError``.

    ``old`` is 0 when ``cursor == 0``, otherwise ``V[cursor - 1][2]``;
    ``new`` is ``W["summary"][2]``.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``capacity``, ``from_cursor``, ``to_cursor``, ``confirmed``,
    ``pending``, ``hits`` and ``summary``: the first three values
    recursively copy the same-named values of ``W``, the two cursor
    values are the original ``cursor`` and ``target_cursor``,
    ``confirmed`` and ``pending`` recursively copy
    ``V[cursor:target_cursor]`` and ``V[target_cursor:]``, ``hits``
    recursively copies ``W["hits"]``, and ``summary`` is
    ``[Q, B, old, new, new - old, Q - new, target_cursor]``. Every
    list is recursively copied, every container is JSON-compatible,
    ``None`` values are preserved and no value is rounded.
    """
    if not isinstance(cursor, int) or isinstance(cursor, bool):
        raise TypeError(
            f"cursor must be an int, got {type(cursor).__name__}"
        )
    if not isinstance(target_cursor, int) or isinstance(target_cursor, bool):
        raise TypeError(
            "target_cursor must be an int, got "
            f"{type(target_cursor).__name__}"
        )

    waves_result = robust_policy_action_waves(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs, capacity,
        rank_threshold, level_threshold, target_cursor,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    wave_entries = waves_result["waves"]
    total_actions, batch_count = waves_result["summary"][0:2]
    if cursor < 0 or cursor > target_cursor:
        raise ValueError(
            "cursors must satisfy 0 <= cursor <= target_cursor, got "
            f"cursor={cursor} and target_cursor={target_cursor}"
        )

    if cursor == 0:
        old = 0
    else:
        old = wave_entries[cursor - 1][2]
    new = waves_result["summary"][2]

    return {
        "budgets": _copy(waves_result["budgets"]),
        "z": _copy(waves_result["z"]),
        "capacity": waves_result["capacity"],
        "from_cursor": cursor,
        "to_cursor": target_cursor,
        "confirmed": _copy(wave_entries[cursor:target_cursor]),
        "pending": _copy(wave_entries[target_cursor:]),
        "hits": _copy(waves_result["hits"]),
        "summary": [
            total_actions,
            batch_count,
            old,
            new,
            new - old,
            total_actions - new,
            target_cursor,
        ],
    }


def robust_policy_action_checkpoint(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    policies: list | tuple,
    options: list | tuple,
    budgets: list[float] | tuple[float, ...],
    weight_sets: list[list[float]] | tuple[tuple[float, ...], ...],
    zs: list[float] | tuple[float, ...],
    capacity: int = 10,
    rank_threshold: float = 0.0,
    level_threshold: float = 0.0,
    cursor: int = 0,
    *,
    target_cursor: int,
    expected_done: int,
) -> dict:
    """Confirm waves and guard the previously-completed action count.

    The signature is :func:`robust_policy_action_confirm` with a
    required keyword-only ``expected_done`` appended after
    ``target_cursor``. Every parameter other than ``expected_done``
    shares the contract of :func:`robust_policy_action_confirm`
    (every ``TypeError`` and ``ValueError`` is inherited verbatim;
    exceptions propagate unchanged). ``expected_done`` must be a
    non-bool int >= 0 or a ``TypeError`` (wrong type) or
    ``ValueError`` (negative) is raised.

    :func:`robust_policy_action_confirm` is called exactly once to
    obtain ``C``; ``old``, ``new``, ``delta`` and ``remaining`` are
    entries 2 through 5 (0-based) of ``C["summary"]``, i.e.
    ``C["summary"][2:6]``. If ``expected_done`` differs from ``old`` a
    ``RuntimeError`` is raised.

    Returns a dict whose keys in order are ``budgets``, ``z``,
    ``capacity``, ``checkpoint``, ``confirmed``, ``pending`` and
    ``hits``: six values recursively copy the same-named values of
    ``C``, and ``checkpoint`` is a dict whose keys in order are
    ``from_cursor``, ``to_cursor``, ``next_cursor``, ``old_done``,
    ``done``, ``delta`` and ``remaining``, with values
    ``C["from_cursor"]``, ``C["to_cursor"]``, ``C["to_cursor"]``,
    ``old``, ``new``, ``delta`` and ``remaining`` respectively.
    ``target_cursor == cursor`` is legal: ``confirmed`` is empty and
    ``delta`` is 0. Every container is JSON-compatible, ``None``
    values are preserved and no value is rounded.
    """
    if not isinstance(expected_done, int) or isinstance(expected_done, bool):
        raise TypeError(
            "expected_done must be an int, got "
            f"{type(expected_done).__name__}"
        )
    if expected_done < 0:
        raise ValueError(
            f"expected_done must be >= 0, got {expected_done}"
        )

    confirm_result = robust_policy_action_confirm(
        H, W, thresholds, quantiles, policies, options,
        budgets, weight_sets, zs, capacity,
        rank_threshold, level_threshold, cursor,
        target_cursor=target_cursor,
    )

    def _copy(value: object) -> object:
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    old, new, delta, remaining = confirm_result["summary"][2:6]
    if expected_done != old:
        raise RuntimeError(
            "expected_done does not match the completed action count: "
            f"expected {expected_done}, got {old}"
        )

    return {
        "budgets": _copy(confirm_result["budgets"]),
        "z": _copy(confirm_result["z"]),
        "capacity": confirm_result["capacity"],
        "checkpoint": {
            "from_cursor": confirm_result["from_cursor"],
            "to_cursor": confirm_result["to_cursor"],
            "next_cursor": confirm_result["to_cursor"],
            "old_done": old,
            "done": new,
            "delta": delta,
            "remaining": remaining,
        },
        "confirmed": _copy(confirm_result["confirmed"]),
        "pending": _copy(confirm_result["pending"]),
        "hits": _copy(confirm_result["hits"]),
    }


def robust_policy_action_restore(
    checkpoints: list | tuple,
    cursor: int = 0,
    expected_done: int = 0,
) -> dict:
    """Replay action-queue checkpoints to restore the confirmed state.

    ``checkpoints`` must be a non-empty list or tuple whose items are
    results of :func:`robust_policy_action_checkpoint`; a wrong
    container type raises ``TypeError`` and an empty container raises
    ``ValueError``. Every item must be a dict with exactly the keys
    ``budgets``, ``z``, ``capacity``, ``checkpoint``, ``confirmed``,
    ``pending`` and ``hits``; an item that is not a dict raises
    ``TypeError``. ``budgets`` and ``z`` must be non-empty
    lists of non-bool finite numbers >= 0, ``capacity`` a non-bool int
    >= 1, and ``confirmed``, ``pending`` and ``hits`` lists. The
    ``checkpoint`` value must be a dict with exactly the keys
    ``from_cursor``, ``to_cursor``, ``next_cursor``, ``old_done``,
    ``done``, ``delta`` and ``remaining`` whose seven values are
    non-bool ints >= 0. Any missing or extra key or malformed value
    raises ``ValueError``; the same is raised when the ``budgets``,
    ``z`` or ``capacity`` values are not identical across all items,
    when a value is negative, when ``next_cursor`` differs from
    ``to_cursor``, when ``delta`` differs from ``done - old_done``,
    when ``remaining`` is negative or when ``to_cursor`` is below
    ``from_cursor`` (a cursor moving backwards).

    ``cursor`` and ``expected_done`` must be non-bool ints >= 0; a
    wrong type raises ``TypeError`` and a negative value raises
    ``ValueError``. The first checkpoint must have ``from_cursor``
    equal to ``cursor`` and ``old_done`` equal to ``expected_done``;
    every later checkpoint must have ``from_cursor`` equal to the
    previous checkpoint's ``to_cursor`` and ``old_done`` equal to the
    previous ``done``. Any mismatch raises ``RuntimeError``. A
    zero-delta checkpoint at the same cursor may be replayed more than
    once, and an empty queue may carry a 0 -> 0 record.

    Returns a dict whose keys in order are ``state``, ``history``,
    ``pending`` and ``hits``: ``state`` is
    ``[next_cursor, done, remaining]`` of the last checkpoint,
    ``history`` recursively copies every checkpoint dict in order, and
    ``pending`` and ``hits`` recursively copy the last item's
    same-named values. Every container is JSON-compatible, ``None`` is
    preserved and no value is rounded.
    """
    if not isinstance(cursor, int) or isinstance(cursor, bool):
        raise TypeError(
            f"cursor must be an int, got {type(cursor).__name__}"
        )
    if not isinstance(expected_done, int) or isinstance(expected_done, bool):
        raise TypeError(
            "expected_done must be an int, got "
            f"{type(expected_done).__name__}"
        )
    if cursor < 0:
        raise ValueError(f"cursor must be >= 0, got {cursor}")
    if expected_done < 0:
        raise ValueError(
            f"expected_done must be >= 0, got {expected_done}"
        )

    if not isinstance(checkpoints, (list, tuple)):
        raise TypeError(
            "checkpoints must be a list or tuple, got "
            f"{type(checkpoints).__name__}"
        )
    if len(checkpoints) == 0:
        raise ValueError("checkpoints must not be empty")

    result_keys = (
        "budgets",
        "z",
        "capacity",
        "checkpoint",
        "confirmed",
        "pending",
        "hits",
    )
    checkpoint_keys = (
        "from_cursor",
        "to_cursor",
        "next_cursor",
        "old_done",
        "done",
        "delta",
        "remaining",
    )

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    def _valid_number_list(name: str, value: object) -> bool:
        if not isinstance(value, list) or len(value) == 0:
            return False
        for element in value:
            if not _is_number(element):
                return False
            if isinstance(element, float) and not math.isfinite(element):
                return False
            if element < 0:
                return False
        return True

    history: list[dict] = []
    ref_budgets: object = None
    ref_z: object = None
    ref_capacity: object = None
    prev_to_cursor = cursor
    prev_done = expected_done

    for index, item in enumerate(checkpoints):
        prefix = f"checkpoints[{index}]"
        if not isinstance(item, dict):
            raise TypeError(
                f"{prefix} must be a dict, got {type(item).__name__}"
            )
        if set(item.keys()) != set(result_keys):
            raise ValueError(
                f"{prefix} must be a dict with keys {list(result_keys)!r}, "
                f"got keys {list(item)!r}"
            )

        budgets = item["budgets"]
        zs = item["z"]
        capacity = item["capacity"]
        checkpoint = item["checkpoint"]

        if not _valid_number_list("budgets", budgets):
            raise ValueError(
                f"{prefix}['budgets'] must be a non-empty list of finite "
                f"numbers >= 0, got {budgets!r}"
            )
        if not _valid_number_list("z", zs):
            raise ValueError(
                f"{prefix}['z'] must be a non-empty list of finite numbers "
                f">= 0, got {zs!r}"
            )
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError(
                f"{prefix}['capacity'] must be an int, got "
                f"{type(capacity).__name__}"
            )
        if capacity < 1:
            raise ValueError(
                f"{prefix}['capacity'] must be >= 1, got {capacity}"
            )
        for list_name in ("confirmed", "pending", "hits"):
            if not isinstance(item[list_name], list):
                raise ValueError(
                    f"{prefix}[{list_name!r}] must be a list, got "
                    f"{type(item[list_name]).__name__}"
                )

        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint.keys()) != set(checkpoint_keys)
        ):
            raise ValueError(
                f"{prefix}['checkpoint'] must be a dict with keys "
                f"{list(checkpoint_keys)!r}, got {type(checkpoint).__name__}"
            )

        numbers: dict[str, int] = {}
        for key in checkpoint_keys:
            value = checkpoint[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{prefix}['checkpoint'][{key!r}] must be an int, got "
                    f"{type(value).__name__}"
                )
            if value < 0:
                raise ValueError(
                    f"{prefix}['checkpoint'][{key!r}] must be >= 0, got "
                    f"{value}"
                )
            numbers[key] = value

        from_cursor = numbers["from_cursor"]
        to_cursor = numbers["to_cursor"]
        if numbers["next_cursor"] != to_cursor:
            raise ValueError(
                f"{prefix}['checkpoint']['next_cursor'] must equal "
                f"'to_cursor' ({to_cursor}), got {numbers['next_cursor']}"
            )
        if numbers["delta"] != numbers["done"] - numbers["old_done"]:
            raise ValueError(
                f"{prefix}['checkpoint']['delta'] must equal done - "
                f"old_done ({numbers['done'] - numbers['old_done']}), got "
                f"{numbers['delta']}"
            )
        if to_cursor < from_cursor:
            raise ValueError(
                f"{prefix}['checkpoint'] cursor must not move backwards: "
                f"from_cursor={from_cursor}, to_cursor={to_cursor}"
            )

        if index == 0:
            ref_budgets = budgets
            ref_z = zs
            ref_capacity = capacity
        elif budgets != ref_budgets:
            raise ValueError(
                f"{prefix}['budgets'] {budgets!r} does not match the first "
                f"checkpoint's budgets {ref_budgets!r}"
            )
        elif zs != ref_z:
            raise ValueError(
                f"{prefix}['z'] {zs!r} does not match the first "
                f"checkpoint's z {ref_z!r}"
            )
        elif capacity != ref_capacity:
            raise ValueError(
                f"{prefix}['capacity'] {capacity!r} does not match the "
                f"first checkpoint's capacity {ref_capacity!r}"
            )

        if from_cursor != prev_to_cursor:
            raise RuntimeError(
                f"{prefix}['checkpoint']['from_cursor'] is {from_cursor}, "
                f"expected {prev_to_cursor}"
            )
        if numbers["old_done"] != prev_done:
            raise RuntimeError(
                f"{prefix}['checkpoint']['old_done'] is "
                f"{numbers['old_done']}, expected {prev_done}"
            )

        history.append(_copy(checkpoint))
        prev_to_cursor = to_cursor
        prev_done = numbers["done"]

    last_item = checkpoints[-1]
    last_checkpoint = last_item["checkpoint"]
    return {
        "state": [
            last_checkpoint["next_cursor"],
            last_checkpoint["done"],
            last_checkpoint["remaining"],
        ],
        "history": history,
        "pending": _copy(last_item["pending"]),
        "hits": _copy(last_item["hits"]),
    }


def robust_policy_action_resume(
    checkpoints: list | tuple,
    cursor: int = 0,
    expected_done: int = 0,
    limit: int | None = 1,
) -> dict:
    """Restore the confirmed state and plan the next pending actions.

    The first three parameters share the contract of
    :func:`robust_policy_action_restore` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). ``limit`` must be ``None`` or a non-bool int >= 1; a
    wrong type raises ``TypeError`` and a value below 1 raises
    ``ValueError``.

    :func:`robust_policy_action_restore` is called exactly once to
    obtain ``S``; ``P`` is ``S["pending"]``. When ``limit`` is ``None``
    the plan takes all of ``P``, otherwise it takes ``P[:limit]``;
    ``left`` holds the remaining items. Both keep the original order
    and are recursively copied.

    Returns a dict whose keys in order are ``state``, ``plan``,
    ``pending``, ``hits`` and ``history``: the values are a recursive
    copy of ``S["state"]``, the plan, ``left``, and recursive copies of
    ``S["hits"]`` and ``S["history"]`` respectively. ``checkpoints`` is
    never modified, no state is advanced when an exception is raised
    and repeated calls with the same input return equal results. When
    ``P`` is empty both ``plan`` and ``pending`` are empty and
    ``state`` is unchanged. Every container is JSON-compatible,
    ``None`` is preserved and no value is rounded.
    """
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError(
                "limit must be an int or None, got "
                f"{type(limit).__name__}"
            )
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

    restored = robust_policy_action_restore(checkpoints, cursor, expected_done)

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    pending = restored["pending"]
    if limit is None:
        plan = _copy(pending)
        left: list = []
    else:
        plan = _copy(pending[:limit])
        left = _copy(pending[limit:])

    return {
        "state": _copy(restored["state"]),
        "plan": plan,
        "pending": left,
        "hits": _copy(restored["hits"]),
        "history": _copy(restored["history"]),
    }


def robust_policy_action_commit(
    checkpoints: list | tuple,
    completed: int,
    failed: bool = False,
    cursor: int = 0,
    expected_done: int = 0,
) -> dict:
    """Commit a wave run: mark completions done and roll the rest back.

    ``checkpoints``, ``cursor`` and ``expected_done`` share the
    contract of :func:`robust_policy_action_resume` (every
    ``TypeError`` and ``ValueError`` is inherited verbatim; exceptions
    propagate unchanged). ``completed`` must be a non-bool int >= 0 and
    ``failed`` a ``bool``; a wrong type raises ``TypeError`` and a
    negative ``completed`` raises ``ValueError``.

    :func:`robust_policy_action_resume` is called exactly once with
    ``limit=max(1, completed + int(failed))`` to obtain ``S``; the
    requested count ``completed + int(failed)`` must not exceed the
    length of ``S["plan"]`` or a ``ValueError`` is raised.

    ``C`` recursively copies the first ``completed`` items of
    ``S["plan"]`` with every status (entry 0) replaced by ``"done"``.
    ``P`` recursively copies ``S["plan"][completed:] + S["pending"]``,
    so the failed item and every unconfirmed item roll back to pending.
    With ``[c, d, r] = S["state"]``, when ``C`` is empty
    ``nc, nd, nr = c, d, r``; otherwise ``nc = C[-1][-1]``,
    ``nd = C[-1][2]`` and ``nr = r - nd + d``.

    Returns a dict in the shape of a
    :func:`robust_policy_action_checkpoint` result, with keys in order
    ``budgets``, ``z``, ``capacity``, ``checkpoint``, ``confirmed``,
    ``pending`` and ``hits``: ``budgets``, ``z`` and ``capacity``
    recursively copy the last input checkpoint's same-named values,
    ``confirmed`` is ``C``, ``pending`` is ``P`` and ``hits``
    recursively copies ``S["hits"]``. The ``checkpoint`` dict keeps the
    established key order ``from_cursor``, ``to_cursor``,
    ``next_cursor``, ``old_done``, ``done``, ``delta`` and
    ``remaining``, with values ``[c, nc, nc, d, nd, nd - d, nr]``. The
    result may be appended to ``checkpoints`` for later restore or
    resume calls. Every list is recursively copied, every container is
    JSON-compatible, ``None`` values are preserved and no value is
    rounded.
    """
    if not isinstance(completed, int) or isinstance(completed, bool):
        raise TypeError(
            f"completed must be an int, got {type(completed).__name__}"
        )
    if completed < 0:
        raise ValueError(f"completed must be >= 0, got {completed}")
    if not isinstance(failed, bool):
        raise TypeError(
            f"failed must be a bool, got {type(failed).__name__}"
        )

    requested = completed + int(failed)
    resumed = robust_policy_action_resume(
        checkpoints, cursor, expected_done,
        limit=max(1, requested),
    )

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    plan = resumed["plan"]
    if requested > len(plan):
        raise ValueError(
            f"requested {requested} actions but the plan has only "
            f"{len(plan)}"
        )

    confirmed: list[list] = []
    for item in plan[:completed]:
        entry = _copy(item)
        entry[0] = "done"
        confirmed.append(entry)
    pending = _copy([*plan[completed:], *resumed["pending"]])

    c, d, r = resumed["state"]
    if not confirmed:
        nc, nd, nr = c, d, r
    else:
        nc = confirmed[-1][-1]
        nd = confirmed[-1][2]
        nr = r - nd + d

    last_item = checkpoints[-1]
    return {
        "budgets": _copy(last_item["budgets"]),
        "z": _copy(last_item["z"]),
        "capacity": last_item["capacity"],
        "checkpoint": {
            "from_cursor": c,
            "to_cursor": nc,
            "next_cursor": nc,
            "old_done": d,
            "done": nd,
            "delta": nd - d,
            "remaining": nr,
        },
        "confirmed": confirmed,
        "pending": pending,
        "hits": _copy(resumed["hits"]),
    }


def robust_policy_action_audit(
    checkpoints: list | tuple,
    attempts: list | tuple,
    cursor: int = 0,
    expected_done: int = 0,
) -> dict:
    """Audit a sequence of wave commits against the recorded checkpoints.

    ``checkpoints``, ``cursor`` and ``expected_done`` share the
    contract of :func:`robust_policy_action_restore` (every
    ``TypeError`` and ``ValueError`` is inherited verbatim; exceptions
    propagate unchanged). ``attempts`` must be a list or tuple with
    exactly one fewer item than ``checkpoints``; a wrong container type
    raises ``TypeError`` and a length mismatch raises ``ValueError``.
    Every item must be a two-element list or tuple ``(completed,
    failed)``; a wrong type or arity raises ``ValueError``.
    ``completed`` must be a non-bool int >= 0 and ``failed`` a
    ``bool``; a wrong type raises ``TypeError`` and a negative
    ``completed`` raises ``ValueError``.

    :func:`robust_policy_action_restore` is called exactly once to
    obtain ``S``. For each index ``i``,
    :func:`robust_policy_action_commit` is called once as
    ``robust_policy_action_commit(C[:i + 1], c, f, cursor,
    expected_done)`` where ``(c, f)`` is ``attempts[i]`` and ``C`` is
    ``checkpoints``; any exception propagates unchanged, and a result
    not equal to ``C[i + 1]`` raises ``RuntimeError``.

    ``retry`` is true when ``i > 0``, ``attempts[i - 1]``'s ``failed``
    flag is true and ``c + int(f) > 0``. ``history[i]`` is
    ``[i, c, f, retry, copy(C[i + 1]["checkpoint"])]``.

    Returns a dict whose keys in order are ``state``, ``history``,
    ``pending``, ``hits`` and ``summary``: ``state``, ``pending`` and
    ``hits`` recursively copy ``S``'s same-named values, and
    ``summary`` is ``[len(attempts), sum of c, count of true f, count
    of true retry, S["state"][1], S["state"][2]]``. Every container is
    JSON-compatible and the inputs are never modified.
    """
    restored = robust_policy_action_restore(checkpoints, cursor, expected_done)

    if not isinstance(attempts, (list, tuple)):
        raise TypeError(
            "attempts must be a list or tuple, got "
            f"{type(attempts).__name__}"
        )
    if len(attempts) != len(checkpoints) - 1:
        raise ValueError(
            f"attempts must have {len(checkpoints) - 1} items, got "
            f"{len(attempts)}"
        )
    for index, attempt in enumerate(attempts):
        prefix = f"attempts[{index}]"
        if not isinstance(attempt, (list, tuple)):
            raise TypeError(
                f"{prefix} must be a list or tuple, got "
                f"{type(attempt).__name__}"
            )
        if len(attempt) != 2:
            raise ValueError(
                f"{prefix} must have exactly 2 items, got {len(attempt)}"
            )
        completed, failed = attempt
        if not isinstance(completed, int) or isinstance(completed, bool):
            raise TypeError(
                f"{prefix}[0] must be an int, got "
                f"{type(completed).__name__}"
            )
        if not isinstance(failed, bool):
            raise TypeError(
                f"{prefix}[1] must be a bool, got {type(failed).__name__}"
            )
        if completed < 0:
            raise ValueError(
                f"{prefix}[0] must be >= 0, got {completed}"
            )

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    history: list = []
    sum_completed = 0
    count_failed = 0
    count_retry = 0
    for index, (completed, failed) in enumerate(attempts):
        committed = robust_policy_action_commit(
            checkpoints[:index + 1],
            completed,
            failed,
            cursor,
            expected_done,
        )
        if committed != checkpoints[index + 1]:
            raise RuntimeError(
                f"attempts[{index}] commit result does not match "
                f"checkpoints[{index + 1}]"
            )
        retry = (
            index > 0
            and bool(attempts[index - 1][1])
            and completed + int(failed) > 0
        )
        history.append(
            [
                index,
                completed,
                failed,
                retry,
                _copy(checkpoints[index + 1]["checkpoint"]),
            ]
        )
        sum_completed += completed
        count_failed += int(failed)
        count_retry += int(retry)

    state = restored["state"]
    return {
        "state": _copy(state),
        "history": history,
        "pending": _copy(restored["pending"]),
        "hits": _copy(restored["hits"]),
        "summary": [
            len(attempts),
            sum_completed,
            count_failed,
            count_retry,
            state[1],
            state[2],
        ],
    }


def robust_policy_action_audit_chain(audits: list | tuple) -> dict:
    """Flatten several :func:`robust_policy_action_audit` results.

    ``audits`` must be a non-empty list or tuple whose items conform to
    the return contract of :func:`robust_policy_action_audit` and are
    internally self-consistent. A wrong container type or any item
    whose value has the wrong type raises ``TypeError``; an empty
    container, a missing or extra key, a malformed structure or any
    inconsistent statistic raises ``ValueError``.

    Every item must be a dict with keys exactly ``state``, ``history``,
    ``pending``, ``hits`` and ``summary``. ``state`` must be a list of
    exactly three non-bool ints >= 0, ``pending`` and ``hits`` lists,
    and ``history`` a list whose entries are five-item lists
    ``[index, completed, failed, retry, checkpoint]``: ``index``,
    ``completed`` non-bool ints >= 0, ``failed`` and ``retry`` bools and
    ``checkpoint`` a dict with exactly the keys ``from_cursor``,
    ``to_cursor``, ``next_cursor``, ``old_done``, ``done``, ``delta``
    and ``remaining`` whose seven values are non-bool ints >= 0. Each
    entry must satisfy ``next_cursor == to_cursor``,
    ``delta == done - old_done``, ``to_cursor >= from_cursor`` and
    ``remaining >= 0``; entries must be ordered by a strictly
    increasing ``index`` starting at 0; ``retry`` is consistent exactly
    when it is false for the first entry and otherwise equals
    ``failed of the previous entry and completed + int(failed) > 0``.
    ``summary`` must be a list of exactly six items
    ``[attempts, total_completed, failed_count, retry_count, done,
    remaining]``: the first four are non-bool ints >= 0 (the first
    equals ``len(history)``), ``done`` and ``remaining`` are ints equal
    to the last state entries; the totals must equal the sums computed
    from ``history``.

    For segments after the first, when the segment's ``history`` is
    non-empty its first checkpoint must have ``from_cursor`` and
    ``old_done`` equal to the previous segment's state's first two
    entries; when its ``history`` is empty its ``state`` must equal the
    previous segment's state. Any mismatch raises ``RuntimeError``.

    Histories are flattened in segment order and numbered globally:
    each flattened entry is ``[global index, segment index, local
    index, completed, failed, retry, checkpoint copy]``. The global
    ``retry`` flag is true exactly when the immediately preceding
    flattened entry has a true ``failed`` flag and this entry satisfies
    ``completed + int(failed) > 0``; the rule also applies across
    segment boundaries. ``retry_chains`` lists the maximal connected
    runs of adjacent retry edges, each as ``[first global index, last
    global index, global index list]``.

    Returns a dict whose keys in order are ``state``, ``history``,
    ``retry_chains``, ``pending``, ``hits`` and ``summary``: ``state``,
    ``pending`` and ``hits`` recursively copy the last segment's
    same-named values, and ``summary`` is ``[segment count, attempt
    count, sum of completed, failed count, retry count, last done, last
    remaining]``. Every container is JSON-compatible, ``None`` is
    preserved, no value is rounded and the input is never modified.
    """
    if not isinstance(audits, (list, tuple)):
        raise TypeError(
            "audits must be a list or tuple, got "
            f"{type(audits).__name__}"
        )
    if len(audits) == 0:
        raise ValueError("audits must not be empty")

    result_keys = (
        "state",
        "history",
        "pending",
        "hits",
        "summary",
    )
    checkpoint_keys = (
        "from_cursor",
        "to_cursor",
        "next_cursor",
        "old_done",
        "done",
        "delta",
        "remaining",
    )

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    def _non_negative_int(prefix: str, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{prefix} must be a non-bool int, got "
                f"{type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"{prefix} must be >= 0, got {value}")
        return value

    flat_history: list = []
    total_attempts = 0
    total_completed = 0
    total_failed = 0
    total_retry = 0
    last_state: list | None = None
    last_pending: list | None = None
    last_hits: list | None = None

    for segment_index, audit in enumerate(audits):
        seg_prefix = f"audits[{segment_index}]"
        if not isinstance(audit, dict):
            raise TypeError(
                f"{seg_prefix} must be a dict, got {type(audit).__name__}"
            )
        if set(audit.keys()) != set(result_keys):
            raise ValueError(
                f"{seg_prefix} must be a dict with keys "
                f"{list(result_keys)!r}, got keys {list(audit)!r}"
            )

        state = audit["state"]
        history = audit["history"]
        pending = audit["pending"]
        hits = audit["hits"]
        summary = audit["summary"]

        if not isinstance(state, list):
            raise TypeError(
                f"{seg_prefix}['state'] must be a list, got "
                f"{type(state).__name__}"
            )
        if len(state) != 3:
            raise ValueError(
                f"{seg_prefix}['state'] must have exactly 3 items, got "
                f"{len(state)}"
            )
        for s_index, s_value in enumerate(state):
            _non_negative_int(f"{seg_prefix}['state'][{s_index}]", s_value)

        for list_name, list_value in (("pending", pending), ("hits", hits)):
            if not isinstance(list_value, list):
                raise TypeError(
                    f"{seg_prefix}[{list_name!r}] must be a list, got "
                    f"{type(list_value).__name__}"
                )

        if not isinstance(history, list):
            raise TypeError(
                f"{seg_prefix}['history'] must be a list, got "
                f"{type(history).__name__}"
            )

        seg_completed = 0
        seg_failed = 0
        prev_failed = False
        prev_index = -1
        prev_checkpoint: dict[str, int] | None = None
        for local_index, entry in enumerate(history):
            entry_prefix = f"{seg_prefix}['history'][{local_index}]"
            if not isinstance(entry, list):
                raise TypeError(
                    f"{entry_prefix} must be a list, got "
                    f"{type(entry).__name__}"
                )
            if len(entry) != 5:
                raise ValueError(
                    f"{entry_prefix} must have exactly 5 items, got "
                    f"{len(entry)}"
                )
            local_no, completed, failed, retry, checkpoint = entry
            _non_negative_int(f"{entry_prefix}[0]", local_no)
            _non_negative_int(f"{entry_prefix}[1]", completed)
            if not isinstance(failed, bool):
                raise TypeError(
                    f"{entry_prefix}[2] must be a bool, got "
                    f"{type(failed).__name__}"
                )
            if not isinstance(retry, bool):
                raise TypeError(
                    f"{entry_prefix}[3] must be a bool, got "
                    f"{type(retry).__name__}"
                )
            if not isinstance(checkpoint, dict):
                raise TypeError(
                    f"{entry_prefix}[4] must be a dict, got "
                    f"{type(checkpoint).__name__}"
                )
            if set(checkpoint.keys()) != set(checkpoint_keys):
                raise ValueError(
                    f"{entry_prefix}[4] must be a dict with keys "
                    f"{list(checkpoint_keys)!r}, got keys "
                    f"{list(checkpoint)!r}"
                )
            numbers: dict[str, int] = {}
            for key in checkpoint_keys:
                numbers[key] = _non_negative_int(
                    f"{entry_prefix}[4][{key!r}]", checkpoint[key]
                )
            if numbers["next_cursor"] != numbers["to_cursor"]:
                raise ValueError(
                    f"{entry_prefix}[4]['next_cursor'] must equal "
                    f"'to_cursor' ({numbers['to_cursor']}), got "
                    f"{numbers['next_cursor']}"
                )
            if numbers["delta"] != numbers["done"] - numbers["old_done"]:
                raise ValueError(
                    f"{entry_prefix}[4]['delta'] must equal done - "
                    f"old_done ({numbers['done'] - numbers['old_done']}), "
                    f"got {numbers['delta']}"
                )
            if numbers["to_cursor"] < numbers["from_cursor"]:
                raise ValueError(
                    f"{entry_prefix}[4] cursor must not move backwards: "
                    f"from_cursor={numbers['from_cursor']}, "
                    f"to_cursor={numbers['to_cursor']}"
                )
            if local_no != prev_index + 1:
                raise ValueError(
                    f"{entry_prefix}[0] must be a strictly increasing "
                    f"0-based index, expected {prev_index + 1}, got "
                    f"{local_no}"
                )
            prev_index = local_no
            expected_retry = (
                local_index > 0
                and prev_failed
                and completed + int(failed) > 0
            )
            if retry != expected_retry:
                raise ValueError(
                    f"{entry_prefix}[3] retry flag must be {expected_retry}"
                )
            if prev_checkpoint is not None:
                if numbers["from_cursor"] != prev_checkpoint["to_cursor"]:
                    raise ValueError(
                        f"{entry_prefix}[4]['from_cursor'] must equal the "
                        f"previous checkpoint's to_cursor "
                        f"{prev_checkpoint['to_cursor']}, got "
                        f"{numbers['from_cursor']}"
                    )
                if numbers["old_done"] != prev_checkpoint["done"]:
                    raise ValueError(
                        f"{entry_prefix}[4]['old_done'] must equal the "
                        f"previous checkpoint's done "
                        f"{prev_checkpoint['done']}, got "
                        f"{numbers['old_done']}"
                    )
                expected_remaining = (
                    prev_checkpoint["remaining"] - numbers["delta"]
                )
                if numbers["remaining"] != expected_remaining:
                    raise ValueError(
                        f"{entry_prefix}[4]['remaining'] must equal the "
                        f"previous checkpoint's remaining "
                        f"{prev_checkpoint['remaining']} minus this "
                        f"checkpoint's delta {numbers['delta']} "
                        f"({expected_remaining}), got {numbers['remaining']}"
                    )
            prev_checkpoint = numbers
            prev_failed = failed
            seg_completed += completed
            seg_failed += int(failed)

        if len(history) > 0:
            last_checkpoint = history[-1][4]
            expected_state = [
                last_checkpoint["next_cursor"],
                last_checkpoint["done"],
                last_checkpoint["remaining"],
            ]
            if state != expected_state:
                raise ValueError(
                    f"{seg_prefix}['state'] must equal the last history "
                    f"checkpoint state {expected_state!r}, got {state!r}"
                )

        if not isinstance(summary, list):
            raise TypeError(
                f"{seg_prefix}['summary'] must be a list, got "
                f"{type(summary).__name__}"
            )
        if len(summary) != 6:
            raise ValueError(
                f"{seg_prefix}['summary'] must have exactly 6 items, got "
                f"{len(summary)}"
            )
        attempts_count = _non_negative_int(
            f"{seg_prefix}['summary'][0]", summary[0]
        )
        summary_completed = _non_negative_int(
            f"{seg_prefix}['summary'][1]", summary[1]
        )
        summary_failed = _non_negative_int(
            f"{seg_prefix}['summary'][2]", summary[2]
        )
        summary_retry = _non_negative_int(
            f"{seg_prefix}['summary'][3]", summary[3]
        )
        if not isinstance(summary[4], int) or isinstance(summary[4], bool):
            raise TypeError(
                f"{seg_prefix}['summary'][4] must be a non-bool int, got "
                f"{type(summary[4]).__name__}"
            )
        if not isinstance(summary[5], int) or isinstance(summary[5], bool):
            raise TypeError(
                f"{seg_prefix}['summary'][5] must be a non-bool int, got "
                f"{type(summary[5]).__name__}"
            )

        if attempts_count != len(history):
            raise ValueError(
                f"{seg_prefix}['summary'][0] must equal the history length "
                f"{len(history)}, got {attempts_count}"
            )
        if summary_completed != seg_completed:
            raise ValueError(
                f"{seg_prefix}['summary'][1] must equal the sum of "
                f"completed {seg_completed}, got {summary_completed}"
            )
        if summary_failed != seg_failed:
            raise ValueError(
                f"{seg_prefix}['summary'][2] must equal the failed count "
                f"{seg_failed}, got {summary_failed}"
            )
        seg_retry = sum(1 for entry in history if entry[3])
        if summary_retry != seg_retry:
            raise ValueError(
                f"{seg_prefix}['summary'][3] must equal the retry count "
                f"{seg_retry}, got {summary_retry}"
            )
        if summary[4] != state[1]:
            raise ValueError(
                f"{seg_prefix}['summary'][4] must equal state[1] "
                f"{state[1]}, got {summary[4]}"
            )
        if summary[5] != state[2]:
            raise ValueError(
                f"{seg_prefix}['summary'][5] must equal state[2] "
                f"{state[2]}, got {summary[5]}"
            )

        if last_state is not None:
            if len(history) > 0:
                first_checkpoint = history[0][4]
                if (
                    first_checkpoint["from_cursor"] != last_state[0]
                    or first_checkpoint["old_done"] != last_state[1]
                ):
                    raise RuntimeError(
                        f"{seg_prefix} first checkpoint must continue the "
                        f"previous segment state {last_state!r}, got "
                        f"from_cursor={first_checkpoint['from_cursor']}, "
                        f"old_done={first_checkpoint['old_done']}"
                    )
            elif state != last_state:
                raise RuntimeError(
                    f"{seg_prefix} has an empty history but its state "
                    f"{state!r} does not equal the previous segment state "
                    f"{last_state!r}"
                )

        global_base = len(flat_history)
        for local_index, entry in enumerate(history):
            local_no, completed, failed, _segment_retry, checkpoint = entry
            global_no = global_base + local_index
            if global_no == 0:
                retry = False
            else:
                retry = (
                    flat_history[global_no - 1][4]
                    and completed + int(failed) > 0
                )
            flat_history.append(
                [
                    global_no,
                    segment_index,
                    local_no,
                    completed,
                    failed,
                    retry,
                    _copy(checkpoint),
                ]
            )
            total_completed += completed
            total_failed += int(failed)
            total_retry += int(retry)
        total_attempts += len(history)

        last_state = list(state)
        last_pending = pending
        last_hits = hits

    retry_chains: list = []
    chain_start = 0
    while chain_start < len(flat_history):
        if not flat_history[chain_start][5]:
            chain_start += 1
            continue
        chain_end = chain_start
        while (
            chain_end + 1 < len(flat_history)
            and flat_history[chain_end + 1][5]
        ):
            chain_end += 1
        retry_chains.append(
            [chain_start, chain_end, list(range(chain_start, chain_end + 1))]
        )
        chain_start = chain_end + 1

    return {
        "state": _copy(last_state),
        "history": flat_history,
        "retry_chains": retry_chains,
        "pending": _copy(last_pending),
        "hits": _copy(last_hits),
        "summary": [
            len(audits),
            total_attempts,
            total_completed,
            total_failed,
            total_retry,
            last_state[1],
            last_state[2],
        ],
    }


def audit_delta(previous: dict, current: dict) -> dict:
    """Difference between two :func:`robust_policy_action_audit_chain` results.

    The two arguments (denoted ``P`` and ``C`` below) must each conform
    to the return contract of :func:`robust_policy_action_audit_chain`:
    a dict with keys exactly ``state``, ``history``, ``retry_chains``,
    ``pending``, ``hits`` and ``summary``. A wrong argument type (or any
    wrong inner type) raises ``TypeError``; any malformed structure,
    inconsistent statistic or broken recursion raises ``ValueError``.

    In addition, ``C["history"]`` must have ``P["history"]`` as a
    prefix; otherwise ``ValueError``. When the suffix is non-empty, the
    first suffix checkpoint must satisfy
    ``[from_cursor, old_done, remaining] == [P["state"][0],
    P["state"][1], P["state"][2] - delta]``; when the suffix is empty
    the two states must be equal. Any mismatch raises ``ValueError``.

    ``D`` is the seven-item list ``C["summary"][i] - P["summary"][i]``
    in summary order (segment count, attempt count, completed sum,
    failed count, retry count, done, remaining). The first six items
    must be ``>= 0`` and the last one ``<= 0`` (a sign violation raises
    ``RuntimeError``), and the increments must be consistent with the
    suffix: item 1 must equal the suffix length, items 2, 3 and 4 must
    equal the suffix sums of ``completed``, true ``failed`` and true
    ``retry``, item 0 must be at least the number of distinct segment
    indices appearing in the suffix, and items 5 and 6 must equal
    ``C["state"][1] - P["state"][1]`` and
    ``C["state"][2] - P["state"][2]``. An inconsistent increment raises
    ``RuntimeError``.

    Retry chains are compared value-by-value (a chain equals another
    when their three items are equal); common chains cancel pairwise
    in ``C`` order. Returns a dict whose keys in order are ``states``,
    ``delta``, ``history`` and ``retries``: ``states`` is
    ``[P["state"] copy, C["state"] copy]``, ``delta`` is ``D``,
    ``history`` is the suffix of ``C["history"]`` and ``retries`` is
    ``[C-only chains, P-only chains]``, each side keeping its original
    order. Every container is recursively copied and JSON-compatible;
    the inputs are never modified.
    """
    result_keys = (
        "state",
        "history",
        "retry_chains",
        "pending",
        "hits",
        "summary",
    )
    checkpoint_keys = (
        "from_cursor",
        "to_cursor",
        "next_cursor",
        "old_done",
        "done",
        "delta",
        "remaining",
    )

    def _non_negative_int(prefix: str, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{prefix} must be a non-bool int, got "
                f"{type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"{prefix} must be >= 0, got {value}")
        return value

    def _validate_chain(name: str, value: object) -> dict:
        if not isinstance(value, dict):
            raise TypeError(
                f"{name} must be a dict, got {type(value).__name__}"
            )
        if set(value.keys()) != set(result_keys):
            raise ValueError(
                f"{name} must be a dict with keys {list(result_keys)!r}, "
                f"got keys {list(value)!r}"
            )

        state = value["state"]
        history = value["history"]
        retry_chains = value["retry_chains"]
        summary = value["summary"]

        if not isinstance(state, list):
            raise TypeError(
                f"{name}['state'] must be a list, got "
                f"{type(state).__name__}"
            )
        if len(state) != 3:
            raise ValueError(
                f"{name}['state'] must have exactly 3 items, got "
                f"{len(state)}"
            )
        for s_index, s_value in enumerate(state):
            _non_negative_int(f"{name}['state'][{s_index}]", s_value)

        for list_name in ("pending", "hits"):
            if not isinstance(value[list_name], list):
                raise TypeError(
                    f"{name}[{list_name!r}] must be a list, got "
                    f"{type(value[list_name]).__name__}"
                )

        if not isinstance(history, list):
            raise TypeError(
                f"{name}['history'] must be a list, got "
                f"{type(history).__name__}"
            )

        total_completed = 0
        total_failed = 0
        total_retry = 0
        current_segment = -1
        local_counter = 0
        prev_checkpoint: dict[str, int] | None = None
        failed_flags: list[bool] = []
        retry_flags: list[bool] = []
        for global_index, entry in enumerate(history):
            entry_prefix = f"{name}['history'][{global_index}]"
            if not isinstance(entry, list):
                raise TypeError(
                    f"{entry_prefix} must be a list, got "
                    f"{type(entry).__name__}"
                )
            if len(entry) != 7:
                raise ValueError(
                    f"{entry_prefix} must have exactly 7 items, got "
                    f"{len(entry)}"
                )
            global_no, segment_no, local_no, completed, failed, retry, checkpoint = entry
            _non_negative_int(f"{entry_prefix}[0]", global_no)
            _non_negative_int(f"{entry_prefix}[1]", segment_no)
            _non_negative_int(f"{entry_prefix}[2]", local_no)
            _non_negative_int(f"{entry_prefix}[3]", completed)
            if not isinstance(failed, bool):
                raise TypeError(
                    f"{entry_prefix}[4] must be a bool, got "
                    f"{type(failed).__name__}"
                )
            if not isinstance(retry, bool):
                raise TypeError(
                    f"{entry_prefix}[5] must be a bool, got "
                    f"{type(retry).__name__}"
                )
            if global_no != global_index:
                raise ValueError(
                    f"{entry_prefix}[0] must be the 0-based global index "
                    f"{global_index}, got {global_no}"
                )
            if segment_no < current_segment:
                raise ValueError(
                    f"{entry_prefix}[1] segment indices must be "
                    f"non-decreasing, got {segment_no} after "
                    f"{current_segment}"
                )
            same_segment = segment_no == current_segment
            if not same_segment:
                current_segment = segment_no
                local_counter = 0
            if local_no != local_counter:
                raise ValueError(
                    f"{entry_prefix}[2] must be the 0-based local index "
                    f"{local_counter}, got {local_no}"
                )
            local_counter += 1
            expected_retry = (
                global_index > 0
                and failed_flags[-1]
                and completed + int(failed) > 0
            )
            if retry != expected_retry:
                raise ValueError(
                    f"{entry_prefix}[5] retry flag must be {expected_retry}"
                )
            failed_flags.append(failed)
            retry_flags.append(retry)

            if not isinstance(checkpoint, dict):
                raise TypeError(
                    f"{entry_prefix}[6] must be a dict, got "
                    f"{type(checkpoint).__name__}"
                )
            if set(checkpoint.keys()) != set(checkpoint_keys):
                raise ValueError(
                    f"{entry_prefix}[6] must be a dict with keys "
                    f"{list(checkpoint_keys)!r}, got keys "
                    f"{list(checkpoint)!r}"
                )
            numbers: dict[str, int] = {}
            for key in checkpoint_keys:
                numbers[key] = _non_negative_int(
                    f"{entry_prefix}[6][{key!r}]", checkpoint[key]
                )
            if numbers["next_cursor"] != numbers["to_cursor"]:
                raise ValueError(
                    f"{entry_prefix}[6]['next_cursor'] must equal "
                    f"'to_cursor' ({numbers['to_cursor']}), got "
                    f"{numbers['next_cursor']}"
                )
            if numbers["delta"] != numbers["done"] - numbers["old_done"]:
                raise ValueError(
                    f"{entry_prefix}[6]['delta'] must equal done - "
                    f"old_done ({numbers['done'] - numbers['old_done']}), "
                    f"got {numbers['delta']}"
                )
            if numbers["to_cursor"] < numbers["from_cursor"]:
                raise ValueError(
                    f"{entry_prefix}[6] cursor must not move backwards: "
                    f"from_cursor={numbers['from_cursor']}, "
                    f"to_cursor={numbers['to_cursor']}"
                )
            if prev_checkpoint is not None:
                if numbers["from_cursor"] != prev_checkpoint["to_cursor"]:
                    raise ValueError(
                        f"{entry_prefix}[6]['from_cursor'] must equal the "
                        f"previous checkpoint's to_cursor "
                        f"{prev_checkpoint['to_cursor']}, got "
                        f"{numbers['from_cursor']}"
                    )
                if numbers["old_done"] != prev_checkpoint["done"]:
                    raise ValueError(
                        f"{entry_prefix}[6]['old_done'] must equal the "
                        f"previous checkpoint's done "
                        f"{prev_checkpoint['done']}, got "
                        f"{numbers['old_done']}"
                    )
                if same_segment:
                    expected_remaining = (
                        prev_checkpoint["remaining"] - numbers["delta"]
                    )
                    if numbers["remaining"] != expected_remaining:
                        raise ValueError(
                            f"{entry_prefix}[6]['remaining'] must equal the "
                            f"previous checkpoint's remaining "
                            f"{prev_checkpoint['remaining']} minus this "
                            f"checkpoint's delta {numbers['delta']} "
                            f"({expected_remaining}), got "
                            f"{numbers['remaining']}"
                        )
            prev_checkpoint = numbers
            total_completed += completed
            total_failed += int(failed)
            total_retry += int(retry)

        if history:
            last_checkpoint = history[-1][6]
            expected_state = [
                last_checkpoint["next_cursor"],
                last_checkpoint["done"],
                last_checkpoint["remaining"],
            ]
            if state != expected_state:
                raise ValueError(
                    f"{name}['state'] must equal the last history "
                    f"checkpoint state {expected_state!r}, got {state!r}"
                )

        if not isinstance(summary, list):
            raise TypeError(
                f"{name}['summary'] must be a list, got "
                f"{type(summary).__name__}"
            )
        if len(summary) != 7:
            raise ValueError(
                f"{name}['summary'] must have exactly 7 items, got "
                f"{len(summary)}"
            )
        for s_index, s_value in enumerate(summary):
            _non_negative_int(f"{name}['summary'][{s_index}]", s_value)

        minimum_segments = current_segment + 1 if history else 1
        if summary[0] < minimum_segments:
            raise ValueError(
                f"{name}['summary'][0] must be a segment count >= "
                f"{minimum_segments}, got {summary[0]}"
            )
        expected_summary = [
            summary[0],
            len(history),
            total_completed,
            total_failed,
            total_retry,
            state[1],
            state[2],
        ]
        if summary[1:] != expected_summary[1:]:
            raise ValueError(
                f"{name}['summary'] must equal {expected_summary!r}, got "
                f"{list(summary)!r}"
            )

        if not isinstance(retry_chains, list):
            raise TypeError(
                f"{name}['retry_chains'] must be a list, got "
                f"{type(retry_chains).__name__}"
            )
        expected_chains: list = []
        chain_start = 0
        while chain_start < len(retry_flags):
            if not retry_flags[chain_start]:
                chain_start += 1
                continue
            chain_end = chain_start
            while (
                chain_end + 1 < len(retry_flags)
                and retry_flags[chain_end + 1]
            ):
                chain_end += 1
            expected_chains.append(
                [
                    chain_start,
                    chain_end,
                    list(range(chain_start, chain_end + 1)),
                ]
            )
            chain_start = chain_end + 1
        if retry_chains != expected_chains:
            raise ValueError(
                f"{name}['retry_chains'] must equal {expected_chains!r}, "
                f"got {retry_chains!r}"
            )

        return value

    p_chain = _validate_chain("previous", previous)
    c_chain = _validate_chain("current", current)

    p_history = p_chain["history"]
    c_history = c_chain["history"]
    p_state = p_chain["state"]
    c_state = c_chain["state"]

    prefix_len = len(p_history)
    if len(c_history) < prefix_len or c_history[:prefix_len] != p_history:
        raise ValueError(
            "current['history'] must have previous['history'] as a "
            f"prefix of length {prefix_len}"
        )
    suffix = c_history[prefix_len:]

    if suffix:
        first_checkpoint = suffix[0][6]
        expected_remaining = p_state[2] - first_checkpoint["delta"]
        if (
            first_checkpoint["from_cursor"] != p_state[0]
            or first_checkpoint["old_done"] != p_state[1]
            or first_checkpoint["remaining"] != expected_remaining
        ):
            raise ValueError(
                "the first suffix checkpoint must continue the previous "
                f"state {p_state!r}: expected "
                f"[from_cursor, old_done, remaining] = "
                f"[{p_state[0]}, {p_state[1]}, {expected_remaining}], "
                f"got [{first_checkpoint['from_cursor']}, "
                f"{first_checkpoint['old_done']}, "
                f"{first_checkpoint['remaining']}]"
            )
    elif c_state != p_state:
        raise ValueError(
            "with an empty history suffix, current['state'] must equal "
            f"previous['state'] {p_state!r}, got {c_state!r}"
        )

    p_summary = p_chain["summary"]
    c_summary = c_chain["summary"]
    delta = [c_summary[i] - p_summary[i] for i in range(7)]
    for index in range(6):
        if delta[index] < 0:
            raise RuntimeError(
                f"delta[{index}] must be >= 0, got {delta[index]}"
            )
    if delta[6] > 0:
        raise RuntimeError(f"delta[6] must be <= 0, got {delta[6]}")

    suffix_attempts = len(suffix)
    suffix_completed = sum(entry[3] for entry in suffix)
    suffix_failed = sum(int(entry[4]) for entry in suffix)
    suffix_retry = sum(int(entry[5]) for entry in suffix)
    suffix_segments = len({entry[1] for entry in suffix})

    def _increment_error(index: int, expected: int) -> RuntimeError:
        return RuntimeError(
            f"delta[{index}] must equal {expected}, got {delta[index]}"
        )

    if delta[0] < suffix_segments:
        raise _increment_error(0, suffix_segments)
    if delta[1] != suffix_attempts:
        raise _increment_error(1, suffix_attempts)
    if delta[2] != suffix_completed:
        raise _increment_error(2, suffix_completed)
    if delta[3] != suffix_failed:
        raise _increment_error(3, suffix_failed)
    if delta[4] != suffix_retry:
        raise _increment_error(4, suffix_retry)
    if delta[5] != c_state[1] - p_state[1]:
        raise _increment_error(5, c_state[1] - p_state[1])
    if delta[6] != c_state[2] - p_state[2]:
        raise _increment_error(6, c_state[2] - p_state[2])

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    unmatched_p = list(p_chain["retry_chains"])
    c_only: list = []
    for chain in c_chain["retry_chains"]:
        for index, candidate in enumerate(unmatched_p):
            if candidate == chain:
                del unmatched_p[index]
                break
        else:
            c_only.append(_copy(chain))
    p_only = [_copy(chain) for chain in unmatched_p]

    return {
        "states": [_copy(p_state), _copy(c_state)],
        "delta": delta,
        "history": _copy(suffix),
        "retries": [c_only, p_only],
    }


def audit_reconcile(snapshots: list | tuple) -> dict:
    """Reconcile a time-ordered series of audit-chain snapshots.

    ``snapshots`` must be a non-empty list or tuple whose items conform
    to the return contract of
    :func:`robust_policy_action_audit_chain`. A wrong container type or
    any item of the wrong type raises ``TypeError``; an empty sequence
    raises ``ValueError``.

    The first snapshot is validated with
    ``audit_delta(S[0], S[0])``; for each ``i`` from 1 to ``L - 1``,
    ``audit_delta(S[i - 1], S[i])`` is called exactly once and any
    exception propagates unchanged. The per-step results form
    ``steps`` (empty when ``L == 1``).

    ``delta`` is the seven-item summary difference between the last and
    first snapshots, item by item (later minus earlier). It must equal
    the item-wise ``sum`` of every step's ``delta``; otherwise
    ``RuntimeError``.

    ``retries`` starts as ``[[], []]``. Processing the steps in order,
    additions are handled first and then removals: for the current
    item, the first equal item in the opposite list cancels it;
    otherwise a recursive copy of the item is appended to this side's
    list.

    Returns a dict whose keys in order are ``states``, ``delta``,
    ``steps`` and ``retries``: ``states`` is ``[first state copy, last
    state copy]`` and ``steps`` holds copies of the per-step delta
    results. Every container is recursively copied and JSON-compatible;
    the input is never modified.
    """
    if not isinstance(snapshots, (list, tuple)):
        raise TypeError(
            "snapshots must be a list or tuple, got "
            f"{type(snapshots).__name__}"
        )
    if len(snapshots) == 0:
        raise ValueError("snapshots must not be empty")

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    steps: list = []
    audit_delta(snapshots[0], snapshots[0])
    for index in range(1, len(snapshots)):
        steps.append(audit_delta(snapshots[index - 1], snapshots[index]))

    first_summary = snapshots[0]["summary"]
    last_summary = snapshots[-1]["summary"]
    delta = [last_summary[i] - first_summary[i] for i in range(7)]
    summed = [sum(step["delta"][i] for step in steps) for i in range(7)]
    if delta != summed:
        raise RuntimeError(
            "overall summary delta must equal the sum of the step deltas "
            f"{summed!r}, got {delta!r}"
        )

    added: list = []
    removed: list = []
    for step in steps:
        for chain in step["retries"][0]:
            for index, candidate in enumerate(removed):
                if candidate == chain:
                    del removed[index]
                    break
            else:
                added.append(_copy(chain))
        for chain in step["retries"][1]:
            for index, candidate in enumerate(added):
                if candidate == chain:
                    del added[index]
                    break
            else:
                removed.append(_copy(chain))

    return {
        "states": [
            _copy(snapshots[0]["state"]),
            _copy(snapshots[-1]["state"]),
        ],
        "delta": delta,
        "steps": _copy(steps),
        "retries": [added, removed],
    }


def audit_window_report(snapshots: list | tuple, size: int = 2) -> dict:
    """Sliding-window reconciliation report over audit-chain snapshots.

    ``snapshots`` follows the exact contract of
    :func:`audit_reconcile`: a non-empty list or tuple of items
    conforming to the :func:`robust_policy_action_audit_chain` return
    contract; a wrong container or item type raises ``TypeError`` and
    an empty sequence raises ``ValueError``.

    ``size`` must be a non-bool ``int`` with ``1 <= size <=
    len(snapshots)``; a wrong type raises ``TypeError`` and an
    out-of-range value raises ``ValueError``.

    For each ``e`` from ``size - 1`` through ``len(snapshots) - 1``,
    :func:`audit_reconcile` is called exactly once on the window
    ``snapshots[e - size + 1:e + 1]``; its result is denoted ``R`` and
    every exception propagates unchanged. The corresponding report
    item is ``[start, e, states, delta, retries]`` where the last three
    values are recursive copies of the same-named values of ``R``;
    reports are ordered by ``e``.

    Returns a dict whose keys in order are ``size``, ``reports`` and
    ``totals``. ``totals`` is accumulated by traversing the reports:
    ``[sum(delta[5]), sum(delta[6]), sum(len(retries[0])),
    sum(len(retries[1]))]``, corresponding to completed, remaining,
    added and removed. Every container is recursively copied and
    JSON-compatible (``None`` values kept, numbers never rounded); the
    inputs are never modified.
    """
    if not isinstance(snapshots, (list, tuple)):
        raise TypeError(
            "snapshots must be a list or tuple, got "
            f"{type(snapshots).__name__}"
        )
    if len(snapshots) == 0:
        raise ValueError("snapshots must not be empty")
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError(
            f"size must be a non-bool int, got {type(size).__name__}"
        )
    if size < 1 or size > len(snapshots):
        raise ValueError(
            f"size must satisfy 1 <= size <= len(snapshots) "
            f"({len(snapshots)}), got {size}"
        )

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    reports: list = []
    totals = [0, 0, 0, 0]
    for end in range(size - 1, len(snapshots)):
        start = end - size + 1
        reconciled = audit_reconcile(snapshots[start:end + 1])
        states = _copy(reconciled["states"])
        delta = _copy(reconciled["delta"])
        retries = _copy(reconciled["retries"])
        reports.append([start, end, states, delta, retries])
        totals[0] += delta[5]
        totals[1] += delta[6]
        totals[2] += len(retries[0])
        totals[3] += len(retries[1])

    return {"size": size, "reports": reports, "totals": totals}


def audit_window_trend(
    snapshots: list | tuple,
    size: int = 2,
    done: int = 1,
    retries: int = 1,
    minimum: int = 2,
) -> dict:
    """Classify sliding-window reconciliation reports into trend runs.

    ``snapshots`` and ``size`` share the exact contract of
    :func:`audit_window_report` (every ``TypeError`` and ``ValueError``
    is inherited verbatim; exceptions propagate unchanged).
    ``done``, ``retries`` and ``minimum`` must each be a non-bool int
    >= 1; a wrong type raises ``TypeError`` and a value below 1 raises
    ``ValueError``.

    :func:`audit_window_report` is called exactly once to obtain ``W``
    and is never called again. For each report ``x`` of ``W["reports"]``
    (in order), define ``d = x[3][5]`` (completed), ``r = -x[3][6]``
    (recovered remaining) and ``n = len(x[4][0]) - len(x[4][1])``
    (added minus removed). The window state is ``-1`` (worsening) when
    ``n >= retries``; otherwise it is ``1`` (recovering) when
    ``d >= done`` and ``r >= done``; otherwise it is ``0`` (steady).
    Each entry of ``windows`` is ``[x[0], x[1], d, r, n, state]``.

    Adjacent windows sharing the same non-zero state are merged into a
    maximal run; state ``0`` and a change between ``-1`` and ``1`` both
    break a run. Only runs whose length is at least ``minimum`` are
    kept, in order. Each run is ``[state, first_window_index,
    last_window_index]`` using the window's zero-based index in
    ``windows``.

    ``warning`` is ``[level, fb, fg]`` where ``fb`` and ``fg`` are the
    indices in ``runs`` of the first ``-1`` and first ``1`` run, or
    ``None`` when no such run exists. ``level`` is ``2`` when a ``-1``
    run exists; otherwise it is ``1`` when any window has a state other
    than ``1``; otherwise it is ``0``.

    Returns a dict whose keys in order are ``size``, ``windows``,
    ``runs`` and ``warning``. Every container is JSON-compatible
    (``None`` values kept, numbers never rounded) and the inputs are
    never modified.
    """
    if not isinstance(done, int) or isinstance(done, bool):
        raise TypeError(
            f"done must be a non-bool int, got {type(done).__name__}"
        )
    if not isinstance(retries, int) or isinstance(retries, bool):
        raise TypeError(
            "retries must be a non-bool int, got "
            f"{type(retries).__name__}"
        )
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            "minimum must be a non-bool int, got "
            f"{type(minimum).__name__}"
        )
    if done < 1:
        raise ValueError(f"done must be >= 1, got {done}")
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    windowed = audit_window_report(snapshots, size)

    windows: list = []
    for report in windowed["reports"]:
        d = report[3][5]
        r = -report[3][6]
        n = len(report[4][0]) - len(report[4][1])
        if n >= retries:
            state = -1
        elif d >= done and r >= done:
            state = 1
        else:
            state = 0
        windows.append([report[0], report[1], d, r, n, state])

    runs: list = []
    index = 0
    total = len(windows)
    while index < total:
        state = windows[index][5]
        if state == 0:
            index += 1
            continue
        last = index
        while last + 1 < total and windows[last + 1][5] == state:
            last += 1
        if last - index + 1 >= minimum:
            runs.append([state, index, last])
        index = last + 1

    fb: int | None = None
    fg: int | None = None
    for run_index, run in enumerate(runs):
        if run[0] == -1 and fb is None:
            fb = run_index
        elif run[0] == 1 and fg is None:
            fg = run_index

    if fb is not None:
        level = 2
    elif any(window[5] != 1 for window in windows):
        level = 1
    else:
        level = 0

    return {
        "size": size,
        "windows": windows,
        "runs": runs,
        "warning": [level, fb, fg],
    }


def audit_trend_grid(
    snapshots: list | tuple,
    sizes: list | tuple,
    settings: list | tuple,
) -> dict:
    """Grid of window-trend warnings over window sizes and settings.

    ``snapshots`` shares the exact contract of
    :func:`audit_window_trend` (every ``TypeError`` and ``ValueError``
    is inherited verbatim; exceptions propagate unchanged).

    ``sizes`` must be a non-empty list or tuple of strictly increasing
    non-bool ints, each satisfying ``1 <= size <= len(snapshots)``; a
    wrong container or element type raises ``TypeError``, while an empty
    axis, a non-increasing pair or an out-of-range value raises
    ``ValueError``. ``settings`` must be a non-empty list or tuple whose
    items are lists or tuples of exactly three non-bool positive ints
    ``(done, retries, minimum)``; a wrong container, item or element type
    raises ``TypeError``, while an empty axis, an item that is not a
    triple or a non-positive value raises ``ValueError``.

    With ``sizes`` as the outer axis and ``settings`` as the inner axis,
    :func:`audit_window_trend` is called exactly once per
    ``(size, setting)`` cell as
    ``audit_window_trend(snapshots, size, done, retries, minimum)``;
    a copy of its ``warning`` (``[level, fb, fg]``) is stored as
    ``W[b][p]``.

    For each ``p``, the cells of ``W`` in size order are compressed into
    maximal runs of adjacent equal warnings; each run is
    ``[first_size, last_size, warning_copy]`` and ``stability[p]`` lists
    the runs in order.

    Returns a dict whose keys in order are ``sizes``, ``settings``,
    ``warnings`` and ``stability``: ``sizes`` and ``settings`` are
    converted to lists, ``warnings`` is the B x P matrix ``W`` and
    ``stability`` is the per-setting list of run lists. Every container
    is recursively copied and JSON-compatible (``None`` values kept,
    numbers never rounded); the inputs are never modified.
    """
    if not isinstance(snapshots, (list, tuple)):
        raise TypeError(
            "snapshots must be a list or tuple, got "
            f"{type(snapshots).__name__}"
        )
    if len(snapshots) == 0:
        raise ValueError("snapshots must not be empty")

    if not isinstance(sizes, (list, tuple)):
        raise TypeError(
            f"sizes must be a list or tuple, got {type(sizes).__name__}"
        )
    if len(sizes) == 0:
        raise ValueError("sizes must not be empty")
    size_values: list = []
    for b, size in enumerate(sizes):
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(
                f"sizes[{b}] must be a non-bool int, got "
                f"{type(size).__name__}"
            )
        if size < 1 or size > len(snapshots):
            raise ValueError(
                f"sizes[{b}] must satisfy 1 <= size <= len(snapshots) "
                f"({len(snapshots)}), got {size}"
            )
        if b > 0 and size <= size_values[b - 1]:
            raise ValueError(
                "sizes must be strictly increasing, got "
                f"{[*size_values, size]!r}"
            )
        size_values.append(size)

    if not isinstance(settings, (list, tuple)):
        raise TypeError(
            "settings must be a list or tuple, got "
            f"{type(settings).__name__}"
        )
    if len(settings) == 0:
        raise ValueError("settings must not be empty")
    setting_values: list = []
    for p, setting in enumerate(settings):
        if not isinstance(setting, (list, tuple)):
            raise TypeError(
                f"settings[{p}] must be a list or tuple, got "
                f"{type(setting).__name__}"
            )
        if len(setting) != 3:
            raise ValueError(
                f"settings[{p}] must contain exactly three values "
                f"(done, retries, minimum), got {len(setting)}"
            )
        triple: list = []
        names = ("done", "retries", "minimum")
        for k, value in enumerate(setting):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"settings[{p}][{k}] ({names[k]}) must be a non-bool "
                    f"int, got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(
                    f"settings[{p}][{k}] ({names[k]}) must be >= 1, "
                    f"got {value}"
                )
            triple.append(value)
        setting_values.append(triple)

    warnings: list = [
        [
            list(
                audit_window_trend(
                    snapshots,
                    size,
                    setting[0],
                    setting[1],
                    setting[2],
                )["warning"]
            )
            for setting in setting_values
        ]
        for size in size_values
    ]

    stability: list = []
    for p in range(len(setting_values)):
        segments: list = []
        for b, size in enumerate(size_values):
            warning = warnings[b][p]
            if segments and segments[-1][2] == warning:
                segments[-1][1] = size
            else:
                segments.append([size, size, list(warning)])
        stability.append(segments)

    return {
        "sizes": list(size_values),
        "settings": [list(setting) for setting in setting_values],
        "warnings": warnings,
        "stability": stability,
    }


def audit_trend_turns(
    snapshots: list | tuple,
    sizes: list | tuple,
    settings: list | tuple,
) -> dict:
    """Locate warning turns across the size and setting axes.

    ``snapshots``, ``sizes`` and ``settings`` share the exact contract
    of :func:`audit_trend_grid` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged).

    :func:`audit_trend_grid` is called exactly once to obtain ``G``;
    write ``W = G["warnings"]``, ``B = len(sizes)`` and
    ``P = len(settings)``.

    ``size_changes`` scans, for each ``p`` and then ``b = 1..B-1``, the
    adjacent size cells; whenever ``W[b-1][p] != W[b][p]`` it records
    ``[[b-1, p], [b, p], former_copy, latter_copy, fields]``.
    ``setting_changes`` scans, for each ``b`` and then ``p = 1..P-1``,
    the adjacent setting cells; whenever ``W[b][p-1] != W[b][p]`` it
    records ``[[b, p-1], [b, p], former_copy, latter_copy, fields]``.
    In both cases ``fields`` lists, in the order ``level``,
    ``first_bad``, ``first_good``, the names of the warning positions
    whose values differ between the former and the latter cell.

    With ``X[p] = [[W[b][p] for b in range(B)], G["stability"][p]]``,
    ``consistency`` is the P x P matrix with
    ``consistency[p][q] = (X[p] == X[q])``.

    Returns a dict whose keys in order are ``sizes``, ``settings``,
    ``size_changes``, ``setting_changes`` and ``consistency``. Every
    container is recursively copied and JSON-compatible
    (``None`` values kept, numbers never rounded); the inputs are never
    modified.
    """
    grid = audit_trend_grid(snapshots, sizes, settings)

    warnings = grid["warnings"]
    stability = grid["stability"]
    b_count = len(sizes)
    p_count = len(settings)

    field_names = ("level", "first_bad", "first_good")

    def _changed_fields(former: list, latter: list) -> list:
        return [
            field_names[k]
            for k in range(3)
            if former[k] != latter[k]
        ]

    size_changes: list = []
    for p in range(p_count):
        for b in range(1, b_count):
            former = warnings[b - 1][p]
            latter = warnings[b][p]
            if former != latter:
                size_changes.append(
                    [
                        [b - 1, p],
                        [b, p],
                        list(former),
                        list(latter),
                        _changed_fields(former, latter),
                    ]
                )

    setting_changes: list = []
    for b in range(b_count):
        for p in range(1, p_count):
            former = warnings[b][p - 1]
            latter = warnings[b][p]
            if former != latter:
                setting_changes.append(
                    [
                        [b, p - 1],
                        [b, p],
                        list(former),
                        list(latter),
                        _changed_fields(former, latter),
                    ]
                )

    columns = [
        [[warnings[b][p] for b in range(b_count)], stability[p]]
        for p in range(p_count)
    ]
    consistency = [
        [columns[p] == columns[q] for q in range(p_count)]
        for p in range(p_count)
    ]

    return {
        "sizes": list(grid["sizes"]),
        "settings": [list(setting) for setting in grid["settings"]],
        "size_changes": size_changes,
        "setting_changes": setting_changes,
        "consistency": consistency,
    }


def audit_trend_summary(
    snapshots: list | tuple,
    sizes: list | tuple,
    settings: list | tuple,
) -> dict:
    """Normalize trend turns into indexed events grouped three ways.

    ``snapshots``, ``sizes`` and ``settings`` share the exact contract
    of :func:`audit_trend_turns` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged).

    :func:`audit_trend_turns` is called exactly once to obtain ``T``.
    The entries of ``T["size_changes"]`` followed by the entries of
    ``T["setting_changes"]`` are traversed in that order; with the
    zero-based event index ``e`` in this traversal, each entry ``x`` is
    normalized as
    ``[axis, x[0], x[1], x[2], x[3], x[4], direction]``, where ``axis``
    is ``"size"`` for the first list and ``"setting"`` for the second.
    Comparing the level of the latter cell with the former cell,
    ``direction`` is ``"up"`` when ``x[3][0] > x[2][0]``, ``"down"``
    when ``x[3][0] < x[2][0]`` and ``"shift"`` when they are equal.

    ``config_events[p]`` lists, in ascending order, the event indices
    whose either end coordinate has its position-1 entry equal to ``p``;
    each event contributes at most once. ``field_events`` is a dict with
    keys in the order ``level``, ``first_bad``, ``first_good``; each
    value lists, in ascending order, the event indices whose ``x[4]``
    contains that key. ``direction_events`` is a dict with keys in the
    order ``up``, ``down``, ``shift``; each value lists, in ascending
    order, the event indices with that direction.

    Returns a dict whose keys in order are ``sizes``, ``settings``,
    ``events``, ``config_events``, ``field_events`` and
    ``direction_events``; the first two values are the same-named
    values of ``T``. Every container is recursively copied and
    JSON-compatible (``None`` values kept, numbers never rounded); the
    inputs are never modified.
    """
    turns = audit_trend_turns(snapshots, sizes, settings)

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    events: list = []
    config_events: dict = {
        p: [] for p in range(len(turns["settings"]))
    }
    field_events: dict = {
        "level": [],
        "first_bad": [],
        "first_good": [],
    }
    direction_events: dict = {
        "up": [],
        "down": [],
        "shift": [],
    }

    for axis, change_list in (
        ("size", turns["size_changes"]),
        ("setting", turns["setting_changes"]),
    ):
        for x in change_list:
            e = len(events)
            if x[3][0] > x[2][0]:
                direction = "up"
            elif x[3][0] < x[2][0]:
                direction = "down"
            else:
                direction = "shift"
            events.append(
                [
                    axis,
                    _copy(x[0]),
                    _copy(x[1]),
                    _copy(x[2]),
                    _copy(x[3]),
                    _copy(x[4]),
                    direction,
                ]
            )
            matched: set = set()
            for coordinate in (x[0], x[1]):
                p = coordinate[1]
                if p not in matched:
                    matched.add(p)
                    config_events[p].append(e)
            for key in x[4]:
                field_events[key].append(e)
            direction_events[direction].append(e)

    return {
        "sizes": _copy(turns["sizes"]),
        "settings": _copy(turns["settings"]),
        "events": events,
        "config_events": config_events,
        "field_events": field_events,
        "direction_events": direction_events,
    }


def audit_trend_report(
    snapshots: list | tuple,
    sizes: list | tuple,
    settings: list | tuple,
) -> dict:
    """Per-event coordinates and per-setting tallies of the trend audit.

    ``snapshots``, ``sizes`` and ``settings`` share the exact contract
    of :func:`audit_trend_summary` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged).

    :func:`audit_trend_summary` is called exactly once to obtain ``S``;
    write ``E = S["events"]`` and ``P = len(settings)``. The event
    report ``events[e]`` is
    ``[e] + recursive_copy(E[e]) + [C]``, where ``C`` lists, without
    duplicates and in ascending order, the position-1 (second) entries
    of the event's ``from`` and ``to`` coordinates ``E[e][1]`` and
    ``E[e][2]``; these are the setting columns touched by the event.

    For each ``p = 0..P-1``, ``I`` is a copy of
    ``S["config_events"][p]`` and ``configs[p]`` is
    ``[p, copy(S["settings"][p]), I, DC, FC, peak]``:

    * ``DC`` counts, in the order ``up``, ``down``, ``shift``, the
      events of ``I`` with each direction;
    * ``FC`` counts, in the order ``level``, ``first_bad``,
      ``first_good``, the events of ``I`` whose fields (``E[e][5]``)
      contain that name;
    * ``peak`` is the maximum level touched by the events of ``I``,
      comparing each event's before level ``E[e][3][0]`` with its after
      level ``E[e][4][0]``; when ``I`` is empty it is ``None``.

    ``overview`` is
    ``[len(E), count of non-empty I, global up count, global down
    count, global shift count]``; the global direction counts cover
    every event of ``E`` once.

    Returns a dict whose keys in order are ``sizes``, ``settings``,
    ``events``, ``configs`` and ``overview``; the first two values are
    copies of the same-named values of ``S``. Every container is
    recursively copied and JSON-compatible (``None`` values kept,
    numbers never rounded); the inputs are never modified.
    """
    summary = audit_trend_summary(snapshots, sizes, settings)

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    source_events = summary["events"]
    config_events = summary["config_events"]
    p_count = len(settings)

    direction_index = {"up": 0, "down": 1, "shift": 2}
    field_index = {"level": 0, "first_bad": 1, "first_good": 2}

    events: list = []
    for e, entry in enumerate(source_events):
        columns = sorted({entry[1][1], entry[2][1]})
        events.append([e] + [_copy(item) for item in entry] + [columns])

    configs: list = []
    non_empty_configs = 0
    for p in range(p_count):
        indices = list(config_events[p])
        direction_counts = [0, 0, 0]
        field_counts = [0, 0, 0]
        peak: int | None = None
        if indices:
            non_empty_configs += 1
        for e in indices:
            entry = source_events[e]
            direction_counts[direction_index[entry[6]]] += 1
            for field in entry[5]:
                field_counts[field_index[field]] += 1
            for level in (entry[3][0], entry[4][0]):
                if peak is None or level > peak:
                    peak = level
        configs.append(
            [
                p,
                _copy(summary["settings"][p]),
                indices,
                direction_counts,
                field_counts,
                peak,
            ]
        )

    global_direction_counts = [0, 0, 0]
    for entry in source_events:
        global_direction_counts[direction_index[entry[6]]] += 1

    overview = [
        len(source_events),
        non_empty_configs,
        global_direction_counts[0],
        global_direction_counts[1],
        global_direction_counts[2],
    ]

    return {
        "sizes": _copy(summary["sizes"]),
        "settings": _copy(summary["settings"]),
        "events": events,
        "configs": configs,
        "overview": overview,
    }


def audit_trend_batches(
    snapshots: list | tuple,
    sizes: list | tuple,
    settings: list | tuple,
    capacity: int = 10,
) -> dict:
    """Trend audit events split into fixed-size batches.

    ``snapshots``, ``sizes`` and ``settings`` share the exact contract
    of :func:`audit_trend_report` (every ``TypeError`` and
    ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). :func:`audit_trend_report` is called exactly once to
    obtain ``R``. ``capacity`` must be a non-bool int >= 1; a wrong
    type raises ``TypeError`` and a value below 1 raises
    ``ValueError``.

    Write ``E = R["events"]`` and ``P = len(settings)``. The events
    are traversed in the order of ``E``, ``capacity`` events per
    batch (the last batch may be shorter); each batch covers the
    half-open interval ``[beg, end)``.

    Within a batch ``D`` counts, in the order ``up``, ``down``,
    ``shift``, each event's direction ``e[7]``. For each
    ``p = 0..P-1``, ``I`` is the set of batch events whose setting
    columns (``e[-1]``) contain ``p``. When ``I`` is empty,
    ``peaks[p]`` is ``None`` and ``changes[p]`` is ``0``; otherwise
    ``peaks[p]`` is the maximum of each event's before level
    ``e[4][0]`` and after level ``e[5][0]``, and ``changes[p]`` is the
    sum of ``e[5][0] - e[4][0]`` over ``I``. Each batch is stored as
    ``[beg, end, D, peaks, changes]``.

    Returns a dict whose keys in order are ``sizes``, ``settings``,
    ``capacity``, ``batches`` and ``overview``; the first two values
    copy the same-named values of ``R``, and ``overview`` is
    ``[len(E), number of batches, item-wise sums across the batches
    of each batch's D entries (a length-3 vector), item-wise sums
    across the batches of each batch's changes entries (a length-P
    vector)]``. When ``E`` is empty ``batches`` is empty and the last
    two overview vectors are ``[0, 0, 0]`` and ``P`` zeros. Every
    container is recursively copied and JSON-compatible
    (``None`` values kept, numbers never rounded); the inputs are
    never modified.
    """
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise TypeError(
            f"capacity must be an int, got {type(capacity).__name__}"
        )
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")

    report = audit_trend_report(snapshots, sizes, settings)

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    events = report["events"]
    p_count = len(settings)

    direction_index = {"up": 0, "down": 1, "shift": 2}

    batches: list = []
    direction_totals = [0, 0, 0]
    changes_totals = [0] * p_count
    for beg in range(0, len(events), capacity):
        end = min(beg + capacity, len(events))
        direction_counts = [0, 0, 0]
        peaks: list = [None] * p_count
        changes = [0] * p_count
        for e in events[beg:end]:
            direction_counts[direction_index[e[7]]] += 1
            for p in e[-1]:
                before = e[4][0]
                after = e[5][0]
                if peaks[p] is None or before > peaks[p]:
                    peaks[p] = before
                if after > peaks[p]:
                    peaks[p] = after
                changes[p] += after - before
        batches.append([beg, end, direction_counts, peaks, changes])
        for k in range(3):
            direction_totals[k] += direction_counts[k]
        for p in range(p_count):
            changes_totals[p] += changes[p]

    return {
        "sizes": _copy(report["sizes"]),
        "settings": _copy(report["settings"]),
        "capacity": capacity,
        "batches": batches,
        "overview": [
            len(events),
            len(batches),
            direction_totals,
            changes_totals,
        ],
    }


def audit_trend_batch_alerts(
    snapshots: list | tuple,
    sizes: list | tuple,
    settings: list | tuple,
    capacity: int = 10,
    threshold: int = 1,
    minimum: int = 1,
) -> dict:
    """Maximal active batch runs per setting column.

    ``snapshots``, ``sizes``, ``settings`` and ``capacity`` share the
    exact contract of :func:`audit_trend_batches` (every ``TypeError``
    and ``ValueError`` is inherited verbatim; exceptions propagate
    unchanged). ``threshold`` and ``minimum`` must each be a non-bool
    int >= 1; a wrong type raises ``TypeError`` and a value below 1
    raises ``ValueError``.

    :func:`audit_trend_batches` is called exactly once to obtain ``A``
    and is never called again. Write ``P = len(settings)``. For each
    configuration ``p = 0..P-1`` the batches of ``A["batches"]`` are
    scanned in batch order; a batch ``b`` is *active* for ``p`` when
    ``abs(changes[p]) >= threshold`` (the batch's ``changes`` is
    ``batch[4]``). Consecutive active batches are merged into maximal
    runs and only runs whose length is at least ``minimum`` are kept.
    Each run is ``[p, first_batch, last_batch, length, net, peak]``
    where ``net`` is the :func:`math.fsum` of the run's changes for
    ``p`` and ``peak`` is the maximum of the run's ``peaks[p]`` values
    (batch ``peaks`` is ``batch[3]``); every active batch has a
    non-``None`` peak for ``p``. All runs are ordered by
    ``(p, first_batch)``.

    ``priority`` lists the configurations having at least one run,
    ordered by the :func:`math.fsum` of ``abs(net)`` over that
    configuration's runs (descending), then by the number of batches
    covered by its runs (descending), then by ``p`` (ascending).

    Returns a dict whose keys in order are ``sizes``, ``settings``,
    ``capacity``, ``threshold``, ``minimum``, ``runs`` and
    ``priority``; the first three values copy the same-named values of
    ``A``. When no configuration has a qualifying run, ``runs`` and
    ``priority`` are both empty. Every container is recursively copied
    and JSON-compatible (numbers never rounded); the inputs are never
    modified.
    """
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise TypeError(
            f"capacity must be an int, got {type(capacity).__name__}"
        )
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        raise TypeError(
            "threshold must be a non-bool int, got "
            f"{type(threshold).__name__}"
        )
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            "minimum must be a non-bool int, got "
            f"{type(minimum).__name__}"
        )
    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    batched = audit_trend_batches(snapshots, sizes, settings, capacity)

    def _copy(value: object) -> object:
        if isinstance(value, dict):
            return {key: _copy(value[key]) for key in value}
        if isinstance(value, list):
            return [_copy(item) for item in value]
        return value

    batches = batched["batches"]
    p_count = len(settings)
    batch_count = len(batches)

    runs: list = []
    per_config: list = []
    for p in range(p_count):
        config_runs: list = []
        b = 0
        while b < batch_count:
            if abs(batches[b][4][p]) < threshold:
                b += 1
                continue
            first = b
            while (
                b + 1 < batch_count
                and abs(batches[b + 1][4][p]) >= threshold
            ):
                b += 1
            last = b
            length = last - first + 1
            if length >= minimum:
                net = math.fsum(batches[k][4][p] for k in range(first, last + 1))
                peak = batches[first][3][p]
                for k in range(first + 1, last + 1):
                    candidate = batches[k][3][p]
                    if candidate > peak:
                        peak = candidate
                run = [p, first, last, length, net, peak]
                config_runs.append(run)
                runs.append(run)
            b = last + 1
        per_config.append(config_runs)

    priority: list = []
    for p, config_runs in enumerate(per_config):
        if config_runs:
            net_weight = math.fsum(abs(run[4]) for run in config_runs)
            covered = sum(run[3] for run in config_runs)
            priority.append((p, net_weight, covered))
    priority.sort(key=lambda item: (-item[1], -item[2], item[0]))

    return {
        "sizes": _copy(batched["sizes"]),
        "settings": _copy(batched["settings"]),
        "capacity": capacity,
        "threshold": threshold,
        "minimum": minimum,
        "runs": _copy(runs),
        "priority": [item[0] for item in priority],
    }


def receptor_level_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
) -> list[list[int]]:
    """Weighted quantiles of the per-receptor health level.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1]`` and in non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    Returns ``Q`` where, with ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - H[k][i]) / (sigma * sqrt(2)))``
      when ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= H[k][i]``
      else ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``P[i][r] = fsum(w[k] * p[k][i][r] for k in range(K))`` is the
      weighted probability that receptor ``i`` is at level ``r``;
    * ``Q[m][i]`` is the smallest level ``r`` whose cumulative probability
      ``fsum(P[i][s] for s in range(r + 1))`` is ``>= quantiles[m]``
      (``0`` when ``quantiles[m] == 0``).

    ``Q`` is an M x N list of lists of ints, in ``quantiles`` then
    receptor input order.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1], got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    if weights is None:
        w = [1.0 / n_scenarios] * n_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != n_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {n_scenarios}, "
                f"got {len(weights)}"
            )
        raw = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            raw.append(value)
        total_weight = math.fsum(raw)
        if total_weight <= 0:
            raise ValueError(f"weights sum must be > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw]

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    P = [
        [
            math.fsum(
                w[k] * per_scenario[k][i][r] for k in range(n_scenarios)
            )
            for r in range(4)
        ]
        for i in range(n_receptors)
    ]

    Q: list[list[int]] = []
    for q in qs:
        row: list[int] = []
        for i in range(n_receptors):
            if q == 0.0:
                row.append(0)
            else:
                chosen = 3
                for r in range(4):
                    if math.fsum(P[i][s] for s in range(r + 1)) >= q:
                        chosen = r
                        break
                row.append(chosen)
        Q.append(row)

    return Q


def receptor_level_transition(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
) -> list[list[list[float]]]:
    """Health-level transition matrices between consecutive scenarios.

    Scenario errors are treated as independent and receptors are equally
    weighted.

    H: K x N (K >= 2, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``;
    * ``T[t][a][b] = fsum(p[t][i][a] * p[t + 1][i][b]
      for i in range(N)) / N`` for ``t = 0..K - 2`` and ``a, b = 0..3``.

    Returns a ``(K - 1) x 4 x 4`` ``list[list[list[float]]]`` indexed by
    transition, then level of the earlier scenario, then level of the
    following scenario, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(impacts) < 2:
        raise ValueError(f"H must have at least 2 scenarios, got {len(impacts)}")
    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    transitions: list[list[list[float]]] = []
    for t in range(n_scenarios - 1):
        matrix = [
            [
                math.fsum(
                    per_scenario[t][i][a] * per_scenario[t + 1][i][b]
                    for i in range(n_receptors)
                )
                / n_receptors
                for b in range(4)
            ]
            for a in range(4)
        ]
        transitions.append(matrix)

    return transitions


def receptor_level_rise_distribution(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
) -> list[list[float]]:
    """Per-receptor distribution of the number of health-level rises.

    Scenario errors are treated as independent and scenarios are folded in
    their given order; a "rise" is a transition from level ``a`` at scenario
    ``k - 1`` to a strictly higher level ``b`` (``a < b``) at scenario ``k``.

    H: K x N (K >= 2, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each receptor the recursion starts from ``D[r][0] = p[0][i][r]``
    (all other entries zero) and, for ``k = 1..K - 1`` and ``c = 0..k``,

    ``D'[b][c] = fsum(D[a][c - (a < b)] * p[k][i][b] for a in range(4))``,

    omitting out-of-range terms. The result is
    ``R[i][c] = fsum(D[r][c] for r in range(4))``.

    Returns an N x K ``list[list[float]]`` indexed by receptor, then number
    of rises (``0..K - 1``), unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(impacts) < 2:
        raise ValueError(f"H must have at least 2 scenarios, got {len(impacts)}")
    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    result: list[list[float]] = []
    for i in range(n_receptors):
        distribution: list[list[float]] = [
            [per_scenario[0][i][r]] for r in range(4)
        ]
        for k in range(1, n_scenarios):
            probabilities = per_scenario[k][i]
            updated = [[0.0] * (k + 1) for _ in range(4)]
            for b in range(4):
                for c in range(k + 1):
                    updated[b][c] = math.fsum(
                        distribution[a][c - (1 if a < b else 0)] * probabilities[b]
                        for a in range(4)
                        if 0 <= c - (1 if a < b else 0) <= k - 1
                    )
            distribution = updated
        result.append(
            [
                math.fsum(distribution[r][c] for r in range(4))
                for c in range(n_scenarios)
            ]
        )

    return result


def receptor_level_run_cvar(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
) -> list[list[list[float]]]:
    """Upper-tail conditional value at risk of the longest level run.

    Scenario errors are treated as independent and scenarios are folded in
    their given order, independently for every receptor and level.

    H: K x N (K > 0, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each receptor ``i`` and level ``l`` a Markov recursion over states
    ``(r, m)`` (current run length ``r`` and longest run seen so far ``m``)
    starts from ``D[(0, 0)] = 1.0``. Scenarios are processed in order; with
    state keys in lexicographic order, an ``l`` event with probability
    ``p[k][i][l]`` moves to ``(min(r + 1, K), max(m, r + 1))`` while any
    other event with probability ``1 - p[k][i][l]`` moves to ``(0, m)``;
    probabilities landing on the same target state are combined with
    ``math.fsum``. The terminal mass on each longest run is
    ``G[m] = fsum(D[(r, m)] for r in 0..K)``. For each quantile ``q``, the
    atoms ``G[m]`` are taken from ``m = K`` down to ``0``, contributing
    ``take = min(G[m], 1 - q)`` each, and
    ``C[q][i][l] = fsum(m * take) / (1 - q)``.

    Returns an M x N x 4 ``list[list[list[float]]]`` indexed by quantile
    (in ``quantiles`` order), then receptor, then level, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    result: list[list[list[float]]] = [
        [[0.0] * 4 for _ in range(n_receptors)] for _ in qs
    ]
    for i in range(n_receptors):
        for l in range(4):
            state: dict[tuple[int, int], float] = {(0, 0): 1.0}
            for k in range(n_scenarios):
                hit = per_scenario[k][i][l]
                miss = 1.0 - hit
                updated: dict[tuple[int, int], list[float]] = {}
                for (r, m) in sorted(state):
                    probability = state[(r, m)]
                    hit_target = (min(r + 1, n_scenarios), max(m, r + 1))
                    miss_target = (0, m)
                    updated.setdefault(hit_target, []).append(probability * hit)
                    updated.setdefault(miss_target, []).append(probability * miss)
                state = {
                    target: math.fsum(terms) for target, terms in updated.items()
                }
            longest_terms: list[list[float]] = [
                [] for _ in range(n_scenarios + 1)
            ]
            for (r, m), probability in state.items():
                longest_terms[m].append(probability)
            G = [math.fsum(terms) for terms in longest_terms]
            for m_index, quantile in enumerate(qs):
                remaining = 1.0 - quantile
                terms: list[float] = []
                for m in range(n_scenarios, -1, -1):
                    if remaining <= 0.0:
                        break
                    take = min(G[m], remaining)
                    terms.append(take * m)
                    remaining -= take
                result[m_index][i][l] = math.fsum(terms) / (1.0 - quantile)

    return result


def receptor_level_run_distribution(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
) -> list[list[list[float]]]:
    """Per-receptor distribution of the longest run of each health level.

    Scenario errors are treated as independent and scenarios are folded in
    their given order, independently for every receptor and level.

    H: K x N (K > 0, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each receptor ``i`` and level ``l`` a Markov recursion over states
    ``(r, m)`` (current run length ``r`` and longest run seen so far ``m``)
    starts from ``D[(0, 0)] = 1.0``. Scenarios are processed in order; with
    state keys in lexicographic order, an ``l`` event with probability
    ``p[k][i][l]`` moves to ``(min(r + 1, K), max(m, r + 1))`` while any
    other event with probability ``1 - p[k][i][l]`` moves to ``(0, m)``;
    probabilities landing on the same target state are combined with
    ``math.fsum``. The result is ``R[i][l][m]`` equal to the final
    probability mass on states whose longest run is ``m``.

    Returns an N x 4 x (K + 1) ``list[list[list[float]]]`` indexed by
    receptor, then level, then longest run length (``0..K``), unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    result: list[list[list[float]]] = []
    for i in range(n_receptors):
        receptor: list[list[float]] = []
        for l in range(4):
            state: dict[tuple[int, int], float] = {(0, 0): 1.0}
            for k in range(n_scenarios):
                hit = per_scenario[k][i][l]
                miss = 1.0 - hit
                updated: dict[tuple[int, int], list[float]] = {}
                for (r, m) in sorted(state):
                    probability = state[(r, m)]
                    hit_target = (min(r + 1, n_scenarios), max(m, r + 1))
                    miss_target = (0, m)
                    updated.setdefault(hit_target, []).append(probability * hit)
                    updated.setdefault(miss_target, []).append(probability * miss)
                state = {
                    target: math.fsum(terms) for target, terms in updated.items()
                }
            longest_terms: list[list[float]] = [
                [] for _ in range(n_scenarios + 1)
            ]
            for (r, m), probability in state.items():
                longest_terms[m].append(probability)
            receptor.append([math.fsum(terms) for terms in longest_terms])
        result.append(receptor)

    return result


def receptor_level_run_entropy(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
) -> list[list[float]]:
    """Shannon entropy of the longest level-run distribution per receptor.

    Scenario errors are treated as independent and scenarios are folded in
    their given order, independently for every receptor and level.

    H: K x N (K > 0, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each receptor ``i`` and level ``l`` a Markov recursion over states
    ``(r, m)`` (current run length ``r`` and longest run seen so far ``m``)
    starts from ``D[(0, 0)] = 1.0``. Scenarios are processed in order; with
    state keys in lexicographic order, an ``l`` event with probability
    ``p[k][i][l]`` moves to ``(min(r + 1, K), max(m, r + 1))`` while any
    other event with probability ``1 - p[k][i][l]`` moves to ``(0, m)``;
    probabilities landing on the same target state are combined with
    ``math.fsum``. The terminal mass on each longest run is
    ``G[m] = fsum(D[(r, m)] for r in 0..K)`` and
    ``E[i][l] = -fsum(G[m] * log(G[m]) for G[m] > 0)``.

    Returns an N x 4 ``list[list[float]]`` indexed by receptor, then level,
    unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    result: list[list[float]] = []
    for i in range(n_receptors):
        receptor: list[float] = []
        for l in range(4):
            state: dict[tuple[int, int], float] = {(0, 0): 1.0}
            for k in range(n_scenarios):
                hit = per_scenario[k][i][l]
                miss = 1.0 - hit
                updated: dict[tuple[int, int], list[float]] = {}
                for (r, m) in sorted(state):
                    probability = state[(r, m)]
                    hit_target = (min(r + 1, n_scenarios), max(m, r + 1))
                    miss_target = (0, m)
                    updated.setdefault(hit_target, []).append(probability * hit)
                    updated.setdefault(miss_target, []).append(probability * miss)
                state = {
                    target: math.fsum(terms) for target, terms in updated.items()
                }
            longest_terms: list[list[float]] = [
                [] for _ in range(n_scenarios + 1)
            ]
            for (r, m), probability in state.items():
                longest_terms[m].append(probability)
            G = [math.fsum(terms) for terms in longest_terms]
            receptor.append(
                -math.fsum(g * math.log(g) for g in G if g > 0.0)
            )
        result.append(receptor)

    return result


def receptor_level_run_quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    quantiles: list[float] | tuple[float, ...],
) -> list[list[list[int]]]:
    """Quantiles of the longest run of each health level per receptor.

    Scenario errors are treated as independent and scenarios are folded in
    their given order, independently for every receptor and level.

    H: K x N (K > 0, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    quantiles: non-empty list or tuple of finite numbers, each in
        ``[0, 1)`` and in non-decreasing order.
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each receptor ``i`` and level ``l`` a Markov recursion over states
    ``(r, m)`` (current run length ``r`` and longest run seen so far ``m``)
    starts from ``D[(0, 0)] = 1.0``. Scenarios are processed in ascending
    order; with state keys in lexicographic order, an ``l`` event with
    probability ``p[k][i][l]`` moves to
    ``(min(r + 1, K), max(m, r + 1))`` while any other event moves to
    ``(0, m)``; probabilities landing on the same target state are combined
    with ``math.fsum``. The terminal mass on each longest run is
    ``G[m] = fsum(D[(r, m)] for r in 0..K)``. For quantile ``q`` the result
    is ``0`` when ``q == 0``, otherwise the smallest ``m`` such that
    ``fsum(G[r] for r in 0..m) >= q``.

    Returns an M x N x 4 ``list[list[list[int]]]`` indexed by quantile
    (in ``quantiles`` order), then receptor, then level, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    qs: list[float] = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"quantiles[{m}] must be in [0, 1), got {value!r}")
        if m > 0 and value < qs[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got {[*qs, value]!r}"
            )
        qs.append(value)

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    result: list[list[list[int]]] = [
        [[0] * 4 for _ in range(n_receptors)] for _ in qs
    ]
    for i in range(n_receptors):
        for l in range(4):
            state: dict[tuple[int, int], float] = {(0, 0): 1.0}
            for k in range(n_scenarios):
                hit = per_scenario[k][i][l]
                miss = 1.0 - hit
                updated: dict[tuple[int, int], list[float]] = {}
                for (r, m) in sorted(state):
                    probability = state[(r, m)]
                    hit_target = (min(r + 1, n_scenarios), max(m, r + 1))
                    miss_target = (0, m)
                    updated.setdefault(hit_target, []).append(probability * hit)
                    updated.setdefault(miss_target, []).append(probability * miss)
                state = {
                    target: math.fsum(terms) for target, terms in updated.items()
                }
            longest_terms: list[list[float]] = [
                [] for _ in range(n_scenarios + 1)
            ]
            for (r, m), probability in state.items():
                longest_terms[m].append(probability)
            G = [math.fsum(terms) for terms in longest_terms]
            for m_index, quantile in enumerate(qs):
                if quantile == 0.0:
                    continue
                cumulative_terms: list[float] = []
                for m in range(n_scenarios + 1):
                    cumulative_terms.append(G[m])
                    if math.fsum(cumulative_terms) >= quantile:
                        result[m_index][i][l] = m
                        break

    return result


def receptor_level_run_stats(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    minimum: int = 1,
) -> tuple[list[list[float]], list[list[float]]]:
    """Per-receptor run statistics for each health level.

    Scenario errors are treated as independent and scenarios are folded in
    their given order, independently for every receptor and level.

    H: K x N (K > 0, N > 0) scenario x receptor matrix of health impacts;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        (negative values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    thresholds: list or tuple of exactly 3 finite, non-negative, strictly
        increasing numbers.
    minimum: minimum run length that qualifies as a completed run; a
        non-bool int >= 1 (default 1).
    With ``mu = H[k][i]`` and ``sigma = W[k][i]``:

    * ``q[j] = 0.5 * erfc((thresholds[j] - mu) / (sigma * sqrt(2)))`` when
      ``sigma > 0``, otherwise ``1.0`` if ``thresholds[j] <= mu`` else
      ``0.0``;
    * ``p[k][i] = [1 - q0, q0 - q1, q1 - q2, q2]`` holds the probabilities
      of the four health levels for scenario ``k`` at receptor ``i``.

    For each receptor ``i`` and level ``l`` a Markov recursion over states
    ``(r, c)`` (current run length ``r`` truncated at ``m = minimum`` and
    completed qualifying runs ``c``) starts from ``D[(0, 0)] = 1.0``.
    Scenarios are processed in order; with state keys in lexicographic
    order, an ``l`` event with probability ``p[k][i][l]`` moves to
    ``(min(r + 1, m), c + 1)`` when ``r < m`` and ``r + 1 == m`` and to
    ``(min(r + 1, m), c)`` otherwise, while any other event with
    probability ``1 - p[k][i][l]`` moves to ``(0, c)``; probabilities
    landing on the same target state are combined with ``math.fsum``.
    Then ``probability[i][l] = fsum(D[(r, c)] for c >= 1)`` is the
    probability of at least one qualifying run and
    ``expected_runs[i][l] = fsum(c * D[(r, c)])`` the expected number of
    completed qualifying runs.

    Returns ``(probability, expected_runs)``, each an N x 4
    ``list[list[float]]`` indexed by receptor, then level, unrounded.
    """
    impacts = _validate_matrix("H", H, non_negative=False)
    uncertainties = _validate_matrix("W", W)

    if len(uncertainties) != len(impacts):
        raise ValueError(
            f"H and W must have the same number of scenarios, "
            f"got {len(impacts)} and {len(uncertainties)}"
        )
    n_scenarios = len(impacts)
    n_receptors = len(impacts[0])
    for k, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"W[{k}] must have {n_receptors} elements, got {len(row)}"
            )

    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have exactly 3 elements, got {len(thresholds)}"
        )
    levels = []
    for j, item in enumerate(thresholds):
        if not _is_number(item):
            raise TypeError(
                f"thresholds[{j}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"thresholds[{j}]", value)
        if value < 0:
            raise ValueError(f"thresholds[{j}] must be >= 0, got {value!r}")
        levels.append(value)
    for j in range(1, 3):
        if not levels[j] > levels[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {levels!r}"
            )

    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            f"minimum must be an int, got {type(minimum).__name__}"
        )
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    per_scenario: list[list[list[float]]] = []
    for k in range(n_scenarios):
        rows: list[list[float]] = []
        for i in range(n_receptors):
            mu = impacts[k][i]
            sigma = uncertainties[k][i]
            q: list[float] = []
            for level in levels:
                if sigma > 0:
                    q.append(0.5 * math.erfc((level - mu) / (sigma * math.sqrt(2))))
                else:
                    q.append(1.0 if level <= mu else 0.0)
            rows.append([1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]])
        per_scenario.append(rows)

    probability: list[list[float]] = []
    expected_runs: list[list[float]] = []
    for i in range(n_receptors):
        probability_row: list[float] = []
        expected_row: list[float] = []
        for l in range(4):
            state: dict[tuple[int, int], float] = {(0, 0): 1.0}
            for k in range(n_scenarios):
                hit = per_scenario[k][i][l]
                miss = 1.0 - hit
                updated: dict[tuple[int, int], list[float]] = {}
                for (r, c) in sorted(state):
                    mass = state[(r, c)]
                    completed = 1 if r < minimum and r + 1 == minimum else 0
                    hit_target = (min(r + 1, minimum), c + completed)
                    miss_target = (0, c)
                    updated.setdefault(hit_target, []).append(mass * hit)
                    updated.setdefault(miss_target, []).append(mass * miss)
                state = {
                    target: math.fsum(terms) for target, terms in updated.items()
                }
            probability_row.append(
                math.fsum(mass for (r, c), mass in state.items() if c >= 1)
            )
            expected_row.append(
                math.fsum(c * mass for (r, c), mass in state.items())
            )
        probability.append(probability_row)
        expected_runs.append(expected_row)

    return probability, expected_runs

