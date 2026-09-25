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
    "at_least_count_probability",
    "exceedance_probability",
    "excess_interval",
    "excess_warning",
    "expected_excess",
    "level_probability",
    "max_level_probability",
    "max_level_quantile",
    "quantile",
    "receptor_count_interval",
    "receptor_count_probability",
    "receptor_count_quantile",
    "receptor_excess",
    "receptor_excess_interval",
    "receptor_excess_warning",
    "receptor_expected_excess",
    "receptor_level_count_interval",
    "receptor_level_count_probability",
    "receptor_level_count_quantile",
    "receptor_level_count_share",
    "receptor_level_count_warning",
    "receptor_level_event_probability",
    "receptor_level_interval",
    "receptor_level_probability",
    "receptor_level_quantile",
    "receptor_mitigation_frontier",
    "receptor_mitigation_plan",
    "receptor_quantile",
    "receptor_risk_interval",
    "receptor_risk_quantile",
    "receptor_risk_share",
    "risk_contribution",
    "risk_interval",
    "risk_probability",
    "risk_share",
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

