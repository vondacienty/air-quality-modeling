"""Health impact assessment relative to a baseline scenario."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number

__all__ = [
    "aggregate",
    "assess",
    "exceedance_probability",
    "level_probability",
    "quantile",
    "risk_interval",
    "risk_probability",
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


def quantile(
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float]]:
    """Weighted quantiles of per-scenario total impacts and their intervals.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (negative
        values allowed).
    W: matrix of uncertainties with the same shape as ``H``; each value
        finite and >= 0.
    quantiles: non-empty list or tuple of finite numbers in ``[0, 1]`` in
        non-decreasing order.
    weights: ``None`` (the default) or a K-long list or tuple of finite,
        non-negative entries whose ``fsum`` is positive. With ``None``
        every scenario has weight ``1 / K``; otherwise the weights are
        normalized by their ``fsum``.
    z: number of standard deviations for the interval half-width; finite
        and >= 0 (default 1.96).
    Returns ``(Q, L, U)`` where, with
    ``mu[k] = fsum(H[k])`` and ``sigma[k] = hypot(*W[k])``:

    * ``l[k] = mu[k] - z * sigma[k]``, ``u[k] = mu[k] + z * sigma[k]``;
    * for each requested quantile ``q``, scenarios are sorted by
      ``(value, k)`` ascending (by ``mu`` for ``Q``, by ``l`` for ``L``
      and by ``u`` for ``U``); with ``q == 0`` the first scenario is
      chosen, otherwise the first scenario whose cumulative normalized
      weight is ``>= q``;
    * the corresponding ``mu``, ``l`` and ``u`` values are collected.

    All three results are M-long lists of floats in ``quantiles`` order,
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
        if not 0 <= value <= 1:
            raise ValueError(f"quantiles[{j}] must be in [0, 1], got {value!r}")
        qs.append(value)
    for j in range(1, len(qs)):
        if qs[j] < qs[j - 1]:
            raise ValueError(f"quantiles must be non-decreasing, got {qs!r}")

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

    def select(values: list[float], q: float) -> float:
        order = sorted(range(n_scenarios), key=lambda k: (values[k], k))
        if q == 0:
            return values[order[0]]
        prefix: list[float] = []
        for k in order:
            prefix.append(w[k])
            if math.fsum(prefix) >= q:
                return values[k]
        return values[order[-1]]

    Q = [select(mus, q) for q in qs]
    L = [select(lowers, q) for q in qs]
    U = [select(uppers, q) for q in qs]

    return Q, L, U
