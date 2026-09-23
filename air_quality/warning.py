"""Air quality warning level assessment relative to a baseline scenario."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number, _validate_scalar

__all__ = [
    "assess",
    "forecast",
    "forecast_interval",
    "aggregate",
    "risk_interval",
    "window",
    "transition",
    "exceedance_probability",
    "level_probability",
    "quantile",
    "alert_run",
    "alert_run_distribution",
    "alert_run_profile",
]


def _validate_matrix(
    name: str, matrix: object, allow_negative: bool
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
            if not allow_negative and value < 0:
                raise ValueError(f"{name}[{k}][{i}] must be >= 0, got {value!r}")
            values.append(value)
        validated.append(values)
    return validated


def _validate_thresholds(thresholds: object) -> list[float]:
    if not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"thresholds must be a list or tuple, got {type(thresholds).__name__}"
        )
    if len(thresholds) != 3:
        raise ValueError(
            f"thresholds must have 3 elements, got {len(thresholds)}"
        )
    validated = []
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
        if j > 0 and value <= validated[j - 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got "
                f"thresholds[{j - 1}]={validated[j - 1]!r} and "
                f"thresholds[{j}]={value!r}"
            )
        validated.append(value)
    return validated


def _validate_vector(name: str, vector: object, length: int) -> list[float]:
    if not isinstance(vector, (list, tuple)):
        raise TypeError(
            f"{name} must be a list or tuple, got {type(vector).__name__}"
        )
    if len(vector) != length:
        raise ValueError(f"{name} must have {length} elements, got {len(vector)}")
    validated = []
    for i, item in enumerate(vector):
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
    H: list[list[float]] | tuple[tuple[float, ...], ...],
    W: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    baseline: int = 0,
) -> tuple[
    list[int],
    list[tuple[float, float]],
    list[float],
]:
    """Assess warning levels of scenarios relative to a baseline scenario.

    H: non-empty scenario x receptor matrix of health impacts; both the
        outer container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (may be
        negative).
    W: matrix of health impact uncertainties with the same shape and
        constraints as ``H``, except each value must be >= 0.
    thresholds: list or tuple of 3 strictly increasing, finite, non-negative
        warning thresholds.
    baseline: index of the baseline scenario; a non-bool int in
        ``[0, len(H))`` (default 0).
    Returns ``(levels, evidence, scores)`` where, for each non-baseline
    scenario ``k``:

    * ``I = fsum(max(h, 0) for h in H[k])``,
    * ``V = hypot(*W[k])``, equal to ``sqrt(fsum(w ** 2 for w in W[k]))``
      but without intermediate overflow,
    * ``scores[k] = I + V``,
    * ``levels[k]`` is the count of thresholds ``<= scores[k]`` (0-3),
    * ``evidence[k] = (I, V)``.

    The baseline entries are ``levels[k] == 0``, ``evidence[k] == (0.0,
    0.0)`` and ``scores[k] == 0.0``. All results are plain lists in input
    order and are not rounded.
    """
    impacts = _validate_matrix("H", H, allow_negative=True)
    uncertainties = _validate_matrix("W", W, allow_negative=False)

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

    levels_thresholds = _validate_thresholds(thresholds)

    if not isinstance(baseline, int) or isinstance(baseline, bool):
        raise TypeError(
            f"baseline must be an int, got {type(baseline).__name__}"
        )
    if not 0 <= baseline < len(impacts):
        raise ValueError(
            f"baseline must be in [0, {len(impacts)}), got {baseline}"
        )

    levels: list[int] = []
    evidence: list[tuple[float, float]] = []
    scores: list[float] = []
    for k in range(len(impacts)):
        if k == baseline:
            levels.append(0)
            evidence.append((0.0, 0.0))
            scores.append(0.0)
        else:
            i_total = math.fsum(max(h, 0.0) for h in impacts[k])
            v_total = math.hypot(*uncertainties[k])
            score = i_total + v_total
            level = sum(1 for t in levels_thresholds if t <= score)
            levels.append(level)
            evidence.append((i_total, v_total))
            scores.append(score)

    return levels, evidence, scores


def forecast(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[
    list[int],
    list[tuple[float, float]],
    list[float],
]:
    """Forecast population-weighted air quality warning levels over time.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    Returns ``(levels, evidence, scores)``, plain lists in time order,
    where for each time ``t`` and receptor ``i``::

        h[i] = population[i] * beta * values[t][i]
        w[i] = population[i] * beta * uncertainty[t][i]
        I[t] = fsum(max(h[i], 0.0) for i in range(N))
        V[t] = hypot(*w)
        score[t] = I[t] + V[t]
        level[t] = number of thresholds <= score[t]   (0-3)

    and ``evidence[t] = (I[t], V[t])``. Results are not rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)

    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for t, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)

    levels: list[int] = []
    evidence: list[tuple[float, float]] = []
    scores: list[float] = []
    for t in range(len(val_matrix)):
        h_row = [populations[i] * beta_value * val_matrix[t][i]
                 for i in range(n_receptors)]
        w_row = [populations[i] * beta_value * unc_matrix[t][i]
                 for i in range(n_receptors)]
        i_total = math.fsum(max(h, 0.0) for h in h_row)
        v_total = math.hypot(*w_row)
        score = i_total + v_total
        level = sum(1 for t_value in levels_thresholds if t_value <= score)
        levels.append(level)
        evidence.append((i_total, v_total))
        scores.append(score)

    return levels, evidence, scores


def forecast_interval(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    z: float = 1.96,
) -> tuple[
    list[int],
    list[int],
    list[float],
    list[float],
]:
    """Forecast warning-level intervals over time under value uncertainty.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    z: finite, non-negative confidence multiplier (default 1.96).
    Returns ``(low_levels, high_levels, low_scores, high_scores)``, four
        plain lists of length K in time order, where for each time ``t``
        and receptor ``i``::

        lo[t][i] = max(values[t][i] - z * uncertainty[t][i], 0.0)
        hi[t][i] = values[t][i] + z * uncertainty[t][i]
        L[t] = fsum(population[i] * beta * lo[t][i] for i in range(N))
        H[t] = fsum(population[i] * beta * hi[t][i] for i in range(N))
        low_levels[t]  = number of thresholds <= L[t]   (0-3)
        high_levels[t] = number of thresholds <= H[t]   (0-3)
        low_scores[t], high_scores[t] = L[t], H[t]

    Levels are ints and scores are floats; results are not rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)

    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for t, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)
    z_value = _validate_scalar("z", z)
    if z_value < 0:
        raise ValueError(f"z must be >= 0, got {z_value!r}")

    low_levels: list[int] = []
    high_levels: list[int] = []
    low_scores: list[float] = []
    high_scores: list[float] = []
    for t in range(len(val_matrix)):
        low_score = math.fsum(
            populations[i] * beta_value
            * max(val_matrix[t][i] - z_value * unc_matrix[t][i], 0.0)
            for i in range(n_receptors)
        )
        high_score = math.fsum(
            populations[i] * beta_value
            * (val_matrix[t][i] + z_value * unc_matrix[t][i])
            for i in range(n_receptors)
        )
        low_levels.append(
            sum(1 for t_value in levels_thresholds if t_value <= low_score)
        )
        high_levels.append(
            sum(1 for t_value in levels_thresholds if t_value <= high_score)
        )
        low_scores.append(low_score)
        high_scores.append(high_score)

    return low_levels, high_levels, low_scores, high_scores


def aggregate(
    T: list[list[float]] | tuple[tuple[float, ...], ...],
    U: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[int, float, int, int, float, float, list[float], list[float]]:
    """Aggregate weighted scenario means into a population warning level.

    T: non-empty list or tuple of K scenarios; each scenario is a
        non-empty list or tuple of N values; each value finite and >= 0.
        All scenarios must share one shape.
    U: uncertainty matrix following the same contract as ``T`` and with
        the same K-by-N shape.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    weights: ``None`` for equal weights (``w[k] = 1 / K``), or a list or
        tuple of K finite, non-bool, non-negative numbers; their sum must
        be > 0. Weights are normalized to sum to one.
    z: non-bool, finite int or float >= 0 giving the band multiplier
        (default 1.96).

    Writing ``F(x) = math.fsum(x[k] for k in range(K))`` and
    ``G(x) = math.fsum(x[i] for i in range(N))``::

        M[i] = F(w[k] * T[k][i])
        V[i] = F(w[k] * ((T[k][i] - M[i]) ** 2 + U[k][i] ** 2))
        R[i] = z * sqrt(V[i])
        J[i] = population[i] * beta
        S = G(J[i] * M[i])
        L = G(J[i] * max(M[i] - R[i], 0))
        H = G(J[i] * (M[i] + R[i]))

    Each level counts the thresholds ``<=`` its score (equality included):
    ``level`` from S, ``low_level`` from L and ``high_level`` from H.

    Returns ``(level, S, low_level, high_level, L, H, M, R)`` where level
    is an int, S/L/H are floats and M/R are N-long plain lists of floats.
    Results are not rounded.
    """
    parsed_t = _validate_matrix("T", T, allow_negative=False)
    parsed_u = _validate_matrix("U", U, allow_negative=False)
    if len(parsed_u) != len(parsed_t):
        raise ValueError(
            f"T and U must have the same number of scenarios, "
            f"got {len(parsed_t)} and {len(parsed_u)}"
        )
    n_receptors = len(parsed_t[0])
    for k, row_u in enumerate(parsed_u):
        if len(row_u) != n_receptors:
            raise ValueError(
                f"U[{k}] must have {n_receptors} elements, got {len(row_u)}"
            )

    k_scenarios = len(parsed_t)

    if weights is None:
        w = [1.0 / k_scenarios] * k_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != k_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {k_scenarios}, "
                f"got {len(weights)}"
            )
        validated_weights = []
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
            validated_weights.append(value)
        weight_sum = math.fsum(validated_weights)
        if weight_sum <= 0:
            raise ValueError(f"weights sum must be > 0, got {weight_sum!r}")
        w = [value / weight_sum for value in validated_weights]

    z_value = _validate_scalar("z", z)
    if z_value < 0:
        raise ValueError(f"z must be >= 0, got {z_value!r}")

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)

    mean = [
        math.fsum(w[k] * parsed_t[k][i] for k in range(k_scenarios))
        for i in range(n_receptors)
    ]
    variance = [
        math.fsum(
            w[k]
            * ((parsed_t[k][i] - mean[i]) ** 2 + parsed_u[k][i] ** 2)
            for k in range(k_scenarios)
        )
        for i in range(n_receptors)
    ]
    spread = [z_value * math.sqrt(variance[i]) for i in range(n_receptors)]

    factor = [populations[i] * beta_value for i in range(n_receptors)]
    score = math.fsum(factor[i] * mean[i] for i in range(n_receptors))
    low_score = math.fsum(
        factor[i] * max(mean[i] - spread[i], 0.0) for i in range(n_receptors)
    )
    high_score = math.fsum(
        factor[i] * (mean[i] + spread[i]) for i in range(n_receptors)
    )

    level = sum(1 for t_value in levels_thresholds if t_value <= score)
    low_level = sum(1 for t_value in levels_thresholds if t_value <= low_score)
    high_level = sum(1 for t_value in levels_thresholds if t_value <= high_score)

    return (
        level,
        score,
        low_level,
        high_level,
        low_score,
        high_score,
        mean,
        spread,
    )


def risk_interval(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Weighted shares of scenarios whose population risk band meets levels.

    values: non-empty scenario x receptor (K x N) matrix of values; both
        the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    weights: ``None`` for equal weights (``w[k] = 1 / K``), or a list or
        tuple of K finite, non-bool, non-negative numbers; their sum must
        be > 0. Weights are normalized to sum to one.
    z: non-bool, finite int or float >= 0 giving the band multiplier
        (default 1.96).

    For each scenario ``k`` and receptor ``i``::

        L[k] = fsum(population[i] * beta
                    * max(values[k][i] - z * uncertainty[k][i], 0.0)
                    for i in range(N))
        H[k] = fsum(population[i] * beta
                    * (values[k][i] + z * uncertainty[k][i])
                    for i in range(N))
        pl[j] = fsum(w[k] for k in range(K) if L[k] >= thresholds[j])
        ph[j] = fsum(w[k] for k in range(K) if H[k] >= thresholds[j])

    Returns ``(pl, ph, L, H)`` where ``pl`` and ``ph`` are 3-long plain
    lists of floats (one entry per threshold, in order) and ``L`` and
    ``H`` are K-long plain lists of floats in scenario order. Results are
    not rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for k, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{k}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    k_scenarios = len(val_matrix)

    if weights is None:
        w = [1.0 / k_scenarios] * k_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != k_scenarios:
            raise ValueError(
                f"weights length must equal scenario count {k_scenarios}, "
                f"got {len(weights)}"
            )
        validated_weights = []
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
            validated_weights.append(value)
        weight_sum = math.fsum(validated_weights)
        if weight_sum <= 0:
            raise ValueError(f"weights sum must be > 0, got {weight_sum!r}")
        w = [value / weight_sum for value in validated_weights]

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)
    z_value = _validate_scalar("z", z)
    if z_value < 0:
        raise ValueError(f"z must be >= 0, got {z_value!r}")

    low_scores: list[float] = []
    high_scores: list[float] = []
    for k in range(k_scenarios):
        low_score = math.fsum(
            populations[i] * beta_value
            * max(val_matrix[k][i] - z_value * unc_matrix[k][i], 0.0)
            for i in range(n_receptors)
        )
        high_score = math.fsum(
            populations[i] * beta_value
            * (val_matrix[k][i] + z_value * unc_matrix[k][i])
            for i in range(n_receptors)
        )
        low_scores.append(low_score)
        high_scores.append(high_score)

    pl = [
        math.fsum(w[k] for k in range(k_scenarios) if low_scores[k] >= threshold)
        for threshold in levels_thresholds
    ]
    ph = [
        math.fsum(w[k] for k in range(k_scenarios) if high_scores[k] >= threshold)
        for threshold in levels_thresholds
    ]

    return pl, ph, low_scores, high_scores


def window(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    z: float = 1.96,
    minimum: int = 1,
) -> tuple[list[int], int, int, int]:
    """Locate sustained maximum-level warning windows over time.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    z: non-bool, finite int or float >= 0 giving the confidence
        multiplier (default 1.96).
    minimum: non-bool int >= 1 giving the minimum run length that
        qualifies as a window (default 1).

    For each time ``t`` and receptor ``i``::

        H[t] = fsum(population[i] * beta
                    * (values[t][i] + z * uncertainty[t][i])
                    for i in range(N))
        level[t] = number of thresholds <= H[t]   (0-3)
        x[t] = (level[t] == 3)

    Consecutive ``True`` values of ``x`` form runs (segments) along the
    time axis.

    Returns ``(levels, longest, first_start, qualifying_runs)`` where
    ``levels`` is a K-long plain list of ints in time order; ``longest``
    is the length of the longest run of ``True`` values (0 when there is
    no run); ``first_start`` is the smallest start index among the
    longest runs (-1 when there is no run); ``qualifying_runs`` is the
    number of runs whose length is >= ``minimum``. All three summary
    values are ints and results are not rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for t, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)
    z_value = _validate_scalar("z", z)
    if z_value < 0:
        raise ValueError(f"z must be >= 0, got {z_value!r}")
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            f"minimum must be an int, got {type(minimum).__name__}"
        )
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    levels: list[int] = []
    for t in range(len(val_matrix)):
        high_score = math.fsum(
            populations[i] * beta_value
            * (val_matrix[t][i] + z_value * unc_matrix[t][i])
            for i in range(n_receptors)
        )
        levels.append(
            sum(1 for t_value in levels_thresholds if t_value <= high_score)
        )

    longest = 0
    first_start = -1
    qualifying_runs = 0
    run_length = 0
    run_start = -1
    for t, level in enumerate(levels):
        if level == 3:
            if run_length == 0:
                run_start = t
            run_length += 1
        else:
            if run_length > 0:
                if run_length >= minimum:
                    qualifying_runs += 1
                if run_length > longest:
                    longest = run_length
                    first_start = run_start
            run_length = 0
            run_start = -1
    if run_length > 0:
        if run_length >= minimum:
            qualifying_runs += 1
        if run_length > longest:
            longest = run_length
            first_start = run_start

    return levels, longest, first_start, qualifying_runs


def transition(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    z: float = 1.96,
) -> tuple[list[int], int, int, int, int]:
    """Count warning-level transitions over time under upper uncertainty.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    z: non-bool, finite int or float >= 0 giving the confidence
        multiplier (default 1.96).

    For each time ``t`` and receptor ``i``::

        H[t] = fsum(population[i] * beta
                    * (values[t][i] + z * uncertainty[t][i])
                    for i in range(N))
        level[t] = number of thresholds <= H[t]   (0-3)

    Returns ``(levels, up, down, peak, peak_time)`` where ``levels`` is a
    K-long plain list of ints in time order; ``up`` counts the times
    ``t >= 1`` with ``level[t] > level[t - 1]`` and ``down`` counts those
    with ``level[t] < level[t - 1]`` (both 0 when K == 1); ``peak`` is
    ``max(levels)`` and ``peak_time`` is the smallest index at which
    ``peak`` first occurs. All summary values are ints and results are not
    rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for t, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)
    z_value = _validate_scalar("z", z)
    if z_value < 0:
        raise ValueError(f"z must be >= 0, got {z_value!r}")

    levels: list[int] = []
    for t in range(len(val_matrix)):
        high_score = math.fsum(
            populations[i] * beta_value
            * (val_matrix[t][i] + z_value * unc_matrix[t][i])
            for i in range(n_receptors)
        )
        levels.append(
            sum(1 for t_value in levels_thresholds if t_value <= high_score)
        )

    up = sum(1 for t in range(1, len(levels)) if levels[t] > levels[t - 1])
    down = sum(1 for t in range(1, len(levels)) if levels[t] < levels[t - 1])
    peak = max(levels)
    peak_time = levels.index(peak)

    return levels, up, down, peak, peak_time


def exceedance_probability(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[list[list[float]], list[float]]:
    """Gaussian exceedance probabilities of warning thresholds over time.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.

    For each time ``t`` and receptor ``i`` the population-weighted impact
    is modeled as a Gaussian with::

        mu[t] = fsum(population[i] * beta * values[t][i] for i in range(N))
        sigma[t] = hypot(*(population[i] * beta * uncertainty[t][i]
                           for i in range(N)))

    and for each threshold index ``j`` in 0..2::

        p[t][j] = 0.5 * erfc((thresholds[j] - mu[t])
                             / (sigma[t] * sqrt(2)))   if sigma[t] > 0
        p[t][j] = 1.0 if thresholds[j] <= mu[t] else 0.0
                                                      if sigma[t] == 0

    Returns ``(p, expected)`` where ``p`` is a K x 3 plain list of lists
    of floats in time order and ``expected`` is a K-long plain list of
    floats with ``expected[t] = fsum(p[t])``. Results are not rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for t, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)

    p: list[list[float]] = []
    expected: list[float] = []
    for t in range(len(val_matrix)):
        mu = math.fsum(
            populations[i] * beta_value * val_matrix[t][i]
            for i in range(n_receptors)
        )
        sigma = math.hypot(
            *(populations[i] * beta_value * unc_matrix[t][i]
              for i in range(n_receptors))
        )
        row: list[float] = []
        for j in range(3):
            if sigma > 0:
                row.append(
                    0.5
                    * math.erfc(
                        (levels_thresholds[j] - mu) / (sigma * math.sqrt(2))
                    )
                )
            else:
                row.append(1.0 if levels_thresholds[j] <= mu else 0.0)
        p.append(row)
        expected.append(math.fsum(row))

    return p, expected


def level_probability(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Probability distribution over the four warning levels over time.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.

    For each time ``t`` and receptor ``i`` the population-weighted impact
    is modeled as a Gaussian with::

        mu[t] = fsum(population[i] * beta * values[t][i] for i in range(N))
        sigma[t] = hypot(*(population[i] * beta * uncertainty[t][i]
                           for i in range(N)))

    and for each threshold index ``j`` in 0..2::

        q[t][j] = 0.5 * erfc((thresholds[j] - mu[t])
                             / (sigma[t] * sqrt(2)))   if sigma[t] > 0
        q[t][j] = 1.0 if thresholds[j] <= mu[t] else 0.0
                                                      if sigma[t] == 0

    Writing ``qj = q[t][j]``, the four level probabilities are::

        probabilities[t] = [1 - q0, q0 - q1, q1 - q2, q2]

    Returns ``(probabilities, expected_level, alert_probability)`` where
    ``probabilities`` is a K x 4 plain list of lists of floats in time
    order, ``expected_level`` is a K-long plain list of floats with
    ``expected_level[t] = fsum(j * probabilities[t][j] for j in 0..3)``
    and ``alert_probability`` is a K-long plain list of floats with
    ``alert_probability[t] = probabilities[t][3]``. Results are not
    rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for t, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels_thresholds = _validate_thresholds(thresholds)

    probabilities: list[list[float]] = []
    expected_level: list[float] = []
    alert_probability: list[float] = []
    for t in range(len(val_matrix)):
        mu = math.fsum(
            populations[i] * beta_value * val_matrix[t][i]
            for i in range(n_receptors)
        )
        sigma = math.hypot(
            *(populations[i] * beta_value * unc_matrix[t][i]
              for i in range(n_receptors))
        )
        q: list[float] = []
        for j in range(3):
            if sigma > 0:
                q.append(
                    0.5
                    * math.erfc(
                        (levels_thresholds[j] - mu) / (sigma * math.sqrt(2))
                    )
                )
            else:
                q.append(1.0 if levels_thresholds[j] <= mu else 0.0)
        row = [1.0 - q[0], q[0] - q[1], q[1] - q[2], q[2]]
        probabilities.append(row)
        expected_level.append(math.fsum(j * row[j] for j in range(4)))
        alert_probability.append(row[3])

    return probabilities, expected_level, alert_probability


def _validate_quantiles(quantiles: object) -> list[float]:
    if not isinstance(quantiles, (list, tuple)):
        raise TypeError(
            f"quantiles must be a list or tuple, got {type(quantiles).__name__}"
        )
    if len(quantiles) == 0:
        raise ValueError("quantiles must not be empty")
    validated = []
    for m, item in enumerate(quantiles):
        if not _is_number(item):
            raise TypeError(
                f"quantiles[{m}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"quantiles[{m}]", value)
        if value < 0 or value > 1:
            raise ValueError(
                f"quantiles[{m}] must be in [0, 1], got {value!r}"
            )
        if m > 0 and value < validated[m - 1]:
            raise ValueError(
                f"quantiles must be non-decreasing, got "
                f"quantiles[{m - 1}]={validated[m - 1]!r} and "
                f"quantiles[{m}]={value!r}"
            )
        validated.append(value)
    return validated


def _validate_weights(weights: object, k_scenarios: int) -> list[float]:
    if weights is None:
        return [1.0 / k_scenarios] * k_scenarios
    if not isinstance(weights, (list, tuple)):
        raise TypeError(
            f"weights must be a list or tuple, got {type(weights).__name__}"
        )
    if len(weights) != k_scenarios:
        raise ValueError(
            f"weights length must equal scenario count {k_scenarios}, "
            f"got {len(weights)}"
        )
    validated = []
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
        validated.append(value)
    weight_sum = math.fsum(validated)
    if weight_sum <= 0:
        raise ValueError(f"weights sum must be > 0, got {weight_sum!r}")
    return [value / weight_sum for value in validated]


def _weighted_quantile(
    scores: list[float], weights: list[float], level: float
) -> float:
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    ordered = [scores[k] for k in order]
    if level == 0:
        return ordered[0]
    for j in range(len(ordered)):
        cumulative = math.fsum(weights[order[i]] for i in range(j + 1))
        if cumulative >= level:
            return ordered[j]
    return ordered[-1]


def quantile(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    quantiles: list[float] | tuple[float, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float]]:
    """Weighted quantiles of scenario means and uncertainty bands.

    values: non-empty scenario x receptor (K x N) matrix of values; both
        the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value finite
        and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    quantiles: non-empty list or tuple of finite, non-bool numbers in
        [0, 1] in non-decreasing order.
    weights: ``None`` for equal weights (``w[k] = 1 / K``), or a list or
        tuple of K finite, non-bool, non-negative numbers; their sum must
        be > 0. Weights are normalized to sum to one.
    z: non-bool, finite int or float >= 0 giving the band multiplier
        (default 1.96).

    For each scenario ``k`` and receptor ``i``::

        mu[k] = fsum(population[i] * beta * values[k][i] for i in range(N))
        sigma[k] = hypot(*(population[i] * beta * uncertainty[k][i]
                          for i in range(N)))
        l[k] = max(0.0, mu[k] - z * sigma[k])
        h[k] = mu[k] + z * sigma[k]

    Each of ``mu``, ``l`` and ``h`` is sorted ascending by ``(value, k)``
    and defines a weighted empirical distribution with the normalized
    weights. For a requested level ``q``, the quantile is the first item
    whose cumulative normalized weight is >= q; for ``q == 0`` it is the
    first item.

    Returns ``(Q, L, H)``, three M-long plain lists of floats in the
    order of ``quantiles``, where ``Q`` holds the quantiles of ``mu``,
    ``L`` of ``l`` and ``H`` of ``h``. Results are not rounded.
    """
    val_matrix = _validate_matrix("values", values, allow_negative=False)
    unc_matrix = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    if len(unc_matrix) != len(val_matrix):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(val_matrix)} and {len(unc_matrix)}"
        )
    n_receptors = len(val_matrix[0])
    for k, row in enumerate(unc_matrix):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{k}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    populations = _validate_vector("population", population, n_receptors)
    beta_value = _validate_scalar("beta", beta)
    if beta_value < 0:
        raise ValueError(f"beta must be >= 0, got {beta_value!r}")
    levels = _validate_quantiles(quantiles)
    w = _validate_weights(weights, len(val_matrix))
    z_value = _validate_scalar("z", z)
    if z_value < 0:
        raise ValueError(f"z must be >= 0, got {z_value!r}")

    mu: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for k in range(len(val_matrix)):
        mean = math.fsum(
            populations[i] * beta_value * val_matrix[k][i]
            for i in range(n_receptors)
        )
        sigma = math.hypot(
            *(populations[i] * beta_value * unc_matrix[k][i]
              for i in range(n_receptors))
        )
        mu.append(mean)
        low.append(max(0.0, mean - z_value * sigma))
        high.append(mean + z_value * sigma)

    q_values = [_weighted_quantile(mu, w, level) for level in levels]
    l_values = [_weighted_quantile(low, w, level) for level in levels]
    h_values = [_weighted_quantile(high, w, level) for level in levels]

    return q_values, l_values, h_values


def alert_run(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    minimum: int = 1,
) -> tuple[list[float], list[float], float]:
    """Probability of completing a run of consecutive top-level alerts.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    minimum: non-bool int >= 1 giving the required run length of
        consecutive top-level alerts (default 1).

    ``q`` is exactly the third item (``alert_probability``) returned by
    ``level_probability(values, uncertainty, population, beta,
    thresholds)``, i.e. for each time ``t`` the probability that the
    highest warning level is reached. Times are treated as independent.
    Writing ``m = minimum``, the state ``s`` starts as
    ``[1] + [0] * (m - 1)`` and, for each time ``t``::

        first[t] = s[m - 1] * q[t]
        new[0] = fsum(s[r] * (1 - q[t]) for r in 0..m-1)
        new[r + 1] = s[r] * q[t]          for r in 0..m-2
        s = new

    Returns ``(q, first, p_any)`` where ``q`` and ``first`` are K-long
    plain lists of floats in time order — ``first[t]`` is the probability
    that a run of ``m`` consecutive top-level alerts is first completed
    exactly at time ``t`` — and ``p_any`` is the float
    ``1 - fsum(s)``, the probability that such a run is completed at any
    time within the horizon. Results are not rounded.
    """
    _, _, q = level_probability(values, uncertainty, population, beta, thresholds)

    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            f"minimum must be an int, got {type(minimum).__name__}"
        )
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    m = minimum
    s = [1.0] + [0.0] * (m - 1)
    first: list[float] = []
    for q_t in q:
        first.append(s[m - 1] * q_t)
        new = [math.fsum(s[r] * (1.0 - q_t) for r in range(m))]
        for r in range(m - 1):
            new.append(s[r] * q_t)
        s = new
    p_any = 1.0 - math.fsum(s)

    return q, first, p_any


def alert_run_distribution(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    minimum: int = 1,
) -> tuple[list[float], list[float], float]:
    """Distribution of completed runs of consecutive top-level alerts.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    minimum: non-bool int >= 1 giving the required run length of
        consecutive top-level alerts (default 1).

    ``q`` is exactly the third item (``alert_probability``) returned by
    ``level_probability(values, uncertainty, population, beta,
    thresholds)``, i.e. for each time ``t`` the probability that the
    highest warning level is reached. The alert at time ``t`` is treated
    as an independent Bernoulli(``q[t]``) event. Writing ``m = minimum``,
    the state ``(r, c)`` tracks the current run length ``r`` of
    consecutive alerts truncated to ``m`` and the count ``c`` of
    completed qualifying runs; the initial state is ``P(0, 0) = 1``. For
    each time ``t``::

        alert (probability q[t]):
            (r, c) -> (r + 1, c)      if r < m
            (r, c) -> (r, c)          if r == m
            and c increases by one exactly when r rises from m - 1 to m
        no alert (probability 1 - q[t]):
            (r, c) -> (0, c)

    Returns ``(q, distribution, expected_runs)`` where ``q`` is a K-long
    plain list of floats in time order, ``distribution`` is a (K + 1)-long
    plain list of floats with ``distribution[c]`` the probability of
    completing exactly ``c`` qualifying runs within the horizon, and
    ``expected_runs`` is the float ``fsum(c * distribution[c] for c in
    range(K + 1))``. Results are not rounded.
    """
    _, _, q = level_probability(values, uncertainty, population, beta, thresholds)

    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            f"minimum must be an int, got {type(minimum).__name__}"
        )
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    m = minimum
    k_times = len(q)
    state = {(0, 0): 1.0}
    for q_t in q:
        new: dict[tuple[int, int], float] = {}
        for (r, c), p in state.items():
            key = (0, c)
            new[key] = new.get(key, 0.0) + p * (1.0 - q_t)
            if r < m:
                key = (r + 1, c + 1 if r + 1 == m else c)
            else:
                key = (r, c)
            new[key] = new.get(key, 0.0) + p * q_t
        state = new

    distribution = [0.0] * (k_times + 1)
    for (_, c), p in state.items():
        distribution[c] += p
    expected_runs = math.fsum(
        c * distribution[c] for c in range(k_times + 1)
    )

    return q, distribution, expected_runs


def alert_run_profile(
    values: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    population: list[float] | tuple[float, ...],
    beta: float,
    thresholds: list[float] | tuple[float, ...],
    minimum: int = 1,
) -> tuple[list[float], list[float], float, list[float], list[float]]:
    """Timing profile of completed runs of consecutive top-level alerts.

    values: non-empty time x receptor (K x N) matrix of forecast values;
        both the outer container and each row must be a list or tuple,
        rows must be non-empty and share one receptor count; each value
        finite and >= 0.
    uncertainty: K x N matrix with the same shape and constraints as
        ``values``.
    population: list or tuple of N finite, non-negative receptor
        populations.
    beta: finite, non-negative scaling factor.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative warning thresholds.
    minimum: non-bool int >= 1 giving the required run length of
        consecutive top-level alerts (default 1).

    ``q`` is exactly the third item (``alert_probability``) returned by
    ``level_probability(values, uncertainty, population, beta,
    thresholds)``, i.e. for each time ``t`` the probability that the
    highest warning level is reached. The alert at time ``t`` is treated
    as an independent Bernoulli(``q[t]``) event. Writing ``m = minimum``,
    the state ``(r, c)`` tracks the current run length ``r`` of
    consecutive alerts truncated to ``m`` and the count ``c`` of
    completed qualifying runs; the initial state is ``P(0, 0) = 1``. For
    each time ``t``::

        alert (probability q[t]):
            (r, c) -> (r + 1, c + (1 if r + 1 == m else 0))   if r < m
            (r, c) -> (r, c)                                   if r == m
        no alert (probability 1 - q[t]):
            (r, c) -> (0, c)

    Contributions to each destination state are merged with
    ``math.fsum``, visiting the source states in ascending ``(r, c)``
    order. ``first[t]`` is the probability that a qualifying run is first
    completed exactly at time ``t`` (mass transitioning from ``r = m - 1``
    on an alert) and ``cumulative_any[t] = fsum(first[: t + 1])`` is the
    probability that at least one qualifying run has been completed by
    time ``t``.

    Returns ``(q, distribution, expected_runs, first, cumulative_any)``
    where ``q``, ``first`` and ``cumulative_any`` are K-long plain lists
    of floats in time order; ``distribution`` is a (K + 1)-long plain
    list of floats with ``distribution[c]`` the probability of completing
    exactly ``c`` qualifying runs within the horizon; and
    ``expected_runs`` is the float ``fsum(c * distribution[c] for c in
    range(K + 1))``. Results are not rounded.
    """
    _, _, q = level_probability(values, uncertainty, population, beta, thresholds)

    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise TypeError(
            f"minimum must be an int, got {type(minimum).__name__}"
        )
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum}")

    m = minimum
    k_times = len(q)
    state: dict[tuple[int, int], float] = {(0, 0): 1.0}
    first: list[float] = []
    for q_t in q:
        contributions: dict[tuple[int, int], list[float]] = {}
        first_terms: list[float] = []
        for (r, c), p in sorted(state.items()):
            key = (0, c)
            contributions.setdefault(key, []).append(p * (1.0 - q_t))
            if r < m:
                completes = r + 1 == m
                key = (r + 1, c + 1 if completes else c)
                if completes and c == 0:
                    first_terms.append(p * q_t)
            else:
                key = (r, c)
            contributions.setdefault(key, []).append(p * q_t)
        state = {
            key: math.fsum(parts) for key, parts in contributions.items()
        }
        first.append(math.fsum(first_terms))

    cumulative_any = [
        math.fsum(first[: t + 1]) for t in range(k_times)
    ]

    distribution = [
        math.fsum(p for (_, c), p in sorted(state.items()) if c == c_value)
        for c_value in range(k_times + 1)
    ]
    expected_runs = math.fsum(
        c * distribution[c] for c in range(k_times + 1)
    )

    return q, distribution, expected_runs, first, cumulative_any
