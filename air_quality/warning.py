"""Air quality warning level assessment relative to a baseline scenario."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number, _validate_scalar

__all__ = ["assess", "forecast", "forecast_interval", "aggregate", "risk_interval"]


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
