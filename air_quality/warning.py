"""Air quality warning level assessment relative to a baseline scenario."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number, _validate_scalar

__all__ = ["assess", "forecast"]


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


def _validate_population(population: object, length: int) -> list[float]:
    if not isinstance(population, (list, tuple)):
        raise TypeError(
            f"population must be a list or tuple, got {type(population).__name__}"
        )
    if len(population) == 0:
        raise ValueError("population must not be empty")
    if len(population) != length:
        raise ValueError(
            f"population must have {length} elements, got {len(population)}"
        )
    validated = []
    for i, item in enumerate(population):
        if not _is_number(item):
            raise TypeError(
                f"population[{i}] must be an int or float, "
                f"got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"population[{i}]", value)
        if value < 0:
            raise ValueError(f"population[{i}] must be >= 0, got {value!r}")
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
    """Forecast warning levels over time for a population-weighted scenario.

    values: non-empty time x receptor matrix (K x N, K, N >= 1) of forecast
        values; both the outer container and each row must be a list or
        tuple, rows must be non-empty and share one receptor count; each
        value finite and >= 0.
    uncertainty: matrix of forecast uncertainties with the same shape and
        constraints as ``values``.
    population: list or tuple of N finite, non-negative population weights.
    beta: finite, non-negative scaling factor (non-bool int or float).
    thresholds: list or tuple of 3 strictly increasing, finite, non-negative
        warning thresholds.
    Returns ``(levels, evidence, scores)`` where, for each time ``t`` and
    receptor ``i``, with ``h[i] = population[i] * beta * values[t][i]`` and
    ``w[i] = population[i] * beta * uncertainty[t][i]``:

    * ``I = fsum(max(h[i], 0.0) for i in range(N))``,
    * ``V = hypot(*w)``, equal to ``sqrt(fsum(x ** 2 for x in w))`` but
      without intermediate overflow,
    * ``scores[t] = I + V``,
    * ``levels[t]`` is the count of thresholds ``<= scores[t]`` (0-3),
    * ``evidence[t] = (I, V)``.

    All results are plain lists in time order and are not rounded.
    """
    forecast_values = _validate_matrix("values", values, allow_negative=False)
    uncertainties = _validate_matrix("uncertainty", uncertainty, allow_negative=False)

    if len(uncertainties) != len(forecast_values):
        raise ValueError(
            f"values and uncertainty must have the same number of rows, "
            f"got {len(forecast_values)} and {len(uncertainties)}"
        )
    n_receptors = len(forecast_values[0])
    for t, row in enumerate(uncertainties):
        if len(row) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(row)}"
            )

    weights = _validate_population(population, n_receptors)

    factor = _validate_scalar("beta", beta)
    if factor < 0:
        raise ValueError(f"beta must be >= 0, got {factor!r}")

    levels_thresholds = _validate_thresholds(thresholds)

    levels: list[int] = []
    evidence: list[tuple[float, float]] = []
    scores: list[float] = []
    for t in range(len(forecast_values)):
        h = [weights[i] * factor * forecast_values[t][i] for i in range(n_receptors)]
        w = [weights[i] * factor * uncertainties[t][i] for i in range(n_receptors)]
        i_total = math.fsum(max(h[i], 0.0) for i in range(n_receptors))
        v_total = math.hypot(*w)
        score = i_total + v_total
        level = sum(1 for threshold in levels_thresholds if threshold <= score)
        levels.append(level)
        evidence.append((i_total, v_total))
        scores.append(score)

    return levels, evidence, scores
