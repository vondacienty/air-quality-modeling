"""Scenario comparison for source attribution runs."""

from __future__ import annotations

import math

from .attribution import attribute
from .gaussian import _check_finite, _is_number

__all__ = ["predict", "aggregate"]


def _validate_scenarios(name: str, values: object) -> list | tuple:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    return values


def _validate_matrix(name: str, values: object) -> list[list[float]]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    validated = []
    width = None
    for k, row in enumerate(values):
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
        checked_row = []
        for i, item in enumerate(row):
            if not _is_number(item):
                raise TypeError(
                    f"{name}[{k}][{i}] must be an int or float, got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"{name}[{k}][{i}]", value)
            if value < 0:
                raise ValueError(f"{name}[{k}][{i}] must be >= 0, got {value!r}")
            checked_row.append(value)
        validated.append(checked_row)
    return validated


def predict(
    sources: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    receptors: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    rates: list[list[float]] | tuple[tuple[float, ...], ...],
    sigma: list[list[float]] | tuple[tuple[float, ...], ...],
    baseline: int,
    u: float,
    direction: float,
    sy: float,
    sz: float,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[list[float]]],
    list[list[float]],
]:
    """Compare attribution results across emission scenarios.

    sources, receptors, u, direction, sy, sz: as in
        :func:`air_quality.attribution.attribute`.
    rates: non-empty list or tuple of scenarios; each scenario is a list
        or tuple of emission rates following the ``rates`` contract of
        :func:`attribute`.
    sigma: non-empty list or tuple of scenarios, one per scenario in
        ``rates``; each scenario follows the ``rate_uncertainty``
        contract of :func:`attribute`.
    baseline: index of the reference scenario; a non-bool int with
        ``0 <= baseline < len(rates)``.

    Runs :func:`attribute` once per scenario and returns ``(T, D, G, U)``
    where, writing the per-scenario contributions, totals and
    uncertainties as Ck, Tk, Uk and the baseline scenario as b:

    * ``T[k][i] = Tk[i]`` is the total concentration at receptor i,
    * ``D[k][i] = Tk[i] - Tb[i]`` is the change in total concentration
      relative to the baseline (0.0 for the baseline scenario),
    * ``G[k][i][j] = Ck[i][j] - Cb[i][j]`` is the change in the
      contribution of source j at receptor i relative to the baseline
      (0.0 for the baseline scenario),
    * ``U[k][i] = Uk[i]`` is the propagated standard uncertainty.

    All results are plain lists of floats in input order and are not
    rounded.
    """
    rates = _validate_scenarios("rates", rates)
    sigma = _validate_scenarios("sigma", sigma)
    if len(rates) != len(sigma):
        raise ValueError(
            f"rates and sigma must have the same number of scenarios, "
            f"got {len(rates)} and {len(sigma)}"
        )
    if isinstance(baseline, bool) or not isinstance(baseline, int):
        raise TypeError(
            f"baseline must be an int, got {type(baseline).__name__}"
        )
    if not 0 <= baseline < len(rates):
        raise ValueError(
            f"baseline must be in [0, {len(rates)}), got {baseline!r}"
        )

    contributions = []
    totals = []
    uncertainties = []
    for scenario_rates, scenario_sigma in zip(rates, sigma):
        ck, _, _, tk, uk = attribute(
            sources, receptors, scenario_rates, scenario_sigma, u, direction, sy, sz
        )
        contributions.append(ck)
        totals.append(tk)
        uncertainties.append(uk)

    base_c = contributions[baseline]
    base_t = totals[baseline]

    T = [list(tk) for tk in totals]
    D = [
        [0.0] * len(tk) if k == baseline else [tk[i] - base_t[i] for i in range(len(tk))]
        for k, tk in enumerate(totals)
    ]
    G = [
        [[0.0] * len(ck[i]) for i in range(len(ck))]
        if k == baseline
        else [
            [ck[i][j] - base_c[i][j] for j in range(len(ck[i]))]
            for i in range(len(ck))
        ]
        for k, ck in enumerate(contributions)
    ]
    U = [list(uk) for uk in uncertainties]

    return T, D, G, U


def aggregate(
    T: list[list[float]] | tuple[tuple[float, ...], ...],
    U: list[list[float]] | tuple[tuple[float, ...], ...],
    weights: list[float] | tuple[float, ...] | None = None,
    z: float = 1.96,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Aggregate scenario totals into a weighted mean and uncertainty band.

    T: non-empty list or tuple of K scenarios; each scenario is a
        non-empty list or tuple of total concentrations at every
        receptor; each value must be a finite, non-bool int or float
        with value >= 0. All scenarios must share one shape.
    U: uncertainty matrix following the same contract as ``T`` and with
        the same K-by-receptor shape.
    weights: ``None`` for equal weights (``w[k] = 1 / K``), or a list or
        tuple of K finite, non-bool, non-negative numbers; their sum
        must be > 0.
    z: non-bool, finite int or float >= 0 giving the band multiplier.

    Returns ``(mean, spread, lower, upper)`` where, writing the weight
    of scenario k as w[k] and ``S(e) = math.fsum(e[k] for k in range(K))``:

    * ``mean[i] = S(w[k] * T[k][i])``,
    * ``v[i] = S(w[k] * ((T[k][i] - mean[i]) ** 2 + U[k][i] ** 2))``,
    * ``spread[i] = z * sqrt(v[i])``,
    * ``lower[i] = max(0.0, mean[i] - spread[i])``,
    * ``upper[i] = mean[i] + spread[i]``.

    All four results are plain lists of floats in receptor order and are
    not rounded.
    """
    parsed_t = _validate_matrix("T", T)
    parsed_u = _validate_matrix("U", U)
    if len(parsed_u) != len(parsed_t):
        raise ValueError(
            f"T and U must have the same number of scenarios, got {len(parsed_t)} and {len(parsed_u)}"
        )
    for k, (row_t, row_u) in enumerate(zip(parsed_t, parsed_u)):
        if len(row_u) != len(row_t):
            raise ValueError(
                f"U[{k}] must have {len(row_t)} elements, got {len(row_u)}"
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
                f"weights length must equal scenario count {k_scenarios}, got {len(weights)}"
            )
        validated_weights = []
        for k, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{k}] must be an int or float, got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{k}]", value)
            if value < 0:
                raise ValueError(f"weights[{k}] must be >= 0, got {value!r}")
            validated_weights.append(value)
        s = math.fsum(validated_weights)
        if s <= 0:
            raise ValueError(f"weights sum must be > 0, got {s!r}")
        w = [value / s for value in validated_weights]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    n_receptors = len(parsed_t[0])
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
    spread = [z * math.sqrt(variance[i]) for i in range(n_receptors)]
    lower = [
        max(0.0, mean[i] - spread[i]) for i in range(n_receptors)
    ]
    upper = [mean[i] + spread[i] for i in range(n_receptors)]

    return mean, spread, lower, upper
