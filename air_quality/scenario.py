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


def _validate_matrix(name: str, values: object) -> tuple[list[list[float]], int]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    rows: list[list[float]] = []
    width: int | None = None
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
                f"{name} rows must all have the same width, got row {k} of length "
                f"{len(row)} expected {width}"
            )
        validated_row: list[float] = []
        for i, item in enumerate(row):
            if not _is_number(item):
                raise TypeError(
                    f"{name}[{k}][{i}] must be an int or float, got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"{name}[{k}][{i}]", value)
            if value < 0:
                raise ValueError(f"{name}[{k}][{i}] must be >= 0, got {value!r}")
            validated_row.append(value)
        rows.append(validated_row)
    return rows, width  # type: ignore[return-value]



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
    """Aggregate scenario totals with a weighted mean and uncertainty band.

    T: non-empty list or tuple of K scenarios; each scenario is a
        non-empty list or tuple of total concentrations, one per
        receptor, in receptor order; each value must be finite and >= 0.
    U: non-empty list or tuple of K scenarios with the same shape as
        ``T``; each value is the standard uncertainty of the
        corresponding total and must be finite and >= 0.
    weights: ``None`` for equal weights, or a list or tuple of K
        non-negative finite weights; the weights must have a positive
        sum.
    z: non-negative finite int or float multiplier applied to the
        aggregated standard spread (e.g. 1.96 for a 95% band).

    Writing ``w[k] = 1 / K`` for equal weights and
    ``w[k] = weights[k] / sum(weights)`` otherwise:

    * ``mean[i] = sum_k w[k] * T[k][i]``,
    * ``v[i] = sum_k w[k] * ((T[k][i] - mean[i]) ** 2 + U[k][i] ** 2)``,
    * ``spread[i] = z * sqrt(v[i])``,
    * ``lower[i] = max(0.0, mean[i] - spread[i])``,
    * ``upper[i] = mean[i] + spread[i]``.

    Returns ``(mean, spread, lower, upper)``; all four are plain lists
    of floats in receptor order and are not rounded.
    """
    t_rows, t_width = _validate_matrix("T", T)
    u_rows, u_width = _validate_matrix("U", U)
    if len(t_rows) != len(u_rows):
        raise ValueError(
            f"T and U must have the same number of scenarios, got {len(t_rows)} "
            f"and {len(u_rows)}"
        )
    if t_width != u_width:
        raise ValueError(
            f"T and U rows must have the same width, got {t_width} and {u_width}"
        )

    k_scenarios = len(t_rows)
    if weights is None:
        w = [1.0 / k_scenarios] * k_scenarios
    else:
        if not isinstance(weights, (list, tuple)):
            raise TypeError(
                f"weights must be a list or tuple, got {type(weights).__name__}"
            )
        if len(weights) != k_scenarios:
            raise ValueError(
                f"weights length must equal number of scenarios {k_scenarios}, "
                f"got {len(weights)}"
            )
        raw_weights = []
        for index, item in enumerate(weights):
            if not _is_number(item):
                raise TypeError(
                    f"weights[{index}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"weights[{index}]", value)
            if value < 0:
                raise ValueError(f"weights[{index}] must be >= 0, got {value!r}")
            raw_weights.append(value)
        total_weight = math.fsum(raw_weights)
        if total_weight <= 0:
            raise ValueError(f"weights must sum to a value > 0, got {total_weight!r}")
        w = [value / total_weight for value in raw_weights]

    if not _is_number(z):
        raise TypeError(f"z must be an int or float, got {type(z).__name__}")
    z = float(z)
    _check_finite("z", z)
    if z < 0:
        raise ValueError(f"z must be >= 0, got {z!r}")

    n_receptors = t_width

    def weighted_sum(i: int) -> float:
        return math.fsum(w[k] * t_rows[k][i] for k in range(k_scenarios))

    mean = [weighted_sum(i) for i in range(n_receptors)]
    spread = [
        z
        * math.sqrt(
            math.fsum(
                w[k]
                * ((t_rows[k][i] - mean[i]) ** 2 + u_rows[k][i] ** 2)
                for k in range(k_scenarios)
            )
        )
        for i in range(n_receptors)
    ]
    lower = [max(0.0, mean[i] - spread[i]) for i in range(n_receptors)]
    upper = [mean[i] + spread[i] for i in range(n_receptors)]

    return mean, spread, lower, upper
