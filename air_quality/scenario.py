"""Scenario comparison for source attribution runs."""

from __future__ import annotations

from .attribution import attribute

__all__ = ["predict"]


def _validate_scenarios(name: str, values: object) -> list[object]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    for index, item in enumerate(values):
        if not isinstance(item, (list, tuple)):
            raise TypeError(
                f"{name}[{index}] must be a list or tuple, got {type(item).__name__}"
            )
    return list(values)


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
    rates: non-empty list or tuple of scenarios; each scenario is a list or
        tuple of emission rates validated exactly as ``rates`` in
        :func:`attribute`.
    sigma: non-empty list or tuple of scenarios with the same number of
        scenarios as ``rates``; each scenario is a list or tuple validated
        exactly as ``rate_uncertainty`` in :func:`attribute`.
    baseline: index of the baseline scenario, an int (not bool) with
        ``0 <= baseline < number of scenarios``.

    Runs :func:`attribute` once per scenario k with ``rates[k]`` and
    ``sigma[k]``, giving contribution matrix Ck, total concentrations Tk and
    uncertainties Uk. Returns ``(T, D, G, U)`` where:

    * ``T[k][i] = Tk[i]`` (scenario x receptor),
    * ``D[k][i] = Tk[i] - T_baseline[i]`` (0.0 for the baseline scenario),
    * ``G[k][i][j] = Ck[i][j] - C_baseline[i][j]`` (0.0 for the baseline
      scenario),
    * ``U[k][i] = Uk[i]``.

    All results are plain lists of floats in input order and are not rounded.
    """
    scenarios_rates = _validate_scenarios("rates", rates)
    scenarios_sigma = _validate_scenarios("sigma", sigma)
    if len(scenarios_rates) != len(scenarios_sigma):
        raise ValueError(
            f"rates and sigma must have the same number of scenarios, "
            f"got {len(scenarios_rates)} and {len(scenarios_sigma)}"
        )
    count = len(scenarios_rates)
    if isinstance(baseline, bool) or not isinstance(baseline, int):
        raise TypeError(
            f"baseline must be an int, got {type(baseline).__name__}"
        )
    if not 0 <= baseline < count:
        raise ValueError(
            f"baseline must be in [0, {count}), got {baseline!r}"
        )

    contributions = []
    totals = []
    uncertainties = []
    for k in range(count):
        ck, _fk, _sk, tk, uk = attribute(
            sources, receptors, scenarios_rates[k], scenarios_sigma[k],
            u, direction, sy, sz,
        )
        contributions.append(ck)
        totals.append(tk)
        uncertainties.append(uk)

    base_c = contributions[baseline]
    base_t = totals[baseline]

    T = totals
    D = []
    G = []
    for k in range(count):
        if k == baseline:
            D.append([0.0 for _ in totals[k]])
            G.append([[0.0 for _ in row] for row in contributions[k]])
        else:
            D.append([totals[k][i] - base_t[i] for i in range(len(totals[k]))])
            G.append([
                [contributions[k][i][j] - base_c[i][j] for j in range(len(contributions[k][i]))]
                for i in range(len(contributions[k]))
            ])

    return T, D, G, uncertainties
