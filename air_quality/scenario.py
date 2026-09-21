"""Scenario comparison for source attribution runs."""

from __future__ import annotations

from .attribution import attribute

__all__ = ["predict"]


def _validate_scenarios(name: str, values: object) -> list | tuple:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    return values


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
