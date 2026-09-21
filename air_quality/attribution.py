"""Source attribution: apportion receptor concentrations among sources."""

from __future__ import annotations

import math

from .gaussian import (
    _check_finite,
    _is_number,
    _validate_records,
    _validate_scalar,
    simulate,
)

__all__ = ["attribute"]


def _validate_nonneg_sequence(
    name: str,
    values: object,
    expected_length: int,
) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(values).__name__}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} length must equal sources length {expected_length}, "
            f"got {len(values)}"
        )
    validated = []
    for index, item in enumerate(values):
        if not _is_number(item):
            raise TypeError(
                f"{name}[{index}] must be an int or float, got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"{name}[{index}]", value)
        if value < 0:
            raise ValueError(f"{name}[{index}] must be >= 0, got {value!r}")
        validated.append(value)
    return validated


def attribute(
    sources: list[tuple[float, float, float]],
    receptors: list[tuple[float, float, float]],
    rates: list[float] | tuple[float, ...],
    rate_uncertainty: list[float] | tuple[float, ...],
    u: float,
    direction: float,
    sy: float,
    sz: float,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[float],
    list[float],
    list[float],
]:
    """Attribute receptor concentrations to individual sources.

    Builds the unit-rate response matrix A from :func:`gaussian.simulate`
    (each source released at q = 1) and combines it with the source rates.

    sources: non-empty sequence of (x, y, h).
    receptors: non-empty sequence of (x, y, z).
    rates: non-empty list or tuple of source rates (mass/s), one per source,
        in source order, each finite and >= 0.
    rate_uncertainty: non-empty list or tuple of independent standard
        uncertainties of the rates, same length and constraints as ``rates``.
    u, direction, sy, sz: dispersion parameters, as in :func:`simulate`.
    Returns ``(C, F, S, T, U)`` without rounding:
    C[i][j] = A[i][j] * rates[j] (receptor x source),
    F[i][j] = C[i][j] / T[i], the source fractions (0 when T[i] == 0),
    S[j] = sum over receptors of C[i][j], in source order,
    T[i] = sum over sources of C[i][j], in receptor order,
    U[i] = sqrt(sum (A[i][j] * rate_uncertainty[j])^2), in receptor order.
    """
    u = _validate_scalar("u", u)
    if u <= 0:
        raise ValueError(f"u must be > 0, got {u!r}")
    direction = _validate_scalar("direction", direction)
    if not 0 <= direction < 360:
        raise ValueError(f"direction must be in [0, 360), got {direction!r}")
    sy = _validate_scalar("sy", sy)
    if sy <= 0:
        raise ValueError(f"sy must be > 0, got {sy!r}")
    sz = _validate_scalar("sz", sz)
    if sz <= 0:
        raise ValueError(f"sz must be > 0, got {sz!r}")

    parsed_sources = _validate_records("sources", sources, 3)
    parsed_receptors = _validate_records("receptors", receptors, 3)

    m = len(parsed_sources)
    parsed_rates = _validate_nonneg_sequence("rates", rates, m)
    parsed_uncertainty = _validate_nonneg_sequence(
        "rate_uncertainty", rate_uncertainty, m
    )

    n = len(parsed_receptors)

    # Unit-rate response: column j is the plume of source j with q = 1.
    columns = [
        simulate([(sx, sy_coord, h, 1.0)], parsed_receptors, u, direction, sy, sz)
        for sx, sy_coord, h in parsed_sources
    ]
    A = [[columns[j][i] for j in range(m)] for i in range(n)]

    C = [[A[i][j] * parsed_rates[j] for j in range(m)] for i in range(n)]
    T = [math.fsum(C[i]) for i in range(n)]
    F = [
        [0.0 if T[i] == 0.0 else C[i][j] / T[i] for j in range(m)]
        for i in range(n)
    ]
    S = [math.fsum(C[i][j] for i in range(n)) for j in range(m)]
    U = [
        math.sqrt(
            math.fsum((A[i][j] * parsed_uncertainty[j]) ** 2 for j in range(m))
        )
        for i in range(n)
    ]

    return C, F, S, T, U
