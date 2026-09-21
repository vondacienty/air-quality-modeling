"""Population health impact assessment across concentration scenarios."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number

__all__ = ["assess"]


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
        if width is not None and len(row) != width:
            raise ValueError(
                f"{name}[{k}] must have {width} receptors, got {len(row)}"
            )
        parsed = []
        for i, item in enumerate(row):
            if not _is_number(item):
                raise TypeError(
                    f"{name}[{k}][{i}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"{name}[{k}][{i}]", value)
            if value < 0:
                raise ValueError(f"{name}[{k}][{i}] must be >= 0, got {value!r}")
            parsed.append(value)
        if width is None:
            width = len(parsed)
        rows.append(parsed)
    assert width is not None
    return rows, width


def _validate_population(name: str, values: object, expected_length: int) -> list[float]:
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
    """Assess population health impacts of concentration changes vs a baseline.

    C: non-empty list or tuple of scenarios; each scenario is a non-empty
        list or tuple of concentrations, one per receptor, in receptor
        order; each must be finite and >= 0. All rows must agree in length.
    U: non-empty list or tuple of scenarios with the same shape as ``C``;
        each entry is the mutually independent standard uncertainty of the
        corresponding concentration, finite and >= 0.
    P: non-empty list or tuple of population sizes, one per receptor; each
        must be finite and >= 0.
    beta: cases per person per unit concentration; finite and >= 0.
    baseline: index of the reference scenario; a non-bool int with
        ``0 <= baseline < len(C)`` (defaults to 0).

    Returns ``(H, S, W, Q)`` where, writing the baseline scenario as b:

    * ``H[k][i] = P[i] * beta * (C[k][i] - C[b][i])`` is the change in
      attributable cases at receptor i,
    * ``W[k][i] = P[i] * beta * sqrt(U[k][i]**2 + U[b][i]**2)`` is its
      propagated standard uncertainty,
    * ``S[k] = fsum(H[k])`` is the total change over receptors,
    * ``Q[k] = sqrt(fsum(W[k][i]**2))`` is the combined uncertainty.

    The baseline rows of ``H`` and ``W`` are directly 0.0 (and hence so are
    ``S[b]`` and ``Q[b]``). All results are plain lists of floats in input
    order and are not rounded.
    """
    Cv, c_width = _validate_matrix("C", C)
    Uv, u_width = _validate_matrix("U", U)
    if len(Cv) != len(Uv):
        raise ValueError(
            f"C and U must have the same number of scenarios, "
            f"got {len(Cv)} and {len(Uv)}"
        )
    if c_width != u_width:
        raise ValueError(
            f"C and U rows must have the same receptor count, "
            f"got {c_width} and {u_width}"
        )

    Pv = _validate_population("P", P, c_width)

    if not _is_number(beta):
        raise TypeError(f"beta must be an int or float, got {type(beta).__name__}")
    beta = float(beta)
    _check_finite("beta", beta)
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta!r}")

    if isinstance(baseline, bool) or not isinstance(baseline, int):
        raise TypeError(f"baseline must be an int, got {type(baseline).__name__}")
    if not 0 <= baseline < len(Cv):
        raise ValueError(f"baseline must be in [0, {len(Cv)}), got {baseline!r}")

    n_scenarios = len(Cv)
    n_receptors = c_width
    base_c = Cv[baseline]
    base_u = Uv[baseline]

    H: list[list[float]] = []
    W: list[list[float]] = []
    S: list[float] = []
    Q: list[float] = []
    for k in range(n_scenarios):
        if k == baseline:
            h = [0.0] * n_receptors
            w = [0.0] * n_receptors
        else:
            h = [
                Pv[i] * beta * (Cv[k][i] - base_c[i]) for i in range(n_receptors)
            ]
            w = [
                Pv[i]
                * beta
                * math.sqrt(Uv[k][i] * Uv[k][i] + base_u[i] * base_u[i])
                for i in range(n_receptors)
            ]
        H.append(h)
        W.append(w)
        S.append(math.fsum(h))
        Q.append(math.sqrt(math.fsum(x * x for x in w)))

    return H, S, W, Q
