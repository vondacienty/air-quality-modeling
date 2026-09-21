"""Steady-state Gaussian plume model with ground reflection."""

from __future__ import annotations

import math

__all__ = ["simulate"]


def _check_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _check_records(records, length, name):
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(records).__name__}")
    if len(records) == 0:
        raise ValueError(f"{name} must not be empty")
    checked = []
    for i, record in enumerate(records):
        if not isinstance(record, (list, tuple)):
            raise TypeError(f"{name}[{i}] must be a list or tuple, got {type(record).__name__}")
        if len(record) != length:
            raise ValueError(f"{name}[{i}] must have {length} elements, got {len(record)}")
        checked.append(tuple(_check_number(v, f"{name}[{i}][{j}]") for j, v in enumerate(record)))
    return checked


def simulate(sources, receptors, u, direction, sy, sz):
    """Simulate a steady-state Gaussian plume with ground reflection.

    sources: non-empty sequence of (x, y, h, q) records — source position (m),
        effective stack height (m) and emission rate (mass/s, q >= 0).
    receptors: non-empty sequence of (x, y, z) records — receptor position (m).
    u: wind speed (m/s, > 0).
    direction: wind direction in degrees, measured counterclockwise from the
        +x axis toward the downwind direction (0 <= direction < 360).
    sy, sz: lateral and vertical dispersion coefficients (m, > 0).

    Returns a list of concentrations (mass/m^3), one per receptor, in
    receptor order, without rounding.
    """
    sources = _check_records(sources, 4, "sources")
    receptors = _check_records(receptors, 3, "receptors")

    u = _check_number(u, "u")
    direction = _check_number(direction, "direction")
    sy = _check_number(sy, "sy")
    sz = _check_number(sz, "sz")

    if u <= 0:
        raise ValueError(f"u must be > 0, got {u!r}")
    if not 0 <= direction < 360:
        raise ValueError(f"direction must satisfy 0 <= direction < 360, got {direction!r}")
    if sy <= 0:
        raise ValueError(f"sy must be > 0, got {sy!r}")
    if sz <= 0:
        raise ValueError(f"sz must be > 0, got {sz!r}")
    for i, (_, _, _, q) in enumerate(sources):
        if q < 0:
            raise ValueError(f"sources[{i}] q must be >= 0, got {q!r}")

    theta = direction * math.pi / 180.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    denom = 2.0 * math.pi * u * sy * sz
    two_sy2 = 2.0 * sy * sy
    two_sz2 = 2.0 * sz * sz

    results = []
    for xr, yr, z in receptors:
        terms = []
        for x, y, h, q in sources:
            dx = xr - x
            dy = yr - y
            d = dx * cos_t + dy * sin_t
            if d <= 0:
                terms.append(0.0)
                continue
            c = -dx * sin_t + dy * cos_t
            lateral = math.exp(-(c * c) / two_sy2)
            dz_down = z - h
            dz_up = z + h
            vertical = math.exp(-(dz_down * dz_down) / two_sz2) + math.exp(-(dz_up * dz_up) / two_sz2)
            terms.append(q * lateral * vertical / denom)
        results.append(math.fsum(terms))
    return results
