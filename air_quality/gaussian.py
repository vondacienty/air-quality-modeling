"""Steady-state Gaussian plume model with ground reflection."""

from __future__ import annotations

import math

__all__ = ["simulate"]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _validate_scalar(name: str, value: object) -> float:
    if not _is_number(value):
        raise TypeError(f"{name} must be an int or float, got {type(value).__name__}")
    result = float(value)
    _check_finite(name, result)
    return result


def _validate_records(name: str, records: object, length: int) -> list[tuple[float, ...]]:
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(records).__name__}")
    if len(records) == 0:
        raise ValueError(f"{name} must not be empty")
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, (list, tuple)):
            raise TypeError(f"{name}[{index}] must be a list or tuple, got {type(record).__name__}")
        if len(record) != length:
            raise ValueError(f"{name}[{index}] must have {length} elements, got {len(record)}")
        values = []
        for field, item in enumerate(record):
            if not _is_number(item):
                raise TypeError(f"{name}[{index}][{field}] must be an int or float, got {type(item).__name__}")
            value = float(item)
            _check_finite(f"{name}[{index}][{field}]", value)
            values.append(value)
        validated.append(tuple(values))
    return validated


def simulate(
    sources: list[tuple[float, float, float, float]],
    receptors: list[tuple[float, float, float]],
    u: float,
    direction: float,
    sy: float,
    sz: float,
) -> list[float]:
    """Simulate a steady-state Gaussian plume with ground reflection.

    sources: non-empty sequence of (x, y, h, q) with q >= 0 (mass/s).
    receptors: non-empty sequence of (x, y, z).
    u: wind speed in m/s (> 0). direction: downwind direction in degrees,
    measured counterclockwise from the +x axis, in [0, 360).
    sy, sz: dispersion coefficients in m (> 0).
    Returns concentrations (mass/m^3) in receptor order.
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

    parsed_sources = _validate_records("sources", sources, 4)
    parsed_receptors = _validate_records("receptors", receptors, 3)
    for index, (_, _, _, q) in enumerate(parsed_sources):
        if q < 0:
            raise ValueError(f"sources[{index}] q must be >= 0, got {q!r}")

    theta = direction * math.pi / 180
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    denom = 2 * math.pi * u * sy * sz
    two_sy2 = 2 * sy * sy
    two_sz2 = 2 * sz * sz

    results = []
    for xr, yr, z in parsed_receptors:
        contributions = []
        for x, y, h, q in parsed_sources:
            dx = xr - x
            dy = yr - y
            d = dx * cos_t + dy * sin_t
            if d <= 0:
                contributions.append(0.0)
                continue
            c = -dx * sin_t + dy * cos_t
            lateral = math.exp(-(c * c) / two_sy2)
            dz_down = z - h
            dz_up = z + h
            vertical = math.exp(-(dz_down * dz_down) / two_sz2) + math.exp(-(dz_up * dz_up) / two_sz2)
            contributions.append(q * lateral * vertical / denom)
        results.append(math.fsum(contributions))
    return results
