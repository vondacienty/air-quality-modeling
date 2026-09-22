"""Monitoring comparison of predicted air quality against observations."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number

__all__ = ["compare"]


def _validate_matrix(
    name: str, matrix: object, allow_negative: bool
) -> list[list[float]]:
    if not isinstance(matrix, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(matrix).__name__}")
    if len(matrix) == 0:
        raise ValueError(f"{name} must not be empty")
    width: int | None = None
    validated = []
    for t, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                f"{name}[{t}] must be a list or tuple, got {type(row).__name__}"
            )
        if len(row) == 0:
            raise ValueError(f"{name}[{t}] must not be empty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"{name}[{t}] must have {width} elements, got {len(row)}"
            )
        values = []
        for i, item in enumerate(row):
            if not _is_number(item):
                raise TypeError(
                    f"{name}[{t}][{i}] must be an int or float, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            _check_finite(f"{name}[{t}][{i}]", value)
            if not allow_negative and value < 0:
                raise ValueError(f"{name}[{t}][{i}] must be >= 0, got {value!r}")
            values.append(value)
        validated.append(values)
    return validated


def compare(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    tolerance: float,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Compare predicted concentrations against observations over time.

    observed, predicted: non-empty time x receptor matrices; both the outer
        container and each row must be a list or tuple, rows must be
        non-empty and share one receptor count; each value finite (may be
        negative). All three matrices must have the same shape.
    uncertainty: matrix with the same shape and constraints as the others,
        except each value must be >= 0.
    tolerance: finite non-bool int or float, >= 0.
    Returns ``(bias, rmse, max_abs, exceedance_count)`` in receptor order,
    where with ``e[t][i] = predicted[t][i] - observed[t][i]`` and
    ``n`` the number of time steps:

    * ``bias[i] = fsum(e[t][i] for t) / n``,
    * ``rmse[i] = sqrt(fsum(e[t][i] ** 2 for t) / n)``,
    * ``max_abs[i] = max(abs(e[t][i]))``,
    * ``exceedance_count[i]`` is the number of time steps where
      ``abs(e[t][i]) > tolerance + uncertainty[t][i]``.

    The first three results are lists of floats and the last a list of
    ints; results are not rounded.
    """
    obs = _validate_matrix("observed", observed, allow_negative=True)
    pred = _validate_matrix("predicted", predicted, allow_negative=True)
    unc = _validate_matrix("uncertainty", uncertainty, allow_negative=False)

    if len(pred) != len(obs):
        raise ValueError(
            f"observed and predicted must have the same number of rows, "
            f"got {len(obs)} and {len(pred)}"
        )
    if len(unc) != len(obs):
        raise ValueError(
            f"observed and uncertainty must have the same number of rows, "
            f"got {len(obs)} and {len(unc)}"
        )
    n_receptors = len(obs[0])
    for t in range(len(obs)):
        if len(pred[t]) != n_receptors:
            raise ValueError(
                f"predicted[{t}] must have {n_receptors} elements, "
                f"got {len(pred[t])}"
            )
        if len(unc[t]) != n_receptors:
            raise ValueError(
                f"uncertainty[{t}] must have {n_receptors} elements, "
                f"got {len(unc[t])}"
            )

    if not _is_number(tolerance):
        raise TypeError(
            f"tolerance must be an int or float, got {type(tolerance).__name__}"
        )
    tolerance = float(tolerance)
    _check_finite("tolerance", tolerance)
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")

    n = len(obs)
    bias: list[float] = []
    rmse: list[float] = []
    max_abs: list[float] = []
    exceedance_count: list[int] = []
    for i in range(n_receptors):
        errors = [pred[t][i] - obs[t][i] for t in range(n)]
        bias.append(math.fsum(errors) / n)
        rmse.append(math.sqrt(math.fsum(e * e for e in errors) / n))
        max_abs.append(max(abs(e) for e in errors))
        exceedance_count.append(
            sum(1 for t in range(n) if abs(errors[t]) > tolerance + unc[t][i])
        )

    return bias, rmse, max_abs, exceedance_count
