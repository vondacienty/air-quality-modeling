"""Comparison of observed and predicted concentrations at monitoring receptors."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number
from .warning import _validate_matrix

__all__ = ["compare", "evaluate"]


def _validate_tolerance(tolerance: object) -> float:
    if not _is_number(tolerance):
        raise TypeError(
            f"tolerance must be an int or float, got {type(tolerance).__name__}"
        )
    value = float(tolerance)
    _check_finite("tolerance", value)
    if value < 0:
        raise ValueError(f"tolerance must be >= 0, got {value!r}")
    return value


def _validate_inputs(
    observed: object,
    predicted: object,
    uncertainty: object,
    tolerance: object,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], float]:
    obs = _validate_matrix("observed", observed, allow_negative=True)
    pred = _validate_matrix("predicted", predicted, allow_negative=True)
    unc = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    tol = _validate_tolerance(tolerance)

    n_rows = len(obs)
    n_receptors = len(obs[0])
    for name, matrix in (("predicted", pred), ("uncertainty", unc)):
        if len(matrix) != n_rows:
            raise ValueError(
                f"observed and {name} must have the same number of rows, "
                f"got {n_rows} and {len(matrix)}"
            )
        for t, row in enumerate(matrix):
            if len(row) != n_receptors:
                raise ValueError(
                    f"{name}[{t}] must have {n_receptors} elements, "
                    f"got {len(row)}"
                )
    return obs, pred, unc, tol


def compare(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    tolerance: float,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Compare predicted against observed concentrations per receptor.

    observed: non-empty time x receptor matrix of observed concentrations;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value a finite
        non-bool int or float.
    predicted: matrix of predicted concentrations with the same shape and
        constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape as
        ``observed``; each value must additionally be >= 0.
    tolerance: non-bool finite int or float >= 0; an absolute error is an
        exceedance only when it is strictly greater than
        ``tolerance + uncertainty[t][i]``.

    Returns ``(bias, rmse, max_abs, exceedance_count)``, one entry per
    receptor in input order. With ``e[t][i] = predicted[t][i] -
    observed[t][i]`` and ``n`` the number of time steps (rows):

    * ``bias[i] = fsum(e[t][i] for t) / n``,
    * ``rmse[i] = sqrt(fsum(e[t][i] ** 2 for t) / n)``,
    * ``max_abs[i] = max(abs(e[t][i]) for t)``,
    * ``exceedance_count[i]`` is the number of rows with
      ``abs(e[t][i]) > tolerance + uncertainty[t][i]``.

    The first three results are floats and the counts are ints; nothing is
    rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )

    n_rows = len(obs)
    n_receptors = len(obs[0])
    bias: list[float] = []
    rmse: list[float] = []
    max_abs: list[float] = []
    exceedance_count: list[int] = []
    for i in range(n_receptors):
        errors = [pred[t][i] - obs[t][i] for t in range(n_rows)]
        bias.append(math.fsum(errors) / n_rows)
        rmse.append(math.sqrt(math.fsum(e * e for e in errors) / n_rows))
        abs_errors = [abs(e) for e in errors]
        max_abs.append(max(abs_errors))
        exceedance_count.append(
            sum(
                1
                for t in range(n_rows)
                if abs_errors[t] > tol + unc[t][i]
            )
        )

    return bias, rmse, max_abs, exceedance_count


def evaluate(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    tolerance: float,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Evaluate prediction intervals against observations per receptor.

    observed: non-empty time x receptor matrix of observed concentrations;
        both the outer container and each row must be a list or tuple, rows
        must be non-empty and share one receptor count; each value a finite
        non-bool int or float.
    predicted: matrix of predicted concentrations with the same shape and
        constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape as
        ``observed``; each value must additionally be >= 0.
    tolerance: non-bool finite int or float >= 0; an absolute error is an
        exceedance only when it is strictly greater than
        ``tolerance + uncertainty[t][i]``.

    Returns ``(coverage, mae, sharpness, exceedance_count)``, one entry per
    receptor in input order. With ``e[t][i] = predicted[t][i] -
    observed[t][i]`` and ``n`` the number of time steps (rows):

    * ``coverage[i] = sum(abs(e[t][i]) <= uncertainty[t][i] for t) / n``,
    * ``mae[i] = fsum(abs(e[t][i]) for t) / n``,
    * ``sharpness[i] = fsum(2 * uncertainty[t][i] for t) / n``,
    * ``exceedance_count[i]`` is the number of rows with
      ``abs(e[t][i]) > tolerance + uncertainty[t][i]``.

    The first three results are floats and the counts are ints; nothing is
    rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )

    n_rows = len(obs)
    n_receptors = len(obs[0])
    coverage: list[float] = []
    mae: list[float] = []
    sharpness: list[float] = []
    exceedance_count: list[int] = []
    for i in range(n_receptors):
        abs_errors = [abs(pred[t][i] - obs[t][i]) for t in range(n_rows)]
        coverage.append(
            sum(
                1
                for t in range(n_rows)
                if abs_errors[t] <= unc[t][i]
            )
            / n_rows
        )
        mae.append(math.fsum(abs_errors) / n_rows)
        sharpness.append(
            math.fsum(2 * unc[t][i] for t in range(n_rows)) / n_rows
        )
        exceedance_count.append(
            sum(
                1
                for t in range(n_rows)
                if abs_errors[t] > tol + unc[t][i]
            )
        )

    return coverage, mae, sharpness, exceedance_count
