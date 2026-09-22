"""Comparison of observed and predicted concentrations at monitoring receptors."""

from __future__ import annotations

import math

from .gaussian import _check_finite, _is_number
from .warning import _validate_matrix, _validate_thresholds

__all__ = [
    "alert",
    "compare",
    "cusum",
    "evaluate",
    "ewma",
    "interval_score",
    "persistence",
    "standardized_error",
    "summarize",
]


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


def _validate_minimum(minimum: object) -> int:
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise TypeError(
            f"minimum must be an int, got {type(minimum).__name__}"
        )
    if minimum < 1:
        raise ValueError(f"minimum must be >= 1, got {minimum!r}")
    return minimum


def _validate_alpha(alpha: object) -> float:
    if not _is_number(alpha):
        raise TypeError(
            f"alpha must be an int or float, got {type(alpha).__name__}"
        )
    value = float(alpha)
    _check_finite("alpha", value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {value!r}")
    return value


def _validate_interval_alpha(alpha: object) -> float:
    if not _is_number(alpha):
        raise TypeError(
            f"alpha must be an int or float, got {type(alpha).__name__}"
        )
    value = float(alpha)
    _check_finite("alpha", value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {value!r}")
    return value


def _validate_ewma_threshold(threshold: object) -> float:
    if not _is_number(threshold):
        raise TypeError(
            f"threshold must be an int or float, got {type(threshold).__name__}"
        )
    value = float(threshold)
    _check_finite("threshold", value)
    if value < 0:
        raise ValueError(f"threshold must be >= 0, got {value!r}")
    return value


def _validate_drift(drift: object) -> float:
    if not _is_number(drift):
        raise TypeError(
            f"drift must be an int or float, got {type(drift).__name__}"
        )
    value = float(drift)
    _check_finite("drift", value)
    if value < 0:
        raise ValueError(f"drift must be >= 0, got {value!r}")
    return value


def _validate_floor(floor: object) -> float:
    if not _is_number(floor):
        raise TypeError(
            f"floor must be an int or float, got {type(floor).__name__}"
        )
    value = float(floor)
    _check_finite("floor", value)
    if value <= 0:
        raise ValueError(f"floor must be > 0, got {value!r}")
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


def interval_score(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    alpha: float = 0.1,
) -> tuple[list[float], list[float], list[float]]:
    """Score prediction intervals against observations per receptor.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    alpha: non-bool finite int or float strictly between 0 and 1
        (default 0.1); penalty weight for observations outside the
        interval.

    Writing ``lo[t][i] = predicted[t][i] - uncertainty[t][i]``,
    ``hi[t][i] = predicted[t][i] + uncertainty[t][i]`` and
    ``r[t][i] = observed[t][i]``, returns ``(coverage, width, score)``,
    plain lists of length N in receptor order:

    * ``coverage[i] = sum(lo[t][i] <= r[t][i] <= hi[t][i] for t) / K``,
    * ``width[i] = fsum(hi[t][i] - lo[t][i] for t) / K``,
    * ``score[i] = fsum(s[t][i] for t) / K`` where
      ``s[t][i] = (hi[t][i] - lo[t][i]) + 2 / alpha * (lo[t][i] -
      r[t][i])`` when ``r[t][i] < lo[t][i]``,
      ``s[t][i] = (hi[t][i] - lo[t][i]) + 2 / alpha * (r[t][i] -
      hi[t][i])`` when ``r[t][i] > hi[t][i]`` and
      ``s[t][i] = hi[t][i] - lo[t][i]`` otherwise.

    All results are floats and are not rounded.
    """
    obs = _validate_matrix("observed", observed, allow_negative=True)
    pred = _validate_matrix("predicted", predicted, allow_negative=True)
    unc = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    alpha_value = _validate_interval_alpha(alpha)

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

    two_over_alpha = 2.0 / alpha_value
    coverage: list[float] = []
    width: list[float] = []
    score: list[float] = []
    for i in range(n_receptors):
        inside = 0
        widths: list[float] = []
        scores: list[float] = []
        for t in range(n_rows):
            low = pred[t][i] - unc[t][i]
            high = pred[t][i] + unc[t][i]
            real = obs[t][i]
            interval_width = high - low
            widths.append(interval_width)
            if real < low:
                scores.append(
                    interval_width + two_over_alpha * (low - real)
                )
            elif real > high:
                scores.append(
                    interval_width + two_over_alpha * (real - high)
                )
            else:
                inside += 1
                scores.append(interval_width)
        coverage.append(inside / n_rows)
        width.append(math.fsum(widths) / n_rows)
        score.append(math.fsum(scores) / n_rows)

    return coverage, width, score


def alert(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    tolerance: float = 0.0,
) -> tuple[list[int], list[tuple[float, int]], list[float]]:
    """Assess alert levels of prediction errors over time.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative alert thresholds.
    tolerance: non-bool finite int or float >= 0 (default 0.0); absolute
        errors up to ``tolerance + uncertainty[t][i]`` are disregarded.

    Returns ``(levels, evidence, scores)``, plain lists of length K in
    time order, where for each time ``t`` and receptor ``i``::

        x[t][i] = max(abs(predicted[t][i] - observed[t][i])
                      - tolerance - uncertainty[t][i], 0.0)
        scores[t] = fsum(x[t])
        evidence[t] = (scores[t], sum(1 for v in x[t] if v > 0))
        levels[t] = number of thresholds <= scores[t]   (0-3)

    Levels are ints, evidence entries are ``(float, int)`` pairs and
    scores are floats; results are not rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )
    levels_thresholds = _validate_thresholds(thresholds)

    levels: list[int] = []
    evidence: list[tuple[float, int]] = []
    scores: list[float] = []
    for t in range(len(obs)):
        x_row = [
            max(abs(pred[t][i] - obs[t][i]) - tol - unc[t][i], 0.0)
            for i in range(len(obs[t]))
        ]
        score = math.fsum(x_row)
        scores.append(score)
        evidence.append((score, sum(1 for v in x_row if v > 0)))
        levels.append(
            sum(1 for threshold in levels_thresholds if threshold <= score)
        )

    return levels, evidence, scores


def ewma(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    alpha: float,
    threshold: float,
    tolerance: float = 0.0,
) -> tuple[list[float], list[float], list[int], list[int]]:
    """Exponentially weighted moving averages of prediction errors.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    alpha: non-bool finite int or float in ``(0, 1]``; EWMA smoothing
        factor (larger values weight new errors more heavily).
    threshold: non-bool finite int or float >= 0; an EWMA level at or
        above it counts as a trigger.
    tolerance: non-bool finite int or float >= 0 (default 0.0); absolute
        errors up to ``tolerance + uncertainty[t][i]`` are disregarded.

    Returns ``(final, peak, first, triggered)``, plain lists of length N
    in receptor order, where for each time ``t`` and receptor ``i``::

        x[t][i] = max(abs(predicted[t][i] - observed[t][i])
                      - tolerance - uncertainty[t][i], 0.0)
        s[-1][i] = 0.0
        s[t][i] = alpha * x[t][i] + (1 - alpha) * s[t - 1][i]
        final[i] = s[K - 1][i]
        peak[i] = max(s[t][i] for t in range(K))
        first[i] = smallest t with s[t][i] >= threshold, else -1
        triggered[i] = number of rows with s[t][i] >= threshold

    ``final`` and ``peak`` are floats; ``first`` and ``triggered`` are
    ints; results are not rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )
    alpha_value = _validate_alpha(alpha)
    threshold_value = _validate_ewma_threshold(threshold)

    n_rows = len(obs)
    n_receptors = len(obs[0])
    one_minus_alpha = 1.0 - alpha_value
    final: list[float] = []
    peak: list[float] = []
    first: list[int] = []
    triggered: list[int] = []
    for i in range(n_receptors):
        previous = 0.0
        peak_value = 0.0
        first_index = -1
        trigger_count = 0
        for t in range(n_rows):
            x = max(abs(pred[t][i] - obs[t][i]) - tol - unc[t][i], 0.0)
            current = alpha_value * x + one_minus_alpha * previous
            if current > peak_value:
                peak_value = current
            if current >= threshold_value:
                if first_index < 0:
                    first_index = t
                trigger_count += 1
            previous = current
        final.append(previous)
        peak.append(peak_value)
        first.append(first_index)
        triggered.append(trigger_count)

    return final, peak, first, triggered


def cusum(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    drift: float,
    threshold: float,
    tolerance: float = 0.0,
) -> tuple[list[float], list[float], list[int], list[int]]:
    """Cumulative sums of prediction errors with a drift allowance.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    drift: non-bool finite int or float >= 0; amount subtracted from the
        cumulative sum at each time step.
    threshold: non-bool finite int or float >= 0; a cumulative-sum level
        at or above it counts as a trigger.
    tolerance: non-bool finite int or float >= 0 (default 0.0); absolute
        errors up to ``tolerance + uncertainty[t][i]`` are disregarded.

    Returns ``(final, peak, first, triggered)``, plain lists of length N
    in receptor order, where for each time ``t`` and receptor ``i``::

        x[t][i] = max(abs(predicted[t][i] - observed[t][i])
                      - tolerance - uncertainty[t][i], 0.0)
        s[-1][i] = 0.0
        s[t][i] = max(0.0, s[t - 1][i] + x[t][i] - drift)
        final[i] = s[K - 1][i]
        peak[i] = max(s[t][i] for t in range(K))
        first[i] = smallest t with s[t][i] >= threshold, else -1
        triggered[i] = number of rows with s[t][i] >= threshold

    ``final`` and ``peak`` are floats; ``first`` and ``triggered`` are
    ints; results are not rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )
    drift_value = _validate_drift(drift)
    threshold_value = _validate_ewma_threshold(threshold)

    n_rows = len(obs)
    n_receptors = len(obs[0])
    final: list[float] = []
    peak: list[float] = []
    first: list[int] = []
    triggered: list[int] = []
    for i in range(n_receptors):
        previous = 0.0
        peak_value = 0.0
        first_index = -1
        trigger_count = 0
        for t in range(n_rows):
            x = max(abs(pred[t][i] - obs[t][i]) - tol - unc[t][i], 0.0)
            current = max(0.0, previous + x - drift_value)
            if current > peak_value:
                peak_value = current
            if current >= threshold_value:
                if first_index < 0:
                    first_index = t
                trigger_count += 1
            previous = current
        final.append(previous)
        peak.append(peak_value)
        first.append(first_index)
        triggered.append(trigger_count)

    return final, peak, first, triggered


def summarize(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    thresholds: list[float] | tuple[float, ...],
    tolerance: float = 0.0,
) -> tuple[list[float], list[float], list[int], list[int], list[int]]:
    """Summarize prediction-error exceedances per receptor.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    thresholds: list or tuple of 3 strictly increasing, finite,
        non-negative alert thresholds.
    tolerance: non-bool finite int or float >= 0 (default 0.0); absolute
        errors up to ``tolerance + uncertainty[t][i]`` are disregarded.

    Returns ``(total, peak, levels, times, triggered)``, plain lists of
    length N in receptor order, where for each time ``t`` and receptor
    ``i``::

        x[t][i] = max(abs(predicted[t][i] - observed[t][i])
                      - tolerance - uncertainty[t][i], 0.0)
        total[i] = fsum(x[t][i] for t)
        peak[i] = max(x[t][i] for t)
        times[i] = smallest t with x[t][i] == peak[i]
        levels[i] = number of thresholds <= peak[i]   (0-3)
        triggered[i] = sum(1 for t if x[t][i] > 0)

    ``total`` and ``peak`` are floats; ``levels``, ``times`` and
    ``triggered`` are ints; results are not rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )
    peak_thresholds = _validate_thresholds(thresholds)

    n_rows = len(obs)
    n_receptors = len(obs[0])
    total: list[float] = []
    peak: list[float] = []
    levels: list[int] = []
    times: list[int] = []
    triggered: list[int] = []
    for i in range(n_receptors):
        column = [
            max(abs(pred[t][i] - obs[t][i]) - tol - unc[t][i], 0.0)
            for t in range(n_rows)
        ]
        peak_value = max(column)
        total.append(math.fsum(column))
        peak.append(peak_value)
        levels.append(
            sum(
                1
                for threshold in peak_thresholds
                if threshold <= peak_value
            )
        )
        times.append(column.index(peak_value))
        triggered.append(sum(1 for value in column if value > 0))

    return total, peak, levels, times, triggered


def persistence(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    tolerance: float = 0.0,
    minimum: int = 1,
) -> tuple[list[int], list[int], list[int]]:
    """Lengths of consecutive prediction-exceedance runs per receptor.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    tolerance: non-bool finite int or float >= 0 (default 0.0); an
        absolute error is an exceedance only when it is strictly greater
        than ``tolerance + uncertainty[t][i]``.
    minimum: non-bool int >= 1 (default 1); a run is counted in
        ``qualifying_runs`` when its length is at least ``minimum``.

    Returns ``(longest, first_start, qualifying_runs)``, plain lists of
    length N in receptor order. Writing ``x[t][i] =
    abs(predicted[t][i] - observed[t][i]) > tolerance +
    uncertainty[t][i]``, adjacent true values within each column form
    consecutive runs, and for each receptor ``i``:

    * ``longest[i]`` is the length of the longest run (0 if no run),
    * ``first_start[i]`` is the smallest starting index ``t`` among the
      runs whose length equals ``longest[i]`` (-1 if no run),
    * ``qualifying_runs[i]`` is the number of runs whose length is
      ``>= minimum``.

    All results are ints and are not rounded.
    """
    obs, pred, unc, tol = _validate_inputs(
        observed, predicted, uncertainty, tolerance
    )
    minimum_value = _validate_minimum(minimum)

    n_rows = len(obs)
    n_receptors = len(obs[0])
    longest: list[int] = []
    first_start: list[int] = []
    qualifying_runs: list[int] = []
    for i in range(n_receptors):
        column_longest = 0
        column_first = -1
        column_qualifying = 0
        run_start = 0
        run_length = 0
        for t in range(n_rows):
            if abs(pred[t][i] - obs[t][i]) > tol + unc[t][i]:
                if run_length == 0:
                    run_start = t
                run_length += 1
            else:
                if run_length > 0:
                    if run_length > column_longest:
                        column_longest = run_length
                        column_first = run_start
                    if run_length >= minimum_value:
                        column_qualifying += 1
                    run_length = 0
        if run_length > 0:
            if run_length > column_longest:
                column_longest = run_length
                column_first = run_start
            if run_length >= minimum_value:
                column_qualifying += 1
        longest.append(column_longest)
        first_start.append(column_first)
        qualifying_runs.append(column_qualifying)

    return longest, first_start, qualifying_runs


def standardized_error(
    observed: list[list[float]] | tuple[tuple[float, ...], ...],
    predicted: list[list[float]] | tuple[tuple[float, ...], ...],
    uncertainty: list[list[float]] | tuple[tuple[float, ...], ...],
    floor: float = 1e-12,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Uncertainty-standardized prediction errors per receptor.

    observed: non-empty time x receptor (K x N) matrix of observed
        concentrations; both the outer container and each row must be a
        list or tuple, rows must be non-empty and share one receptor
        count; each value a finite non-bool int or float (may be
        negative).
    predicted: matrix of predicted concentrations with the same shape
        and constraints as ``observed``.
    uncertainty: matrix of prediction uncertainties with the same shape
        as ``observed``; each value must additionally be >= 0.
    floor: non-bool finite int or float strictly greater than 0
        (default 1e-12); lower bound applied to each uncertainty when
        standardizing.

    Returns ``(mean, peak, coverage, weighted)``, plain lists of length
    N in receptor order, where for each time ``t`` and receptor ``i``::

        e[t][i] = abs(predicted[t][i] - observed[t][i])
        d[t][i] = e[t][i] / max(uncertainty[t][i], floor)
        W = fsum(t + 1 for t in range(K))
        mean[i] = fsum(d[t][i] for t) / K
        peak[i] = max(d[t][i] for t)
        coverage[i] = sum(d[t][i] <= 1 for t) / K
        weighted[i] = fsum((t + 1) * d[t][i] for t) / W

    All results are floats and are not rounded.
    """
    obs = _validate_matrix("observed", observed, allow_negative=True)
    pred = _validate_matrix("predicted", predicted, allow_negative=True)
    unc = _validate_matrix("uncertainty", uncertainty, allow_negative=False)
    floor_value = _validate_floor(floor)

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

    weights_total = math.fsum(t + 1 for t in range(n_rows))
    mean: list[float] = []
    peak: list[float] = []
    coverage: list[float] = []
    weighted: list[float] = []
    for i in range(n_receptors):
        column = [
            abs(pred[t][i] - obs[t][i]) / max(unc[t][i], floor_value)
            for t in range(n_rows)
        ]
        mean.append(math.fsum(column) / n_rows)
        peak.append(max(column))
        coverage.append(sum(1 for value in column if value <= 1) / n_rows)
        weighted.append(
            math.fsum((t + 1) * column[t] for t in range(n_rows))
            / weights_total
        )

    return mean, peak, coverage, weighted
