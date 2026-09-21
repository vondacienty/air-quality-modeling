"""Non-negative ridge-regression inversion of source emission rates."""

from __future__ import annotations

import math

from . import gaussian

__all__ = ["invert"]

_PIVOT_TOL = 1e-12


def _validate_observations(observations: object, expected: int) -> list[float]:
    if not isinstance(observations, (list, tuple)):
        raise TypeError(f"observations must be a list or tuple, got {type(observations).__name__}")
    if len(observations) == 0:
        raise ValueError("observations must not be empty")
    if len(observations) != expected:
        raise ValueError(f"observations must have the same length as receptors, got {len(observations)} != {expected}")
    values = []
    for index, item in enumerate(observations):
        if not gaussian._is_number(item):
            raise TypeError(f"observations[{index}] must be an int or float, got {type(item).__name__}")
        value = float(item)
        gaussian._check_finite(f"observations[{index}]", value)
        values.append(value)
    return values


def _check_full_column_rank(a: list[list[float]]) -> None:
    rows = len(a)
    cols = len(a[0])
    mat = [row[:] for row in a]
    row = 0
    for col in range(cols):
        if row >= rows:
            raise ValueError("response matrix A does not have full column rank")
        pivot = max(range(row, rows), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) <= _PIVOT_TOL:
            raise ValueError("response matrix A does not have full column rank")
        mat[row], mat[pivot] = mat[pivot], mat[row]
        for r in range(row + 1, rows):
            factor = mat[r][col] / mat[row][col]
            for k in range(col, cols):
                mat[r][k] -= factor * mat[row][k]
        row += 1


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(matrix)
    a = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= _PIVOT_TOL:
            raise ValueError("failed to solve the ridge normal equations")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(a[i][i]) <= _PIVOT_TOL:
            raise ValueError("failed to solve the ridge normal equations")
        x[i] = (a[i][n] - math.fsum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    a = [matrix[i][:] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= _PIVOT_TOL:
            raise ValueError("failed to invert the ridge normal equations matrix")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col] / a[col][col]
            if factor == 0.0:
                continue
            for k in range(col, 2 * n):
                a[row][k] -= factor * a[col][k]
    inverse = []
    for i in range(n):
        if abs(a[i][i]) <= _PIVOT_TOL:
            raise ValueError("failed to invert the ridge normal equations matrix")
        inverse.append([a[i][n + j] / a[i][i] for j in range(n)])
    return inverse


def invert(
    sources: list[tuple[float, float, float]],
    receptors: list[tuple[float, float, float]],
    observations: list[float],
    u: float,
    direction: float,
    sy: float,
    sz: float,
    ridge: float = 0.0,
) -> tuple[list[float], list[float], list[float]]:
    """Recover non-negative source emission rates from observed concentrations.

    sources: non-empty sequence of (x, y, h).
    receptors: non-empty sequence of (x, y, z).
    observations: non-empty sequence of concentrations (mass/m^3), one per receptor.
    u, direction, sy, sz: as in gaussian.simulate.
    ridge: non-negative ridge regularization parameter.

    Solves argmin(||Ax - y||^2 + ridge*||x||^2) over x >= 0, where A is the
    response matrix of unit-rate sources. Returns (rates, residuals,
    uncertainty) in source, receptor, and source order respectively, with
    residuals = Ax - y and uncertainty[j] =
    sqrt(fsum(residuals^2)/n * diag((A^T A + ridge*I)^-1)[j]).
    Raises ValueError if A does not have full column rank or a solve fails.
    """
    u = gaussian._validate_scalar("u", u)
    if u <= 0:
        raise ValueError(f"u must be > 0, got {u!r}")
    direction = gaussian._validate_scalar("direction", direction)
    if not 0 <= direction < 360:
        raise ValueError(f"direction must be in [0, 360), got {direction!r}")
    sy = gaussian._validate_scalar("sy", sy)
    if sy <= 0:
        raise ValueError(f"sy must be > 0, got {sy!r}")
    sz = gaussian._validate_scalar("sz", sz)
    if sz <= 0:
        raise ValueError(f"sz must be > 0, got {sz!r}")
    ridge = gaussian._validate_scalar("ridge", ridge)
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge!r}")

    parsed_sources = gaussian._validate_records("sources", sources, 3)
    parsed_receptors = gaussian._validate_records("receptors", receptors, 3)
    y = _validate_observations(observations, len(parsed_receptors))

    n = len(parsed_receptors)
    m = len(parsed_sources)

    columns = [
        gaussian.simulate([(xs, ys, h, 1.0)], parsed_receptors, u, direction, sy, sz)
        for xs, ys, h in parsed_sources
    ]
    a = [[columns[j][i] for j in range(m)] for i in range(n)]

    _check_full_column_rank(a)

    normal = [
        [math.fsum(a[i][j] * a[i][k] for i in range(n)) + (ridge if j == k else 0.0) for k in range(m)]
        for j in range(m)
    ]
    rhs = [math.fsum(a[i][j] * y[i] for i in range(n)) for j in range(m)]

    rates = [rate if rate > 0.0 else 0.0 for rate in _solve(normal, rhs)]

    residuals = [
        math.fsum([a[i][j] * rates[j] for j in range(m)] + [-y[i]])
        for i in range(n)
    ]

    inverse = _invert(normal)
    variance = math.fsum(r * r for r in residuals) / n
    uncertainty = [math.sqrt(variance * inverse[j][j]) for j in range(m)]

    return rates, residuals, uncertainty
