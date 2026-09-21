"""Non-negative ridge-regression source inversion for Gaussian plumes."""

from __future__ import annotations

import math

from .gaussian import (
    _check_finite,
    _is_number,
    _validate_records,
    _validate_scalar,
    simulate,
)

__all__ = ["invert"]

#: Absolute pivot threshold: pivots at or below this magnitude count as zero.
_PIVOT_EPS = 1e-12


def _lu_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``matrix x = rhs`` via Gaussian elimination with partial pivoting.

    Raises ValueError when an absolute pivot no larger than ``_PIVOT_EPS`` is
    encountered (singular system) or the solve otherwise cannot proceed.
    """
    n = len(matrix)
    work = [row[:] for row in matrix]
    b = list(rhs)
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(work[r][k]))
        pivot = work[pivot_row][k]
        if abs(pivot) <= _PIVOT_EPS:
            raise ValueError(
                f"singular system in source inversion: pivot {pivot!r} has "
                f"absolute value <= {_PIVOT_EPS}"
            )
        if pivot_row != k:
            work[k], work[pivot_row] = work[pivot_row], work[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]
        for r in range(k + 1, n):
            factor = work[r][k] / pivot
            if factor == 0.0:
                continue
            row_r = work[r]
            row_k = work[k]
            for c in range(k + 1, n):
                row_r[c] -= factor * row_k[c]
            row_r[k] = 0.0
            b[r] -= factor * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        tail = math.fsum(work[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - tail) / work[i][i]
    return x


def _has_full_column_rank(A: list[list[float]], m: int) -> bool:
    """Rank of A by Gaussian elimination with partial (row) pivoting."""
    n_rows = len(A)
    work = [row[:] for row in A]
    for k in range(m):
        if k >= n_rows:
            return False
        pivot_row = max(range(k, n_rows), key=lambda r: abs(work[r][k]))
        pivot = work[pivot_row][k]
        if abs(pivot) <= _PIVOT_EPS:
            return False
        if pivot_row != k:
            work[k], work[pivot_row] = work[pivot_row], work[k]
        for r in range(k + 1, n_rows):
            factor = work[r][k] / pivot
            if factor == 0.0:
                continue
            row_r = work[r]
            row_k = work[k]
            for c in range(k + 1, m):
                row_r[c] -= factor * row_k[c]
            row_r[k] = 0.0
    return True


def _nnls(
    G: list[list[float]],
    b: list[float],
) -> list[float]:
    """Non-negative least squares (Lawson-Hanson active set).

    Minimizes ``x^T G x - 2 b^T x`` over ``x >= 0``, where
    ``G = A^T A + ridge I`` and ``b = A^T y``. Passive-set subproblems are
    solved by :func:`_lu_solve`, which raises ValueError on rank failure.
    """
    m = len(b)
    x = [0.0] * m
    passive: set[int] = set()
    w = b[:]  # dual gradient b - G x; x starts at zero
    max_iterations = max(32, 8 * m * m)
    iterations = 0

    while True:
        active = [j for j in range(m) if j not in passive]
        if not active:
            break
        incoming = max(active, key=lambda j: w[j])
        if w[incoming] <= 0.0:
            break
        passive.add(incoming)

        while True:
            iterations += 1
            if iterations > max_iterations:
                raise ValueError(
                    "non-negative least squares failed to converge within "
                    f"{max_iterations} active-set iterations"
                )
            indices = sorted(passive)
            sub = [[G[i][j] for j in indices] for i in indices]
            rhs = [b[i] for i in indices]
            sub_solution = _lu_solve(sub, rhs)
            s = [0.0] * m
            for pos, j in enumerate(indices):
                s[j] = sub_solution[pos]

            if all(s[j] > 0.0 for j in passive):
                x = s
                break

            alpha = math.inf
            frozen = -1
            blocking = [j for j in passive if s[j] <= 0.0]
            for j in blocking:
                denom = x[j] - s[j]
                step = x[j] / denom if denom > 0.0 else 0.0
                if step < alpha:
                    alpha = step
                    frozen = j
            if frozen < 0:  # pragma: no cover - defensive, blocking is non-empty
                raise ValueError("non-negative least squares line search failed")
            for j in range(m):
                value = x[j] + alpha * (s[j] - x[j])
                x[j] = value if value > 0.0 else 0.0
            passive.discard(frozen)
            for j in list(passive):
                if x[j] == 0.0:
                    passive.remove(j)

        for i in range(m):
            w[i] = b[i] - math.fsum(G[i][j] * x[j] for j in range(m))

    return x


def invert(
    sources: list[tuple[float, float, float]],
    receptors: list[tuple[float, float, float]],
    observations: list[float] | tuple[float, ...],
    u: float,
    direction: float,
    sy: float,
    sz: float,
    ridge: float = 0.0,
) -> tuple[list[float], list[float], list[float]]:
    """Infer non-negative emission rates from observed concentrations.

    Builds the unit-rate response matrix A from :func:`gaussian.simulate`
    (each source released at q = 1) and solves the non-negative ridge
    regression ``argmin ||A x - y||^2 + ridge ||x||^2``.

    sources: non-empty sequence of (x, y, h).
    receptors: non-empty sequence of (x, y, z).
    observations: non-empty list or tuple of concentrations (mass/m^3), one
        per receptor, in receptor order.
    u, direction, sy, sz: dispersion parameters, as in :func:`simulate`.
    ridge: ridge penalty, finite and >= 0.
    Returns ``(rates, residuals, uncertainty)``: rates in source order,
    residuals ``A x - y`` in receptor order, and per-source standard
    uncertainty ``sqrt(RSS/n * diag((A^T A + ridge I)^-1))`` in source order.
    Raises ValueError if the columns of A are rank deficient (absolute pivot
    <= 1e-12 under partial-pivot elimination) or a solve fails.
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

    if not isinstance(observations, (list, tuple)):
        raise TypeError(
            f"observations must be a list or tuple, got {type(observations).__name__}"
        )
    if len(observations) == 0:
        raise ValueError("observations must not be empty")
    if len(observations) != len(parsed_receptors):
        raise ValueError(
            f"observations length must equal receptors length "
            f"{len(parsed_receptors)}, got {len(observations)}"
        )
    y = []
    for index, item in enumerate(observations):
        if not _is_number(item):
            raise TypeError(
                f"observations[{index}] must be an int or float, got {type(item).__name__}"
            )
        value = float(item)
        _check_finite(f"observations[{index}]", value)
        y.append(value)

    ridge = _validate_scalar("ridge", ridge)
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge!r}")

    n = len(parsed_receptors)
    m = len(parsed_sources)

    # Unit-rate response: column j is the plume of source j with q = 1.
    columns = [
        simulate([(sx, sy_coord, h, 1.0)], parsed_receptors, u, direction, sy, sz)
        for sx, sy_coord, h in parsed_sources
    ]
    A = [[columns[j][i] for j in range(m)] for i in range(n)]

    if not _has_full_column_rank(A, m):
        raise ValueError(
            "response matrix A is column rank deficient (absolute pivot "
            f"<= {_PIVOT_EPS}); sources cannot be distinguished"
        )

    G = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            entry = math.fsum(A[row][i] * A[row][j] for row in range(n))
            if i == j:
                entry += ridge
            G[i][j] = entry
            G[j][i] = entry
    b = [math.fsum(A[row][j] * y[row] for row in range(n)) for j in range(m)]

    rates = _nnls(G, b)

    fitted = [math.fsum(A[i][j] * rates[j] for j in range(m)) for i in range(n)]
    residuals = [fitted[i] - y[i] for i in range(n)]

    rss = math.fsum(r * r for r in residuals)
    residual_variance = rss / n

    uncertainty = []
    for j in range(m):
        unit = [0.0] * m
        unit[j] = 1.0
        inverse_column = _lu_solve(G, unit)
        variance = residual_variance * inverse_column[j]
        uncertainty.append(math.sqrt(variance))

    return rates, residuals, uncertainty
