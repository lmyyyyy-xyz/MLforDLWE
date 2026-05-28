#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import sys
from collections import defaultdict
from itertools import product
from math import ceil, exp
from time import perf_counter
from typing import Iterable, List, Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import qr as scipy_qr, svd as scipy_svd
from scipy.optimize import linprog, minimize
from scipy.sparse.linalg import lsqr

try:
    from sklearn.linear_model import LinearRegression
except Exception as exc:  # sklearn is only required by the regression-style Cauchy solver
    LinearRegression = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except Exception:  # scipy may be old
    Bounds = None
    LinearConstraint = None
    milp = None

try:
    from hint_solver import PyBP, PyGreedy
except Exception as exc:  # hint_solver is optional unless bp/greedy_dh is used
    PyBP = None
    PyGreedy = None
    HINT_SOLVER_IMPORT_ERROR = exc
else:
    HINT_SOLVER_IMPORT_ERROR = None


Dist = List[Tuple[int, float]]
Hint = Tuple[List[int], Dist]


# ============================================================
# Basic utilities
# ============================================================

def normalize_dist(dist: Iterable[Tuple[int, float]]) -> Dist:
    acc = defaultdict(float)
    for value, weight in dist:
        weight = float(weight)
        if weight <= 0.0 or not np.isfinite(weight):
            continue
        acc[int(value)] += weight
    total = sum(acc.values())
    if total <= 0:
        raise ValueError("empty distribution")
    return [(v, w / total) for v, w in sorted(acc.items())]


def round_clip(x: np.ndarray, eta: int) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64)), -eta, eta).astype(np.int64)


def from_compl(v: int, sz_msg: int) -> int:
    """Decode complement-style index used by distribution_hints_solvers."""
    v = int(v)
    sz_msg = int(sz_msg)
    if v > sz_msg / 2:
        return v - sz_msg
    return v


def decode_bp_results(results) -> np.ndarray:
    raw = results[0] if isinstance(results, tuple) and len(results) == 2 else results
    out = []
    for dist in raw:
        if isinstance(dist, (list, tuple, np.ndarray)):
            dist = list(dist)
            idx = int(np.argmax(dist))
            out.append(from_compl(idx, len(dist)))
        else:
            out.append(int(round(float(dist))))
    return np.asarray(out, dtype=np.int64)


def decode_bp_results_with_confidence(results) -> tuple[np.ndarray, np.ndarray]:
    """Decode PyBP probability vectors and return max-marginal confidence."""
    raw = results[0] if isinstance(results, tuple) and len(results) == 2 else results
    values = []
    confidences = []
    for dist in raw:
        if isinstance(dist, (list, tuple, np.ndarray)):
            arr = np.asarray(dist, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                values.append(0)
                confidences.append(0.0)
                continue
            idx = int(np.argmax(arr))
            total = float(np.sum(arr))
            conf = float(arr[idx] / total) if total > 0.0 and np.isfinite(total) else 0.0
            values.append(from_compl(idx, len(arr)))
            confidences.append(conf)
        else:
            values.append(int(round(float(dist))))
            confidences.append(1.0)
    return np.asarray(values, dtype=np.int64), np.asarray(confidences, dtype=np.float64)


def decode_greedy_guess(guess) -> np.ndarray:
    return np.asarray([int(round(float(v))) for v in guess], dtype=np.int64)


# ============================================================
# Secret and ILWE generation
# ============================================================

def sample_sparse_secret(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample s with 0.15n entries in ±1, 0.15n entries in ±2, rest 0."""
    k1 = int(round(0.15 * n))
    k2 = int(round(0.15 * n))
    s = np.zeros(n, dtype=np.int64)
    positions = rng.choice(n, size=k1 + k2, replace=False)
    pos1 = positions[:k1]
    pos2 = positions[k1:]
    s[pos1] = rng.choice([-1, 1], size=k1)
    s[pos2] = rng.choice([-2, 2], size=k2)
    return s


def sample_discrete_gaussian(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.rint(rng.normal(0.0, float(sigma), size=shape)).astype(np.int64)


def sample_uniform_integer(shape, alpha: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(-int(alpha), int(alpha) + 1, size=shape, dtype=np.int64)


def generate_ilwe_instance(
    n: int,
    m: int,
    dist_type: str,
    *,
    sigma_a: float | None,
    sigma_e: float | None,
    alpha_a: int | None,
    alpha_e: int | None,
    seed: int,
):
    """
    Generate exactly m ILWE samples b = A s + e.

    The RNG streams for s, A, and e are separated. Therefore, for a fixed seed,
    a larger m has the same prefix rows as a smaller m.
    """
    if m <= 0:
        raise ValueError("m must be positive")

    rng_s = np.random.default_rng(seed)
    rng_A = np.random.default_rng(seed + 1009)
    rng_e = np.random.default_rng(seed + 2003)

    s = sample_sparse_secret(n, rng_s)

    if dist_type == "dg":
        if sigma_a is None or sigma_e is None:
            raise ValueError("sigma_a and sigma_e are required for dg")
        A = sample_discrete_gaussian((m, n), sigma_a, rng_A)
        e = sample_discrete_gaussian(m, sigma_e, rng_e)
    elif dist_type == "uniform":
        if alpha_a is None or alpha_e is None:
            raise ValueError("alpha_a and alpha_e are required for uniform")
        A = sample_uniform_integer((m, n), alpha_a, rng_A)
        e = sample_uniform_integer(m, alpha_e, rng_e)
    else:
        raise ValueError("dist_type must be 'dg' or 'uniform'")

    b = A @ s + e
    return (
        A.astype(np.int32, copy=False),
        b.astype(np.int64, copy=False),
        s.astype(np.int64, copy=False),
        e.astype(np.int64, copy=False),
    )


# ============================================================
# Distribution hint construction for BP / Greedy-DH
# ============================================================

def _centered_error_support(B: int, max_support: int) -> range:
    if max_support is None or max_support <= 0 or max_support >= 2 * B + 1:
        return range(-B, B + 1)
    half = max_support // 2
    lo = -half
    hi = lo + max_support - 1
    lo = max(lo, -B)
    hi = min(hi, B)
    return range(lo, hi + 1)


def build_rhs_distribution(
    b_i: int,
    a_i: np.ndarray,
    *,
    eta: int,
    error_model: str,
    sigma_e: float | None = None,
    alpha_e: int | None = None,
    tail_sigma: float = 3.0,
    max_rhs_support: int = 0,
    use_inner_bound: bool = True,
) -> Dist:
    """
    Convert b_i = <a_i,s> + e_i into a RHS distribution for <a_i,s>.

    max_rhs_support=0 means use the full truncated support. For large sigma_e,
    full support is extremely expensive. A positive max_rhs_support keeps the
    most likely centered error values only.
    """
    b_i = int(round(float(b_i)))

    if use_inner_bound:
        inner_bound = int(eta * np.sum(np.abs(a_i)))
        ip_min, ip_max = -inner_bound, inner_bound
    else:
        ip_min = ip_max = None

    values: list[tuple[int, float]] = []

    if error_model == "dg":
        if sigma_e is None:
            raise ValueError("sigma_e required for discrete Gaussian error")
        sigma = float(sigma_e)
        B = int(ceil(float(tail_sigma) * sigma))
        for e in _centered_error_support(B, int(max_rhs_support)):
            rhs = b_i - int(e)
            if ip_min is not None and not (ip_min <= rhs <= ip_max):
                continue
            weight = exp(-0.5 * (float(e) / sigma) ** 2)
            values.append((rhs, weight))

    elif error_model == "uniform":
        if alpha_e is None:
            raise ValueError("alpha_e required for uniform error")
        B = int(alpha_e)
        for e in _centered_error_support(B, int(max_rhs_support)):
            rhs = b_i - int(e)
            if ip_min is not None and not (ip_min <= rhs <= ip_max):
                continue
            values.append((rhs, 1.0))
    else:
        raise ValueError("error_model must be 'dg' or 'uniform'")

    if not values:
        rhs = b_i
        if ip_min is not None:
            rhs = min(max(rhs, ip_min), ip_max)
        values = [(rhs, 1.0)]

    return normalize_dist(values)


def build_hints(
    A: np.ndarray,
    b: np.ndarray,
    *,
    eta: int,
    error_model: str,
    sigma_e: float | None = None,
    alpha_e: int | None = None,
    tail_sigma: float = 3.0,
    max_rhs_support: int = 0,
    use_inner_bound: bool = True,
) -> List[Hint]:
    hints: List[Hint] = []
    for a_i, b_i in zip(A, b):
        coeffs = [int(x) for x in a_i]
        D_i = build_rhs_distribution(
            int(b_i),
            a_i,
            eta=eta,
            error_model=error_model,
            sigma_e=sigma_e,
            alpha_e=alpha_e,
            tail_sigma=tail_sigma,
            max_rhs_support=max_rhs_support,
            use_inner_bound=use_inner_bound,
        )
        hints.append((coeffs, D_i))
    return hints


# ============================================================
# Linear / robust / optimization solvers
# ============================================================

def solve_normal_eq(A: np.ndarray, b: np.ndarray, eta: int, lam: float = 1e-10) -> np.ndarray:
    """
    Normal-equation least-squares solver with relative ridge regularization.

    The regularization parameter lam is interpreted as a relative factor:
        ridge = lam * trace(A^T A) / n.

    This keeps the stabilizer on the same scale when n, m, or sigma_a changes.
    Setting lam=0 recovers the unregularized normal equation.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ATA = A.T @ A
    ATb = A.T @ b
    if lam > 0.0:
        scale = float(np.trace(ATA) / max(ATA.shape[0], 1))
        ridge = float(lam) * max(scale, 1.0)
        ATA = ATA + ridge * np.eye(ATA.shape[0], dtype=np.float64)
    x = np.linalg.solve(ATA, ATb)
    return round_clip(x, eta)


def solve_svd(A: np.ndarray, b: np.ndarray, eta: int, rcond: float | None = None) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    U, S, Vh = scipy_svd(A, full_matrices=False, lapack_driver="gesdd")
    if rcond is None:
        rcond = np.finfo(S.dtype).eps * max(m, n)
    tol = S.max() * rcond if S.size else 0.0
    S_inv = np.where(S > tol, 1.0 / S, 0.0)
    x = Vh.T @ (S_inv * (U.T @ b))
    return round_clip(x, eta)


def solve_qr(A: np.ndarray, b: np.ndarray, eta: int, pivoting: bool = True) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    _, n = A.shape
    if pivoting:
        Q, R, piv = scipy_qr(A, mode="economic", pivoting=True)
        x_basic = np.linalg.solve(R, (Q.T @ b)[:n])
        x = np.zeros(n, dtype=np.float64)
        x[piv] = x_basic
    else:
        Q, R = scipy_qr(A, mode="economic", pivoting=False)
        x = np.linalg.solve(R, (Q.T @ b)[:n])
    return round_clip(x, eta)


def solve_l1_lp(A: np.ndarray, b: np.ndarray, eta: int) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    c_obj = np.concatenate([np.zeros(n, dtype=np.float64), np.ones(m, dtype=np.float64)])
    A_sparse = sparse.csr_matrix(A)
    I = sparse.eye(m, dtype=np.float64, format="csr")
    A1 = sparse.hstack([A_sparse, -I], format="csr")
    A2 = sparse.hstack([-A_sparse, -I], format="csr")
    A_ub = sparse.vstack([A1, A2], format="csr")
    b_ub = np.concatenate([b, -b])
    bounds = [(-eta, eta)] * n + [(0.0, None)] * m
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"L1-LP failed: {res.message}")
    return round_clip(res.x[:n], eta)


def _huber_value_grad(x: np.ndarray, A: np.ndarray, b: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    r = A @ x - b
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.empty_like(r, dtype=np.float64)
    loss[mask] = 0.5 * r[mask] ** 2
    loss[~mask] = delta * (abs_r[~mask] - 0.5 * delta)
    grad_r = np.empty_like(r, dtype=np.float64)
    grad_r[mask] = r[mask]
    grad_r[~mask] = delta * np.sign(r[~mask])
    return float(np.sum(loss)), A.T @ grad_r


def solve_huber(A: np.ndarray, b: np.ndarray, eta: int, delta: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = A.shape[1]
    x0 = round_clip(lsqr(sparse.csr_matrix(A), b)[0], eta).astype(np.float64)
    bounds = [(-eta, eta)] * n

    def fun(x):
        return _huber_value_grad(x, A, b, delta)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Huber failed: {res.message}")
    return round_clip(res.x, eta)


def _regression_weighted_linear_coefficients(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    lr,
) -> np.ndarray:
    """Return sklearn LinearRegression(...).coef_ semantics, with a NumPy fallback."""
    if lr is not None:
        lr.fit(A, b, sample_weight=weights)
        return np.asarray(lr.coef_, dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0 or not np.isfinite(weights_sum):
        raise RuntimeError("Linear-regression weights became invalid")
    norm_w = weights / weights_sum
    x_mean = norm_w @ A
    y_mean = float(norm_w @ b)
    sqrt_w = np.sqrt(norm_w)
    Aw = (A - x_mean) * sqrt_w[:, None]
    bw = (b - y_mean) * sqrt_w
    coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    return np.asarray(coef, dtype=np.float64)


def solve_cauchy(
    A: np.ndarray,
    b: np.ndarray,
    eta: int,
    scale: float | None = None,
    max_iter: int = 100,
    convergence_eps: float = 0.01,
    convergence_min_run: int = 10,
    true_x: np.ndarray | None = None,
) -> np.ndarray:
    """
    Cauchy IRLS matching E:\\PythonProject\\regression\\regression.py.

    This intentionally uses sklearn's weighted LinearRegression and the
    regression prototype's weight update:
        w_i = 1 / (1 + r_i^2)

    The scale argument is accepted for compatibility with existing CLWE_Solve
    call sites, but is not used by this regression-style implementation.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    lr = LinearRegression(n_jobs=1) if LinearRegression is not None else None
    weights = np.ones(m, dtype=np.float64) / max(m, 1)
    beta_est = np.zeros(n, dtype=np.float64)
    last_estimate = np.zeros(n, dtype=np.float64)
    true_arr = None if true_x is None else np.asarray(true_x, dtype=np.int64).reshape(-1)
    convergence_counter = 0

    for _ in range(int(max_iter)):
        beta_est = _regression_weighted_linear_coefficients(A, b, weights, lr)

        residuals = b - A @ beta_est
        weights = 1.0 / (1.0 + residuals**2)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            raise RuntimeError("Cauchy weights became invalid")
        weights /= weights_sum

        if true_arr is not None and np.sum(true_arr == np.round(beta_est)) == len(true_arr):
            break

        converged = np.max(beta_est - last_estimate) < convergence_eps
        if converged:
            convergence_counter += 1
        else:
            convergence_counter = 0

        if convergence_counter >= convergence_min_run:
            break

    return np.asarray(np.round(beta_est), dtype=np.int64)


def solve_gd_mle(
    A: np.ndarray,
    b: np.ndarray,
    eta: int,
    max_iter: int = 5000,
    tol: float = 1e-5,
    use_bounds: bool = True,
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = A.shape[1]
    x = np.zeros(n, dtype=np.float64)

    # Lipschitz upper bound for grad 0.5||Ax-b||^2 is ||A||_2^2 <= ||A||_F^2.
    L = float(np.linalg.norm(A, ord="fro") ** 2)
    step = 1.0 / max(L, 1e-12)

    for _ in range(int(max_iter)):
        grad = A.T @ (A @ x - b)
        x_new = x - step * grad
        if use_bounds:
            x_new = np.clip(x_new, -eta, eta)
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return round_clip(x, eta)


def solve_ilp(A: np.ndarray, b: np.ndarray, eta: int, time_limit: float | None = None) -> np.ndarray:
    if milp is None or Bounds is None or LinearConstraint is None:
        raise RuntimeError("scipy.optimize.milp is unavailable in this SciPy version")

    A = sparse.csr_matrix(np.asarray(A, dtype=np.float64))
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    c = np.concatenate([np.zeros(n, dtype=np.float64), np.ones(m, dtype=np.float64)])
    I = sparse.eye(m, dtype=np.float64, format="csr")

    # A x - u <= b ; -A x - u <= -b
    C1 = sparse.hstack([A, -I], format="csr")
    C2 = sparse.hstack([-A, -I], format="csr")
    C = sparse.vstack([C1, C2], format="csr")
    lb_cons = -np.inf * np.ones(2 * m, dtype=np.float64)
    ub_cons = np.concatenate([b, -b])
    constraints = LinearConstraint(C, lb_cons, ub_cons)

    lb = np.concatenate([-eta * np.ones(n), np.zeros(m)])
    ub = np.concatenate([eta * np.ones(n), np.inf * np.ones(m)])
    bounds = Bounds(lb, ub)
    integrality = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(m, dtype=np.int8)])
    options = {"disp": False}
    if time_limit is not None and time_limit > 0:
        options["time_limit"] = float(time_limit)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
    if not res.success:
        raise RuntimeError(f"ILP failed: {res.message}")
    return round_clip(res.x[:n], eta)


# ============================================================
# Distribution-hint solvers
# ============================================================

def secret_prior_for_sparse_eta2(eta: int) -> list[float]:
    if eta == 2:
        # complement order: [0, 1, 2, -2, -1]
        return [0.20, 0.20, 0.20, 0.20, 0.20]
    if eta == 4:
        # complement order: [0, 1, 2, 3, 4, -4, -3, -2, -1]
        return [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
    return [1.0] * (2 * eta + 1)
     

def _bp_message_size() -> int:
    cached = getattr(_bp_message_size, "_cached", None)
    if cached is not None:
        return int(cached)
    if PyBP is None:
        return 2
    probe = PyBP([[1]], [[(0, 1.0)]])
    size = len(probe.get_prior())
    setattr(_bp_message_size, "_cached", int(size))
    return int(size)


def secret_prior_for_eta_message_size(eta: int, sz_msg: int) -> list[float]:
    """Prior matching the compiled hint_solver complement-message size."""
    eta = int(eta)
    sz_msg = int(sz_msg)
    eps = 1e-6
    return [1.0 if abs(from_compl(idx, sz_msg)) <= eta else eps for idx in range(sz_msg)]


def _required_bp_chk_size(coeffs: list[list[int]], rhs_dists: list[Dist], eta: int) -> int:
    """Choose a check FFT size large enough for RHS shifts and variable sums."""
    max_abs = 1
    eta = int(eta)
    for row, dist in zip(coeffs, rhs_dists):
        coeff_bound = int(sum(abs(int(c)) for c in row) * eta)
        rhs_bound = max((abs(int(value)) for value, _prob in dist), default=0)
        max_abs = max(max_abs, coeff_bound + rhs_bound + 1)
    return max(1, int(math.ceil(math.log2(max_abs + 1))))


def _make_bp(hints: List[Hint], eta: int, *, threads: int | None, use_sparse_prior: bool):
    if PyBP is None:
        raise RuntimeError(f"PyBP unavailable: {HINT_SOLVER_IMPORT_ERROR!r}")
    coeffs, rhs_dists = zip(*hints)
    coeffs = list(coeffs)
    rhs_dists = list(rhs_dists)
    sz_msg = _bp_message_size()
    prior = secret_prior_for_eta_message_size(eta, sz_msg) if use_sparse_prior else [1.0] * sz_msg
    chk_size = _required_bp_chk_size(coeffs, rhs_dists, eta)

    bp = None
    last_exc = None
    for make_bp in (
        lambda: PyBP(coeffs, rhs_dists, chk_size, prior),
        lambda: PyBP(coeffs, rhs_dists, None, prior),
        lambda: PyBP(coeffs, rhs_dists, chk_size),
        lambda: PyBP(coeffs, rhs_dists),
    ):
        try:
            bp = make_bp()
            break
        except TypeError as exc:
            last_exc = exc
    if bp is None:
        raise RuntimeError(f"Could not construct PyBP: {last_exc}")
    if threads is not None and hasattr(bp, "set_nthreads"):
        bp.set_nthreads(int(threads))
    return bp


def _run_bp_raw(
    hints: List[Hint],
    eta: int,
    *,
    max_iter: int,
    threads: int | None,
    use_sparse_prior: bool,
):
    bp = _make_bp(hints, eta, threads=threads, use_sparse_prior=use_sparse_prior)

    raw_results = None
    for _ in range(int(max_iter)):
        bp.propagate()
        raw_results = bp.get_results()
    return raw_results


def solve_bp(hints: List[Hint], eta: int, *, max_iter: int, threads: int | None, use_sparse_prior: bool) -> np.ndarray:
    raw_results = _run_bp_raw(
        hints,
        eta,
        max_iter=max_iter,
        threads=threads,
        use_sparse_prior=use_sparse_prior,
    )
    x_hat = decode_bp_results(raw_results)
    return np.clip(np.asarray(x_hat, dtype=np.int64), -eta, eta)


def _reduced_hints_for_fixed_values(
    hints: List[Hint],
    active_indices: list[int],
    fixed_values: dict[int, int],
) -> List[Hint]:
    reduced_hints: List[Hint] = []
    for coeffs, rhs_dist in hints:
        offset = 0
        for var_idx, value in fixed_values.items():
            offset += int(coeffs[var_idx]) * int(value)

        reduced_coeffs = [int(coeffs[idx]) for idx in active_indices]
        if not any(reduced_coeffs):
            continue

        shifted_rhs = [(int(rhs) - offset, float(prob)) for rhs, prob in rhs_dist]
        reduced_hints.append((reduced_coeffs, normalize_dist(shifted_rhs)))

    return reduced_hints


def solve_bp_decimation(
    hints: List[Hint],
    eta: int,
    *,
    max_iter: int,
    threads: int | None,
    use_sparse_prior: bool,
    rounds: int = 8,
    threshold: float = 0.995,
    fraction: float = 0.10,
    min_fix: int = 1,
) -> np.ndarray:
    """
    Belief-propagation decimation.

    Repeatedly run BP, fix high-confidence variables, subtract their
    contribution from all hints, and rerun BP on the remaining variables.
    """
    if not hints:
        return np.empty(0, dtype=np.int64)

    n = len(hints[0][0])
    active_indices = list(range(n))
    fixed_values: dict[int, int] = {}
    best = np.zeros(n, dtype=np.int64)

    rounds = max(1, int(rounds))
    threshold = float(threshold)
    fraction = float(fraction)
    min_fix = max(1, int(min_fix))

    for _ in range(rounds):
        if not active_indices:
            break

        reduced_hints = _reduced_hints_for_fixed_values(hints, active_indices, fixed_values)
        if not reduced_hints:
            break

        raw_results = _run_bp_raw(
            reduced_hints,
            eta,
            max_iter=max_iter,
            threads=threads,
            use_sparse_prior=use_sparse_prior,
        )
        active_guess, confidence = decode_bp_results_with_confidence(raw_results)
        active_guess = np.clip(active_guess, -eta, eta)

        for local_idx, global_idx in enumerate(active_indices):
            best[global_idx] = int(active_guess[local_idx])

        eligible = np.where(confidence >= threshold)[0]
        if eligible.size == 0:
            break

        order = eligible[np.argsort(-confidence[eligible])]
        max_to_fix = max(min_fix, int(math.ceil(len(active_indices) * max(0.0, fraction))))
        selected = order[:max_to_fix]

        for local_idx in selected:
            global_idx = active_indices[int(local_idx)]
            fixed_values[global_idx] = int(active_guess[int(local_idx)])

        fixed_set = set(fixed_values)
        active_indices = [idx for idx in active_indices if idx not in fixed_set]

    return np.clip(best, -eta, eta).astype(np.int64)


def solve_greedy_dh(
    hints: List[Hint],
    eta: int,
    *,
    max_iter: int,
    threads: int | None,
    kappa_mode: str,
    fixed_kappa: int | None,
) -> np.ndarray:
    if PyGreedy is None:
        raise RuntimeError(f"PyGreedy unavailable: {HINT_SOLVER_IMPORT_ERROR!r}")
    coeffs, rhs_dists = zip(*hints)
    coeffs = list(coeffs)
    rhs_dists = list(rhs_dists)
    solver = PyGreedy(coeffs, rhs_dists)
    if threads is not None and hasattr(solver, "set_nthreads"):
        solver.set_nthreads(int(threads))

    n = len(coeffs[0])
    for it in range(int(max_iter)):
        if fixed_kappa is not None:
            kappa = max(1, int(fixed_kappa))
        elif kappa_mode == "decay":
            kappa = max(1, n // (2 ** min(it, 8)))
        elif kappa_mode == "cycle":
            kappa = max(1, n // (2 ** (it % 8)))
        elif kappa_mode == "small":
            kappa = max(1, n // 32)
        elif kappa_mode == "one":
            kappa = 1
        else:
            raise ValueError(f"unknown kappa_mode: {kappa_mode}")
        try:
            solver.solve(int(kappa))
        except TypeError:
            solver.solve()
    return np.clip(decode_greedy_guess(solver.get_guess()), -eta, eta)


# ============================================================
# Residual-L2 Greedy and Hillclimb solvers
# ============================================================

def solve_greedy_l2(
    A: np.ndarray,
    b: np.ndarray,
    eta: int,
    *,
    max_iter: int = 100,
    k: int | None = None,
    init: str = "zeros",
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    values = np.arange(-eta, eta + 1, dtype=np.float64)

    if init == "normal_eq" and m >= n:
        try:
            x = solve_normal_eq(A, b, eta).astype(np.float64)
        except Exception:
            x = np.zeros(n, dtype=np.float64)
    else:
        x = np.zeros(n, dtype=np.float64)

    if k is None:
        k = max(1, n // 16)
    col_norm2 = np.sum(A * A, axis=0)
    r = A @ x - b

    for _ in range(int(max_iter)):
        best_delta = np.zeros(n, dtype=np.float64)
        best_improve = np.zeros(n, dtype=np.float64)
        At_r = A.T @ r
        for j in range(n):
            cur = x[j]
            deltas = values - cur
            # improvement = old_loss - new_loss
            improves = -(2.0 * deltas * At_r[j] + (deltas ** 2) * col_norm2[j])
            idx = int(np.argmax(improves))
            if improves[idx] > 1e-9 and deltas[idx] != 0:
                best_delta[j] = deltas[idx]
                best_improve[j] = improves[idx]
        valid = np.where(best_improve > 0)[0]
        if valid.size == 0:
            break
        chosen = valid[np.argsort(-best_improve[valid])[:k]]
        if chosen.size == 0:
            break
        for j in chosen:
            d = best_delta[j]
            x[j] += d
            x[j] = np.clip(x[j], -eta, eta)
            r += A[:, j] * d
    return round_clip(x, eta)



def _hillclimb_fitness_scalar(
    ip: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fitness_mode: str,
    fitness_lambda: float,
) -> Tuple[float, int]:
    """
    Interval-verification fitness used by the ML-DSA hill-climbing code,
    adapted to standard ILWE constraints:

        lb_i <= <a_i, x> <= ub_i,

    where lb_i = b_i - B and ub_i = b_i + B.

    fitness_mode:
        count    : number of violated relations
        excess   : sum of interval excesses
        combined : lambda * count + excess
    """
    violated_low = ip < lb
    violated_high = ip > ub
    violated = violated_low | violated_high
    F = int(np.count_nonzero(violated))

    if fitness_mode == "count":
        return float(F), F

    excess = np.maximum(lb - ip, 0.0) + np.maximum(ip - ub, 0.0)
    S = float(np.sum(excess))

    if fitness_mode == "excess":
        return S, F

    if fitness_mode == "combined":
        return float(fitness_lambda) * F + S, F

    raise ValueError(f"unknown hillclimb fitness mode: {fitness_mode}")


def _hillclimb_fitness_batch(
    ip_batch: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fitness_mode: str,
    fitness_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch version of _hillclimb_fitness_scalar.

    ip_batch has shape (m, num_candidates).
    """
    violated = (ip_batch < lb[:, None]) | (ip_batch > ub[:, None])
    F_counts = np.count_nonzero(violated, axis=0)

    if fitness_mode == "count":
        return F_counts.astype(np.float64), F_counts

    excess = np.maximum(lb[:, None] - ip_batch, 0.0) + np.maximum(ip_batch - ub[:, None], 0.0)
    S_vals = np.sum(excess, axis=0)

    if fitness_mode == "excess":
        return S_vals, F_counts

    if fitness_mode == "combined":
        return float(fitness_lambda) * F_counts + S_vals, F_counts

    raise ValueError(f"unknown hillclimb fitness mode: {fitness_mode}")


def _hillclimb_precompute_candidates(values: np.ndarray, w: int) -> np.ndarray:
    """Precompute all (2*eta+1)^w candidate tuples."""
    grids = np.indices((len(values),) * int(w)).reshape(int(w), -1).T
    return values[grids].astype(np.int64)


def _hillclimb_best_candidate(
    A_block: np.ndarray,
    ip_base: np.ndarray,
    candidate_tuples: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    fitness_mode: str,
    fitness_lambda: float,
    *,
    chunk_size: int = 1024,
) -> Tuple[int, float, int]:
    """
    Evaluate candidate assignments in chunks and return:
        best_index, best_fitness, best_violation_count.
    """
    best_idx = 0
    best_fit = np.inf
    best_F = 0

    cand_f64 = candidate_tuples.astype(np.float64, copy=False)
    for st in range(0, cand_f64.shape[0], int(chunk_size)):
        ed = min(st + int(chunk_size), cand_f64.shape[0])
        ip_new = ip_base[:, None] + A_block @ cand_f64[st:ed].T
        fit_vals, F_vals = _hillclimb_fitness_batch(
            ip_new, lb, ub, fitness_mode, fitness_lambda
        )
        local = int(np.argmin(fit_vals))
        if float(fit_vals[local]) < best_fit:
            best_idx = st + local
            best_fit = float(fit_vals[local])
            best_F = int(F_vals[local])

    return best_idx, best_fit, best_F


def _hillclimb_make_score_weights(x_float: np.ndarray, eta: int, temperature: float = 2.0) -> np.ndarray:
    """
    Score-guided sampling weights, following the ML-DSA hill-climbing idea:
    coordinates whose continuous warm-start estimate is far from any legal
    integer are treated as less certain and sampled more often.
    """
    vals = np.arange(-int(eta), int(eta) + 1, dtype=np.float64)
    dist = np.min(np.abs(np.asarray(x_float, dtype=np.float64)[:, None] - vals[None, :]), axis=1)
    logits = float(temperature) * dist
    logits -= np.max(logits)
    w = np.exp(logits)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(w) / len(w)
    return w / total


def _hillclimb_diversified_weights(base_weights: np.ndarray, freq_counts: np.ndarray, strength: float) -> np.ndarray:
    penalty = 1.0 + float(strength) * freq_counts.astype(np.float64)
    w = base_weights / penalty
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(w) / len(w)
    return w / total


def solve_hillclimb(
    A: np.ndarray,
    b: np.ndarray,
    eta: int,
    *,
    error_bound: float,
    max_iter: int = 2000,
    block_size: int = 2,
    seed: int = 0,
    init: str = "normal_eq",
    patience: int = 300,
    fitness_mode: str = "excess",
    fitness_lambda: float | None = None,
    adaptive_w: bool = True,
    adaptive_w_max: int = 4,
    adaptive_w_patience: int = 50,
    lateral_moves: bool = True,
    diversify: bool = True,
    diversify_strength: float = 1.0,
    sweep_interval: int = 0,
    perturb_restart: bool = True,
    perturb_strength: int = 30,
    perturb_patience: int = 100,
    perturb_max: int = 100,
    sequential_w: bool = True,
    score_guided: bool = False,
    score_temperature: float = 2.0,
    candidate_chunk_size: int = 1024,
) -> np.ndarray:
    """
    Hill-climbing solver for standard ILWE, adapted from the ML-DSA
    verification-fitness hill-climbing framework.

    Standard ILWE relation:
        b_i = <a_i, s> + e_i.

    The hill-climber converts it to interval constraints:
        b_i - B <= <a_i, x> <= b_i + B,

    where B = error_bound. It then searches over x_j in {-eta,...,eta}
    and minimizes an interval fitness:
        count    : number of violated constraints,
        excess   : sum of interval overshoots,
        combined : lambda * count + excess.

    Compared with the earlier simple residual-L2 hillclimber, this version
    follows the reference ML-DSA implementation more closely:
        - block candidate enumeration;
        - adaptive block size;
        - optional lateral moves;
        - frequency diversification;
        - perturb-restart / ILS;
        - optional w=1 sweep preamble;
        - best-ever solution tracking.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    rng = np.random.default_rng(seed)

    eta = int(eta)
    values = np.arange(-eta, eta + 1, dtype=np.int64)

    B = float(error_bound)
    if B < 0:
        raise ValueError("error_bound must be non-negative")
    lb = b - B
    ub = b + B

    if fitness_lambda is None or fitness_lambda <= 0:
        fitness_lambda = max(B, 1.0)

    # -----------------------
    # Warm start
    # -----------------------
    x_float_for_score = np.zeros(n, dtype=np.float64)

    if init in {"normal_eq", "ols", "lsqr"} and m >= n:
        try:
            # Use the continuous normal-equation/ridge estimate for both
            # the integer warm start and score-guided coordinate sampling.
            A_f = np.asarray(A, dtype=np.float64)
            ATA = A_f.T @ A_f
            ridge = 1e-10 * max(float(np.trace(ATA) / max(n, 1)), 1.0)
            ATA = ATA + ridge * np.eye(n)
            ATb = A_f.T @ b
            x_float_for_score = np.linalg.solve(ATA, ATb)
            x = round_clip(x_float_for_score, eta)
        except Exception:
            x = np.zeros(n, dtype=np.int64)
    else:
        x = np.zeros(n, dtype=np.int64)

    ip = A @ x.astype(np.float64)
    fitness_curr, F_curr = _hillclimb_fitness_scalar(ip, lb, ub, fitness_mode, float(fitness_lambda))

    best_x = x.copy()
    best_ip = ip.copy()
    best_fit = fitness_curr
    best_F = F_curr

    # -----------------------
    # Candidate cache
    # -----------------------
    base_w = max(1, min(int(block_size), n))
    w_adaptive_start = max(base_w, 2) if sequential_w else base_w
    w_curr = 1 if sequential_w else w_adaptive_start
    in_w1_sweep = bool(sequential_w)

    cand_cache: dict[int, np.ndarray] = {}

    def get_candidates(w: int) -> np.ndarray:
        w = int(w)
        if w not in cand_cache:
            cand_cache[w] = _hillclimb_precompute_candidates(values, w)
        return cand_cache[w]

    # -----------------------
    # Sampling state
    # -----------------------
    if score_guided:
        base_weights = _hillclimb_make_score_weights(
            x_float_for_score, eta, temperature=score_temperature
        )
    else:
        base_weights = np.ones(n, dtype=np.float64) / n

    freq_counts = np.zeros(n, dtype=np.int64)
    sweep_perm = None
    sweep_offset = 0
    iters_since_improve = 0
    iters_since_best = 0
    num_perturb = 0

    # -----------------------
    # Main loop
    # -----------------------
    for _it in range(1, int(max_iter) + 1):
        if F_curr == 0:
            break
        if iters_since_best >= int(patience):
            break

        iters_since_best += 1

        # ---- perturb restart / ILS ----
        if (
            perturb_restart
            and not in_w1_sweep
            and iters_since_improve >= int(perturb_patience)
            and num_perturb < int(perturb_max)
        ):
            adaptive_exhausted = (not adaptive_w) or (w_curr >= min(int(adaptive_w_max), n))
            if adaptive_exhausted:
                # Restore best-ever before perturbing.
                x = best_x.copy()
                ip = best_ip.copy()
                fitness_curr = best_fit
                F_curr = best_F

                p = min(int(perturb_strength), n)
                pos = rng.choice(n, size=p, replace=False)
                old = x[pos].astype(np.float64)
                x[pos] = rng.integers(-eta, eta + 1, size=p, dtype=np.int64)
                ip = ip + A[:, pos] @ (x[pos].astype(np.float64) - old)

                fitness_curr, F_curr = _hillclimb_fitness_scalar(
                    ip, lb, ub, fitness_mode, float(fitness_lambda)
                )
                num_perturb += 1
                iters_since_improve = 0
                freq_counts[:] = 0
                if sequential_w:
                    in_w1_sweep = True
                    w_curr = 1
                else:
                    w_curr = w_adaptive_start

        # ---- position selection ----
        if in_w1_sweep:
            # Reference-style w=1 sweep: test all coordinates independently
            # against a frozen snapshot, then apply all individually improving
            # moves and verify that global fitness did not degrade.
            ip_frozen = ip.copy()
            x_pre = x.copy()
            ip_pre = ip.copy()
            fit_pre = fitness_curr
            F_pre = F_curr
            changed = 0

            one_cands = get_candidates(1)
            for j in range(n):
                a_j = A[:, j]
                ip_base_j = ip_frozen - a_j * float(x[j])
                idx, fit_j, _F_j = _hillclimb_best_candidate(
                    a_j.reshape(-1, 1),
                    ip_base_j,
                    one_cands,
                    lb,
                    ub,
                    fitness_mode,
                    float(fitness_lambda),
                    chunk_size=candidate_chunk_size,
                )
                v = int(one_cands[idx, 0])
                if fit_j < fitness_curr and v != int(x[j]):
                    x[j] = v
                    changed += 1

            if changed > 0:
                ip = A @ x.astype(np.float64)
                fitness_new, F_new = _hillclimb_fitness_scalar(
                    ip, lb, ub, fitness_mode, float(fitness_lambda)
                )
                # Avoid simultaneous independent updates making the global
                # state worse; this mirrors the sanity-check idea in the
                # uploaded ML-DSA reference code.
                if fitness_new > fit_pre:
                    x = x_pre
                    ip = ip_pre
                    fitness_curr = fit_pre
                    F_curr = F_pre
                    changed = 0
                else:
                    fitness_curr = fitness_new
                    F_curr = F_new

            if changed > 0 and fitness_curr < best_fit:
                best_fit = fitness_curr
                best_F = F_curr
                best_x = x.copy()
                best_ip = ip.copy()
                iters_since_best = 0
                iters_since_improve = 0
                in_w1_sweep = True
                w_curr = 1
                continue

            # no w=1 progress; move to adaptive block search
            in_w1_sweep = False
            w_curr = w_adaptive_start
            iters_since_improve = 0
            continue

        elif sweep_interval > 0 and (_it % int(sweep_interval)) == 0:
            if sweep_perm is None or sweep_offset >= n:
                sweep_perm = rng.permutation(n)
                sweep_offset = 0
            ed = min(sweep_offset + int(w_curr), n)
            positions = sweep_perm[sweep_offset:ed]
            sweep_offset = ed
        else:
            if diversify:
                weights = _hillclimb_diversified_weights(
                    base_weights, freq_counts, diversify_strength
                )
            else:
                weights = base_weights
            positions = rng.choice(n, size=int(w_curr), replace=False, p=weights)

        positions = np.asarray(positions, dtype=int)
        freq_counts[positions] += 1

        # ---- candidate evaluation ----
        actual_w = len(positions)
        cands = get_candidates(actual_w)
        A_block = A[:, positions]
        old_vals = x[positions].astype(np.float64)
        ip_base = ip - A_block @ old_vals

        best_idx, cand_fit, cand_F = _hillclimb_best_candidate(
            A_block,
            ip_base,
            cands,
            lb,
            ub,
            fitness_mode,
            float(fitness_lambda),
            chunk_size=candidate_chunk_size,
        )

        strict = cand_fit < fitness_curr
        lateral = (not strict) and lateral_moves and (cand_fit == fitness_curr)
        accepted = strict or lateral

        if accepted:
            new_vals = cands[best_idx].astype(np.int64)
            if not np.array_equal(new_vals, x[positions]):
                x[positions] = new_vals
                ip = ip_base + A_block @ new_vals.astype(np.float64)
                fitness_curr = float(cand_fit)
                F_curr = int(cand_F)
            else:
                accepted = False

        # ---- stagnation / adaptive block size ----
        if strict:
            iters_since_improve = 0
            if fitness_curr < best_fit:
                best_fit = fitness_curr
                best_F = F_curr
                best_x = x.copy()
                best_ip = ip.copy()
                iters_since_best = 0

            if sequential_w:
                in_w1_sweep = True
                w_curr = 1
            elif adaptive_w and w_curr != w_adaptive_start:
                w_curr = w_adaptive_start
        else:
            iters_since_improve += 1

        if adaptive_w and (not in_w1_sweep) and iters_since_improve >= int(adaptive_w_patience):
            new_w = min(int(w_curr) + 1, int(adaptive_w_max), n)
            if new_w != w_curr:
                w_curr = new_w
                iters_since_improve = 0

    return np.clip(best_x, -eta, eta).astype(np.int64)



# ============================================================
# Distribution-aware solver parameter selection
# ============================================================

def error_std_from_args(args) -> float:
    """
    Effective standard deviation of the ILWE error e.

    For e ~ D_{sigma_e}, sigma_eff = sigma_e.
    For e uniform over [-alpha_e, alpha_e] ∩ Z,
        Var(e) = alpha_e(alpha_e+1)/3.
    """
    if args.dist == "dg":
        return float(args.sigma_e)
    if args.dist == "uniform":
        alpha = float(args.alpha_e)
        return float(np.sqrt(alpha * (alpha + 1.0) / 3.0))
    raise ValueError(f"unknown distribution: {args.dist}")


def resolve_huber_delta(args) -> float:
    """
    Huber threshold delta.

    If --huber-delta > 0, use the user-provided value.
    Otherwise use delta = 1.345 * sigma_eff.
    """
    if args.huber_delta > 0:
        return float(args.huber_delta)
    return 1.345 * error_std_from_args(args)


def resolve_cauchy_scale(args) -> float:
    """
    Cauchy/IRLS scale parameter.

    If --cauchy-scale > 0, use the user-provided value.
    Otherwise use scale = 2.385 * sigma_eff.
    """
    if args.cauchy_scale > 0:
        return float(args.cauchy_scale)
    return 2.385 * error_std_from_args(args)


def resolve_hillclimb_error_bound(args) -> float:
    """
    Interval bound B for hillclimb:
        b_i - B <= <a_i, x> <= b_i + B.

    Uniform errors have strict bound B = alpha_e.
    Discrete Gaussian errors are truncated at ceil(tail_sigma * sigma_e).
    """
    if args.hillclimb_error_bound > 0:
        return float(args.hillclimb_error_bound)
    if args.dist == "uniform":
        return float(args.alpha_e)
    if args.dist == "dg":
        return float(ceil(float(args.tail_sigma) * float(args.sigma_e)))
    raise ValueError(f"unknown distribution: {args.dist}")

# ============================================================
# Evaluation and binary search
# ============================================================

def solve_instance(solver_name: str, A: np.ndarray, b: np.ndarray, s: np.ndarray, args, *, key_index: int = 0):
    t0 = perf_counter()
    try:
        if solver_name == "normal_eq":
            x_hat = solve_normal_eq(A, b, args.eta, lam=args.normal_lam)
        elif solver_name == "svd":
            x_hat = solve_svd(A, b, args.eta)
        elif solver_name == "qr":
            x_hat = solve_qr(A, b, args.eta, pivoting=not args.qr_no_pivot)
        elif solver_name == "l1_lp":
            x_hat = solve_l1_lp(A, b, args.eta)
        elif solver_name == "huber":
            delta = resolve_huber_delta(args)
            x_hat = solve_huber(A, b, args.eta, delta=delta)
        elif solver_name == "cauchy":
            x_hat = solve_cauchy(A, b, args.eta, max_iter=args.cauchy_iter, true_x=s)
        elif solver_name == "gd_mle":
            x_hat = solve_gd_mle(A, b, args.eta, max_iter=args.gd_iter, tol=args.gd_tol, use_bounds=not args.gd_no_bounds)
        elif solver_name == "ilp":
            if args.ilp_max_m > 0 and A.shape[0] > args.ilp_max_m:
                raise RuntimeError(f"skip ILP for m={A.shape[0]} > ilp_max_m={args.ilp_max_m}")
            x_hat = solve_ilp(A, b, args.eta, time_limit=args.ilp_time_limit)
        elif solver_name in {"bp", "greedy_dh"}:
            if args.hint_max_m > 0 and A.shape[0] > args.hint_max_m:
                raise RuntimeError(f"skip {solver_name} for m={A.shape[0]} > hint_max_m={args.hint_max_m}")
            hints = build_hints(
                A,
                b,
                eta=args.eta,
                error_model=args.dist,
                sigma_e=args.sigma_e,
                alpha_e=args.alpha_e,
                tail_sigma=args.tail_sigma,
                max_rhs_support=args.max_rhs_support,
                use_inner_bound=not args.no_inner_bound,
            )
            if solver_name == "bp":
                x_hat = solve_bp(hints, args.eta, max_iter=args.bp_iter, threads=args.threads, use_sparse_prior=not args.uniform_prior)
            else:
                x_hat = solve_greedy_dh(
                    hints,
                    args.eta,
                    max_iter=args.greedy_dh_iter,
                    threads=args.threads,
                    kappa_mode=args.greedy_kappa_mode,
                    fixed_kappa=args.greedy_kappa,
                )
        elif solver_name == "greedy_l2":
            x_hat = solve_greedy_l2(A, b, args.eta, max_iter=args.greedy_l2_iter, k=args.greedy_l2_k, init=args.greedy_l2_init)
        elif solver_name == "hillclimb":
            hc_bound = resolve_hillclimb_error_bound(args)

            x_hat = solve_hillclimb(
                A,
                b,
                args.eta,
                error_bound=hc_bound,
                max_iter=args.hillclimb_iter,
                block_size=args.hillclimb_block_size,
                seed=args.seed + 5557 * key_index + A.shape[0],
                init=args.hillclimb_init,
                patience=args.hillclimb_patience,
                fitness_mode=args.hillclimb_fitness,
                fitness_lambda=args.hillclimb_lambda,
                adaptive_w=not args.hillclimb_no_adaptive_w,
                adaptive_w_max=args.hillclimb_adaptive_w_max,
                adaptive_w_patience=args.hillclimb_adaptive_w_patience,
                lateral_moves=not args.hillclimb_no_lateral,
                diversify=not args.hillclimb_no_diversify,
                diversify_strength=args.hillclimb_diversify_strength,
                sweep_interval=args.hillclimb_sweep_interval,
                perturb_restart=not args.hillclimb_no_perturb,
                perturb_strength=args.hillclimb_perturb_strength,
                perturb_patience=args.hillclimb_perturb_patience,
                perturb_max=args.hillclimb_perturb_max,
                sequential_w=not args.hillclimb_no_sequential_w,
                score_guided=args.hillclimb_score_guided,
                score_temperature=args.hillclimb_score_temperature,
                candidate_chunk_size=args.hillclimb_candidate_chunk,
            )
        else:
            raise ValueError(f"unknown solver: {solver_name}")
        elapsed = perf_counter() - t0
        recovered = int(np.sum(x_hat == s))
        success = recovered == len(s)
        return success, recovered, elapsed, None
    except Exception as exc:
        elapsed = perf_counter() - t0
        return False, 0, elapsed, exc


def success_at_m(solver_name: str, instance_seeds: list[int], m: int, args):
    successes = 0
    recovered_list: list[int] = []
    times: list[float] = []
    errors: list[str] = []

    for idx, inst_seed in enumerate(instance_seeds):
        A, b, s, e = generate_ilwe_instance(
            n=args.n,
            m=m,
            dist_type=args.dist,
            sigma_a=args.sigma_a,
            sigma_e=args.sigma_e,
            alpha_a=args.alpha_a,
            alpha_e=args.alpha_e,
            seed=inst_seed,
        )
        ok, rec, elapsed, err = solve_instance(solver_name, A, b, s, args, key_index=idx)
        if ok:
            successes += 1
        recovered_list.append(rec)
        times.append(elapsed)
        status = "Success" if ok else "Failure"
        msg = f"[{solver_name}] m={m}, key={idx + 1}/{len(instance_seeds)}, {status}, recovered={rec}/{len(s)}, time={elapsed:.2f}s"
        if err is not None:
            msg += f", error={err}"
            errors.append(str(err))
        print(msg, flush=True)
        del A, b, s, e
        gc.collect()

    need = args.success_keys if args.success_keys is not None else len(instance_seeds)
    return successes >= need, successes, recovered_list, times, errors


def find_min_m_for_solver(solver_name: str, instance_seeds: list[int], args):
    low = 0
    high = int(args.start_m)
    found = False
    max_m = int(args.max_m)
    print(f"\n=== Searching minimal m for {solver_name} ===", flush=True)

    while high <= max_m:
        ok, succ, recs, times, errors = success_at_m(solver_name, instance_seeds, high, args)
        if ok:
            found = True
            break
        low = high
        high *= 2

    if not found:
        print(f"[{solver_name}] No success found up to max_m={max_m}", flush=True)
        return None

    resolution = max(1, int(args.resolution))
    while high - low > resolution:
        mid = ((low + high) // 2 // resolution) * resolution
        if mid <= low:
            mid = low + resolution
        if mid >= high:
            mid = high - resolution
        if mid <= low or mid >= high:
            break
        ok, succ, recs, times, errors = success_at_m(solver_name, instance_seeds, mid, args)
        if ok:
            high = mid
        else:
            low = mid

    print(f"[{solver_name}] minimal m = {high} with resolution={resolution}", flush=True)
    return high


# ============================================================
# CLI
# ============================================================

def main():
    all_solvers = [
        "normal_eq",
        "svd",
        "qr",
        "l1_lp",
        "huber",
        "cauchy",
        "gd_mle",
        "ilp",
        "bp",
        "greedy_l2",
        "greedy_dh",
        "hillclimb",
    ]

    parser = argparse.ArgumentParser(
        description="Compare multiple solvers on standard ILWE under dg/uniform distributions."
    )
    parser.add_argument("--dist", choices=["dg", "uniform"], required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--eta", type=int, default=2)
    parser.add_argument("--sigma-a", type=float, default=None)
    parser.add_argument("--sigma-e", type=float, default=None)
    parser.add_argument("--alpha-a", type=int, default=None)
    parser.add_argument("--alpha-e", type=int, default=None)
    parser.add_argument("--num-keys", type=int, default=1)
    parser.add_argument("--success-keys", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-m", type=int, default=None)
    parser.add_argument("--max-m", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=1, help="sample-count precision; use 1 for exact integer precision")
    parser.add_argument("--solvers", nargs="+", default=["normal_eq", "bp", "greedy_dh"], choices=all_solvers)

    # shared/linear solver parameters
    parser.add_argument("--normal-lam", type=float, default=1e-10,
                        help="relative ridge factor for normal equation; ridge=lam*trace(A^T A)/n")
    parser.add_argument("--qr-no-pivot", action="store_true")

    # robust/MLE solver parameters
    parser.add_argument("--huber-delta", type=float, default=0.0, help="<=0 means use distribution-based scale")
    parser.add_argument("--cauchy-scale", type=float, default=0.0,
                        help="accepted for compatibility; ignored by regression-style Cauchy")
    parser.add_argument("--cauchy-iter", type=int, default=100)
    parser.add_argument("--gd-iter", type=int, default=5000)
    parser.add_argument("--gd-tol", type=float, default=1e-5)
    parser.add_argument("--gd-no-bounds", action="store_true")

    # ILP controls
    parser.add_argument("--ilp-time-limit", type=float, default=120.0)
    parser.add_argument("--ilp-max-m", type=int, default=1500, help="skip ILP above this m; 0 disables limit")

    # distribution hint controls
    parser.add_argument("--bp-iter", type=int, default=100)
    parser.add_argument("--greedy-dh-iter", type=int, default=100)
    parser.add_argument("--greedy-kappa-mode", choices=["decay", "cycle", "small", "one"], default="decay")
    parser.add_argument("--greedy-kappa", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--tail-sigma", type=float, default=3.0)
    parser.add_argument("--max-rhs-support", type=int, default=0, help="0 = full support; positive value caps RHS support for BP/Greedy-DH")
    parser.add_argument("--uniform-prior", action="store_true")
    parser.add_argument("--no-inner-bound", action="store_true")
    parser.add_argument("--hint-max-m", type=int, default=0, help="skip BP/Greedy-DH above this m; 0 disables limit")

    # custom greedy / hillclimb controls
    parser.add_argument("--greedy-l2-iter", type=int, default=100)
    parser.add_argument("--greedy-l2-k", type=int, default=None)
    parser.add_argument("--greedy-l2-init", choices=["zeros", "normal_eq"], default="zeros")
    parser.add_argument("--hillclimb-iter", type=int, default=2000)
    parser.add_argument("--hillclimb-block-size", type=int, default=2)
    parser.add_argument("--hillclimb-patience", type=int, default=300)
    parser.add_argument("--hillclimb-init", choices=["zeros", "normal_eq", "ols", "lsqr"], default="normal_eq")
    parser.add_argument("--hillclimb-error-bound", type=float, default=0.0,
                        help="B in b-B <= A x <= b+B; <=0 uses alpha_e for uniform or ceil(tail_sigma*sigma_e) for dg")
    parser.add_argument("--hillclimb-fitness", choices=["count", "excess", "combined"], default="excess")
    parser.add_argument("--hillclimb-lambda", type=float, default=0.0,
                        help="Penalty lambda for combined fitness; <=0 uses max(error_bound,1)")
    parser.add_argument("--hillclimb-no-adaptive-w", action="store_true")
    parser.add_argument("--hillclimb-adaptive-w-max", type=int, default=4)
    parser.add_argument("--hillclimb-adaptive-w-patience", type=int, default=50)
    parser.add_argument("--hillclimb-no-lateral", action="store_true")
    parser.add_argument("--hillclimb-no-diversify", action="store_true")
    parser.add_argument("--hillclimb-diversify-strength", type=float, default=1.0)
    parser.add_argument("--hillclimb-sweep-interval", type=int, default=0)
    parser.add_argument("--hillclimb-no-perturb", action="store_true")
    parser.add_argument("--hillclimb-perturb-strength", type=int, default=30)
    parser.add_argument("--hillclimb-perturb-patience", type=int, default=100)
    parser.add_argument("--hillclimb-perturb-max", type=int, default=100)
    parser.add_argument("--hillclimb-no-sequential-w", action="store_true")
    parser.add_argument("--hillclimb-score-guided", action="store_true")
    parser.add_argument("--hillclimb-score-temperature", type=float, default=2.0)
    parser.add_argument("--hillclimb-candidate-chunk", type=int, default=1024)

    args = parser.parse_args()
    if args.success_keys is None:
        args.success_keys = args.num_keys
    if args.start_m is None:
        args.start_m = args.n

    if args.dist == "dg" and (args.sigma_a is None or args.sigma_e is None):
        parser.error("--sigma-a and --sigma-e are required for --dist dg")
    if args.dist == "uniform" and (args.alpha_a is None or args.alpha_e is None):
        parser.error("--alpha-a and --alpha-e are required for --dist uniform")

    if "bp" in args.solvers and PyBP is None:
        raise RuntimeError(f"PyBP unavailable: {HINT_SOLVER_IMPORT_ERROR!r}")
    if "greedy_dh" in args.solvers and PyGreedy is None:
        raise RuntimeError(f"PyGreedy unavailable: {HINT_SOLVER_IMPORT_ERROR!r}")

    print("=== Standard ILWE all-solver comparison ===")
    print("generation    = on-demand by current m")
    print(f"dist          = {args.dist}")
    print(f"n             = {args.n}")
    print(f"eta           = {args.eta}")
    print(f"num_keys      = {args.num_keys}")
    print(f"success_keys  = {args.success_keys}")
    print(f"start_m       = {args.start_m}")
    print(f"max_m         = {args.max_m}")
    print(f"resolution    = {args.resolution}")
    print(f"solvers       = {args.solvers}")
    if args.dist == "dg":
        print(f"sigma_a       = {args.sigma_a}")
        print(f"sigma_e       = {args.sigma_e}")
        print(f"tail_sigma    = {args.tail_sigma}")
        print(f"max_rhs_support = {args.max_rhs_support}")
    else:
        print(f"alpha_a       = {args.alpha_a}")
        print(f"alpha_e       = {args.alpha_e}")
        print(f"max_rhs_support = {args.max_rhs_support}")
    print(f"sigma_eff     = {error_std_from_args(args):.6g}")
    print(f"huber_delta   = {resolve_huber_delta(args):.6g}")
    print("cauchy_scale  = ignored (regression-style Cauchy)")
    print(f"normal_lam    = {args.normal_lam}")
    print(f"hill_B        = {resolve_hillclimb_error_bound(args):.6g}")
    print()

    instance_seeds: list[int] = []
    for key_idx in range(args.num_keys):
        inst_seed = args.seed + 1000003 * key_idx
        instance_seeds.append(inst_seed)
        s_preview = sample_sparse_secret(args.n, np.random.default_rng(inst_seed))
        print(f"Prepared key {key_idx + 1}/{args.num_keys}: seed={inst_seed}, nonzero={int(np.count_nonzero(s_preview))}/{args.n}")

    results: dict[str, int | None] = {}
    for solver_name in args.solvers:
        m_min = find_min_m_for_solver(solver_name, instance_seeds, args)
        results[solver_name] = m_min

    print("\n=== Final minimal sample numbers ===")
    for solver_name, m_min in results.items():
        print(f"{solver_name:12s}: {m_min}")


if __name__ == "__main__":
    main()
