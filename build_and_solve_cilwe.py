#!/usr/bin/env python3
"""
Build and solve synthetic CILWE instances.

This script uses the same sample-generation idea as
concealed_ilwe/regression/sampler.py:

    z = C @ s + e

where s is a small-integer secret, each row of C has tau non-zero entries in
{-1, 1}, and each sample independently receives a concealed/non-zero error with
probability p. Rows with |z| above a filter threshold are rejected and replaced
with clean rows, matching the original regression sampler's convention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np

try:
    from ILWE_all_solvers_hillclimb_ref_param import (
        normalize_dist as ref_normalize_dist,
        solve_cauchy as ref_solve_cauchy,
        solve_bp as ref_solve_bp,
        solve_gd_mle as ref_solve_gd_mle,
        solve_greedy_dh as ref_solve_greedy_dh,
        solve_greedy_l2 as ref_solve_greedy_l2,
        solve_hillclimb as ref_solve_hillclimb,
        solve_huber as ref_solve_huber,
        solve_ilp as ref_solve_ilp,
        solve_l1_lp as ref_solve_l1_lp,
        solve_normal_eq as ref_solve_normal_eq,
        solve_qr as ref_solve_qr,
        solve_svd as ref_solve_svd,
    )
except ModuleNotFoundError:
    from CLWE_Solve.ILWE_all_solvers_hillclimb_ref_param import (
        normalize_dist as ref_normalize_dist,
        solve_cauchy as ref_solve_cauchy,
        solve_bp as ref_solve_bp,
        solve_gd_mle as ref_solve_gd_mle,
        solve_greedy_dh as ref_solve_greedy_dh,
        solve_greedy_l2 as ref_solve_greedy_l2,
        solve_hillclimb as ref_solve_hillclimb,
        solve_huber as ref_solve_huber,
        solve_ilp as ref_solve_ilp,
        solve_l1_lp as ref_solve_l1_lp,
        solve_normal_eq as ref_solve_normal_eq,
        solve_qr as ref_solve_qr,
        solve_svd as ref_solve_svd,
    )


SOLVER_CHOICES = [
    "normal_eq",
    "svd",
    "qr",
    # "l1_lp",
     "huber",
     # "cauchy",
     "gd_mle",
    "ilp",
    # "bp",
     "greedy_l2",
    # "greedy_dh",
    "hillclimb",
]


@dataclass(frozen=True)
class CilweParams:
    level: int
    n: int
    eta: int
    tau: int
    l: int

    @staticmethod
    def from_level(level: int) -> "CilweParams":
        if level == 2:
            return CilweParams(level=2, n=256, eta=2, tau=39, l=4)
        if level == 3:
            return CilweParams(level=3, n=256, eta=4, tau=49, l=5)
        if level == 5:
            return CilweParams(level=5, n=256, eta=2, tau=60, l=7)
        raise ValueError(f"Unsupported Dilithium level: {level}")


@dataclass
class CilweSamples:
    rows: list[np.ndarray]
    z_values: list[np.ndarray]
    error_values: list[np.ndarray]

    def append(self, rows: np.ndarray, z_values: np.ndarray, error_values: np.ndarray) -> None:
        if len(z_values) == 0:
            return
        self.rows.append(rows.astype(np.float64, copy=False))
        self.z_values.append(z_values.astype(np.float64, copy=False))
        self.error_values.append(error_values.astype(np.float64, copy=False))

    def stack(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.rows:
            return np.empty((0, n), dtype=np.float64), np.empty(0), np.empty(0)
        return np.vstack(self.rows), np.concatenate(self.z_values), np.concatenate(self.error_values)


def keygen(params: CilweParams, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(-params.eta, params.eta + 1, size=(params.l, params.n), dtype=np.int64)


def generate_sparse_matrix(
    sample_count: int,
    dimension: int,
    tau: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if tau > dimension:
        raise ValueError(f"tau={tau} cannot exceed dimension={dimension}")

    matrix = np.zeros((sample_count, dimension), dtype=np.float64)
    for row_idx in range(sample_count):
        non_zero_indices = rng.choice(dimension, tau, replace=False)
        matrix[row_idx, non_zero_indices] = rng.choice(np.array([-1.0, 1.0]), size=tau)
    return matrix


def generate_errors(
    sample_count: int,
    tau: int,
    p: float,
    secret: np.ndarray,
    matrix: np.ndarray,
    filter_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    errors = np.zeros(sample_count, dtype=np.float64)

    for row_idx in range(sample_count):
        if rng.random() >= p:
            continue

        while True:
            error = rng.uniform(-4 * tau, 4 * tau)
            z_value = matrix[row_idx] @ secret + error
            if abs(z_value) <= filter_threshold:
                errors[row_idx] = int(error)
                break

    return errors


def generate_cilwe_sample(
    sample_count: int,
    tau: int,
    p: float,
    *,
    secret: np.ndarray,
    dimension: int,
    filter_threshold: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = generate_sparse_matrix(sample_count, dimension, tau, rng)
    errors = generate_errors(sample_count, tau, p, secret, matrix, filter_threshold, rng)
    z_values = matrix @ secret + errors

    while True:
        keep = np.abs(z_values) <= filter_threshold
        if np.all(keep):
            break

        matrix = matrix[keep]
        z_values = z_values[keep]
        errors = errors[keep]

        missing = sample_count - len(z_values)
        replacement = generate_sparse_matrix(missing, dimension, tau, rng)
        replacement_z = replacement @ secret

        matrix = np.vstack([matrix, replacement])
        z_values = np.concatenate([z_values, replacement_z])
        errors = np.concatenate([errors, np.zeros(missing, dtype=np.float64)])

    return matrix, z_values, errors


def build_cilwe_samples(
    secrets: np.ndarray,
    params: CilweParams,
    rng: np.random.Generator,
    sample_count: int,
    p: float,
    filter_threshold: float,
) -> tuple[list[CilweSamples], int]:
    samples = [CilweSamples([], [], []) for _ in range(params.l)]
    max_equation_residual = 0

    for poly_idx in range(params.l):
        matrix, z_values, errors = generate_cilwe_sample(
            sample_count=sample_count,
            tau=params.tau,
            p=p,
            secret=secrets[poly_idx],
            dimension=params.n,
            filter_threshold=filter_threshold,
            rng=rng,
        )
        residual = z_values - matrix @ secrets[poly_idx] - errors
        max_equation_residual = max(max_equation_residual, int(np.max(np.abs(residual))))
        samples[poly_idx].append(matrix, z_values, errors)

    return samples, max_equation_residual


def _centered_integer_support(bound: int, max_support: int) -> range:
    if max_support <= 0 or max_support >= 2 * bound + 1:
        return range(-bound, bound + 1)
    half = max_support // 2
    lo = max(-bound, -half)
    hi = min(bound, lo + max_support - 1)
    return range(lo, hi + 1)


def build_cilwe_hints(
    matrix: np.ndarray,
    z_values: np.ndarray,
    params: CilweParams,
    *,
    p: float,
    max_rhs_support: int,
    use_inner_bound: bool,
) -> list[tuple[list[int], list[tuple[int, float]]]]:
    """
    Convert CILWE rows into the distribution-hint format consumed by BP/Greedy-DH.

    The imported solvers expect hints of the form (coefficients, RHS distribution)
    for <C_i, s>. For this CILWE sampler, e is 0 with probability 1-p and otherwise
    lies in the integer window [-4*tau, 4*tau], so rhs = z_i - e.
    """
    matrix = np.asarray(matrix)
    z_values = np.asarray(z_values).reshape(-1)
    error_bound = int(ceil(4 * params.tau))
    error_support = _centered_integer_support(error_bound, int(max_rhs_support))

    hints: list[tuple[list[int], list[tuple[int, float]]]] = []
    for row, z_i in zip(matrix, z_values):
        coeffs = [int(round(float(value))) for value in row]
        inner_bound = int(params.eta * np.sum(np.abs(row))) if use_inner_bound else None

        rhs_candidates: list[tuple[int, float]] = []
        if p < 1.0:
            rhs = int(round(float(z_i)))
            if inner_bound is None or -inner_bound <= rhs <= inner_bound:
                rhs_candidates.append((rhs, 1.0 - float(p)))

        if p > 0.0:
            noisy_weight = float(p) / max(1, len(error_support))
            for error in error_support:
                rhs = int(round(float(z_i) - float(error)))
                if inner_bound is not None and not (-inner_bound <= rhs <= inner_bound):
                    continue
                rhs_candidates.append((rhs, noisy_weight))

        if not rhs_candidates:
            rhs = int(round(float(z_i)))
            if inner_bound is not None:
                rhs = min(max(rhs, -inner_bound), inner_bound)
            rhs_candidates = [(rhs, 1.0)]

        hints.append((coeffs, ref_normalize_dist(rhs_candidates)))

    return hints


def solve_secret(
    matrix: np.ndarray,
    z_values: np.ndarray,
    params: CilweParams,
    args: argparse.Namespace,
    *,
    key_index: int,
) -> np.ndarray:
    method = args.method

    if method == "normal_eq":
        return ref_solve_normal_eq(matrix, z_values, params.eta, lam=args.normal_lam)
    if method == "svd":
        return ref_solve_svd(matrix, z_values, params.eta)
    if method == "qr":
        return ref_solve_qr(matrix, z_values, params.eta, pivoting=not args.qr_no_pivot)
    if method == "l1_lp":
        return ref_solve_l1_lp(matrix, z_values, params.eta)
    if method == "huber":
        return ref_solve_huber(matrix, z_values, params.eta, delta=args.huber_delta)
    if method == "cauchy":
        return ref_solve_cauchy(
            matrix,
            z_values,
            params.eta,
            scale=args.cauchy_scale,
            max_iter=args.cauchy_iter,
        )
    if method == "gd_mle":
        return ref_solve_gd_mle(
            matrix,
            z_values,
            params.eta,
            max_iter=args.gd_iter,
            tol=args.gd_tol,
            use_bounds=not args.gd_no_bounds,
        )
    if method == "ilp":
        if args.ilp_max_m > 0 and matrix.shape[0] > args.ilp_max_m:
            raise RuntimeError(f"skip ILP for m={matrix.shape[0]} > ilp_max_m={args.ilp_max_m}")
        return ref_solve_ilp(matrix, z_values, params.eta, time_limit=args.ilp_time_limit)
    if method in {"bp", "greedy_dh"}:
        hint_max_m = int(getattr(args, "hint_max_m", 0))
        if hint_max_m > 0 and matrix.shape[0] > hint_max_m:
            raise RuntimeError(f"skip {method} for m={matrix.shape[0]} > hint_max_m={hint_max_m}")
        hints = build_cilwe_hints(
            matrix,
            z_values,
            params,
            p=float(getattr(args, "p", 1.0)),
            max_rhs_support=int(getattr(args, "max_rhs_support", 0)),
            use_inner_bound=not getattr(args, "no_inner_bound", False),
        )
        if method == "bp":
            return ref_solve_bp(
                hints,
                params.eta,
                max_iter=int(getattr(args, "bp_iter", 100)),
                threads=getattr(args, "threads", None),
                use_sparse_prior=not getattr(args, "uniform_prior", False),
            )
        return ref_solve_greedy_dh(
            hints,
            params.eta,
            max_iter=int(getattr(args, "greedy_dh_iter", 100)),
            threads=getattr(args, "threads", None),
            kappa_mode=getattr(args, "greedy_kappa_mode", "decay"),
            fixed_kappa=getattr(args, "greedy_kappa", None),
        )
    if method == "greedy_l2":
        return ref_solve_greedy_l2(
            matrix,
            z_values,
            params.eta,
            max_iter=args.greedy_l2_iter,
            k=args.greedy_l2_k,
            init=args.greedy_l2_init,
        )
    if method == "hillclimb":
        error_bound = args.hillclimb_error_bound
        if error_bound <= 0:
            error_bound = 4.0 * params.tau
        return ref_solve_hillclimb(
            matrix,
            z_values,
            params.eta,
            error_bound=error_bound,
            max_iter=args.hillclimb_iter,
            block_size=args.hillclimb_block_size,
            seed=args.seed + 5557 * key_index + matrix.shape[0],
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

    raise ValueError(f"unknown solver: {method}")


def solve_secret_with_timing(
    matrix: np.ndarray,
    z_values: np.ndarray,
    params: CilweParams,
    args: argparse.Namespace,
    *,
    key_index: int,
) -> tuple[np.ndarray, float]:
    from time import perf_counter

    start = perf_counter()
    estimate = solve_secret(matrix, z_values, params, args, key_index=key_index)
    elapsed = perf_counter() - start
    return estimate, elapsed


def save_npz(path: Path, secrets: np.ndarray, stacked_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"s1": secrets}
    for poly_idx, (matrix, z_values, errors) in enumerate(stacked_samples):
        payload[f"C_poly{poly_idx}"] = matrix
        payload[f"z_poly{poly_idx}"] = z_values
        payload[f"e_poly{poly_idx}"] = errors
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and solve synthetic CILWE samples.")
    parser.add_argument("--level", type=int, choices=[2, 3, 5], default=2)
    parser.add_argument("--samples", "--m", dest="samples", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.1, help="concealment/contamination rate")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--filter-threshold", type=float, default=None)
    parser.add_argument("--method", choices=SOLVER_CHOICES, default="cauchy")

    # Parameters mirrored from ILWE_all_solvers_hillclimb_ref_param.py.
    parser.add_argument("--normal-lam", type=float, default=1e-10)
    parser.add_argument("--qr-no-pivot", action="store_true")
    parser.add_argument("--huber-delta", type=float, default=0.125)
    parser.add_argument("--cauchy-scale", type=float, default=1.0)
    parser.add_argument("--cauchy-iter", type=int, default=100)
    parser.add_argument("--gd-iter", type=int, default=5000)
    parser.add_argument("--gd-tol", type=float, default=1e-5)
    parser.add_argument("--gd-no-bounds", action="store_true")
    parser.add_argument("--ilp-time-limit", type=float, default=120.0)
    parser.add_argument("--ilp-max-m", type=int, default=1500)
    parser.add_argument("--bp-iter", type=int, default=100)
    parser.add_argument("--greedy-dh-iter", type=int, default=100)
    parser.add_argument("--greedy-kappa-mode", choices=["decay", "cycle", "small", "one"], default="decay")
    parser.add_argument("--greedy-kappa", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-rhs-support", type=int, default=0)
    parser.add_argument("--uniform-prior", action="store_true")
    parser.add_argument("--no-inner-bound", action="store_true")
    parser.add_argument("--hint-max-m", type=int, default=0)
    parser.add_argument("--greedy-l2-iter", type=int, default=100)
    parser.add_argument("--greedy-l2-k", type=int, default=None)
    parser.add_argument("--greedy-l2-init", choices=["zeros", "normal_eq"], default="zeros")
    parser.add_argument("--hillclimb-iter", type=int, default=2000)
    parser.add_argument("--hillclimb-block-size", type=int, default=2)
    parser.add_argument("--hillclimb-patience", type=int, default=300)
    parser.add_argument("--hillclimb-init", choices=["zeros", "normal_eq", "ols", "lsqr"], default="normal_eq")
    parser.add_argument("--hillclimb-error-bound", type=float, default=0.0)
    parser.add_argument("--hillclimb-fitness", choices=["count", "excess", "combined"], default="excess")
    parser.add_argument("--hillclimb-lambda", type=float, default=0.0)
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
    parser.add_argument("--save-npz", type=Path)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples/--m must be positive")
    if not 0.0 <= args.p <= 1.0:
        raise ValueError("--p must be in [0, 1]")

    rng = np.random.default_rng(args.seed)
    params = CilweParams.from_level(args.level)
    filter_threshold = args.filter_threshold if args.filter_threshold is not None else float(params.tau)
    if filter_threshold <= 0:
        raise ValueError("--filter-threshold must be positive")

    secrets = keygen(params, rng)
    samples, max_residual = build_cilwe_samples(
        secrets=secrets,
        params=params,
        rng=rng,
        sample_count=args.samples,
        p=args.p,
        filter_threshold=filter_threshold,
    )

    print("Build phase")
    print(f"  level: {params.level}")
    print(f"  params: n={params.n}, l={params.l}, eta={params.eta}, tau={params.tau}")
    print(f"  samples per poly: {args.samples}")
    print(f"  concealment/contamination p: {args.p}")
    print(f"  filter threshold: {filter_threshold:g}")
    print(f"  max |z - C@s - e|: {max_residual}")

    stacked_samples = [sample.stack(params.n) for sample in samples]
    if args.save_npz is not None:
        save_npz(args.save_npz, secrets, stacked_samples)
        print(f"  saved data: {args.save_npz}")

    print()
    print("Solve phase")
    total_errors = 0
    total_coefficients = params.l * params.n

    for poly_idx, (matrix, z_values, errors) in enumerate(stacked_samples):
        row_count = len(z_values)
        contaminated = int(np.count_nonzero(errors != 0))
        clean_rows = row_count - contaminated
        print(f"  poly {poly_idx}: rows={row_count}, clean_rows={clean_rows}, contaminated={contaminated}")

        if row_count < params.n:
            print(f"    skipped: need at least {params.n} rows")
            total_errors += params.n
            continue

        try:
            estimate, elapsed = solve_secret_with_timing(
                matrix=matrix,
                z_values=z_values,
                params=params,
                args=args,
                key_index=poly_idx,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    method={args.method}, failed: {exc}")
            total_errors += params.n
            continue

        rounded = np.rint(estimate).astype(np.int64)
        errors_count = int(np.count_nonzero(rounded != secrets[poly_idx]))
        total_errors += errors_count
        print(f"    method={args.method}, time={elapsed:.3f}s, recovered={params.n - errors_count}/{params.n}")

    recovered = total_coefficients - total_errors
    print()
    print("Summary")
    print(f"  recovered coefficients: {recovered}/{total_coefficients}")
    print(f"  success: {total_errors == 0}")

    return 0 if max_residual == 0 and total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
