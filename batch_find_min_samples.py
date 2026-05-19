#!/usr/bin/env python3
"""
Batch-search minimal sample counts for CILWE solvers.

This script reuses build_and_solve_cilwe.py for both pieces:

    1. synthetic CILWE sample generation, z = C @ s + e;
    2. solver dispatch to the implementations imported from
       ILWE_all_solvers_hillclimb_ref_param.py.

For each parameter combination and solver, it searches for the smallest number
of samples per polynomial m that recovers all coefficients for enough trial
seeds. The search first doubles m until it finds a successful upper bound, then
binary-searches to the requested resolution.

Within the same level/filter/solver track, p values are processed from low to
high. Once a solver cannot recover at a lower p before max_m, higher p values
for that same track are skipped and recorded in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_and_solve_cilwe import (
    SOLVER_CHOICES,
    CilweParams,
    build_cilwe_samples,
    keygen,
    solve_secret_with_timing,
)


# ============================================================
# Batch configuration arrays
# ============================================================
#
# Edit these arrays for the default batch run. Command-line arguments
# --levels and --p-values can still override them for one-off experiments.

SECURITY_LEVEL_CANDIDATES = [3]
P_CANDIDATES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class TrialResult:
    success: bool
    recovered: int
    total: int
    elapsed: float
    error: str | None = None


@dataclass
class SearchResult:
    level: int
    p: float
    filter_threshold: float
    solver: str
    min_m: int | None
    status: str
    success_trials: int
    total_trials: int
    avg_recovered: float
    avg_time: float
    last_m: int
    errors: str


def clone_solver_args(args: argparse.Namespace, solver: str, seed: int, p: float) -> argparse.Namespace:
    data = vars(args).copy()
    data["method"] = solver
    data["seed"] = seed
    data["p"] = p
    return SimpleNamespace(**data)


def run_trial(
    *,
    level: int,
    p: float,
    m: int,
    filter_threshold: float,
    solver: str,
    trial_seed: int,
    args: argparse.Namespace,
) -> TrialResult:
    params = CilweParams.from_level(level)
    rng = np.random.default_rng(trial_seed)
    secrets = keygen(params, rng)
    samples, max_residual = build_cilwe_samples(
        secrets=secrets,
        params=params,
        rng=rng,
        sample_count=m,
        p=p,
        filter_threshold=filter_threshold,
    )
    if max_residual != 0:
        return TrialResult(False, 0, params.l * params.n, 0.0, f"max residual {max_residual}")

    solver_args = clone_solver_args(args, solver, trial_seed, p)
    total_recovered = 0
    total_elapsed = 0.0
    total_coefficients = params.l * params.n

    try:
        for poly_idx, (matrix, z_values, _errors) in enumerate(sample.stack(params.n) for sample in samples):
            if len(z_values) < params.n:
                return TrialResult(False, total_recovered, total_coefficients, total_elapsed, "not enough rows")
            estimate, elapsed = solve_secret_with_timing(
                matrix=matrix,
                z_values=z_values,
                params=params,
                args=solver_args,
                key_index=poly_idx,
            )
            total_elapsed += elapsed
            rounded = np.rint(estimate).astype(np.int64)
            total_recovered += int(np.count_nonzero(rounded == secrets[poly_idx]))
    except Exception as exc:  # noqa: BLE001
        return TrialResult(False, total_recovered, total_coefficients, total_elapsed, str(exc))

    return TrialResult(
        success=total_recovered == total_coefficients,
        recovered=total_recovered,
        total=total_coefficients,
        elapsed=total_elapsed,
    )


def success_at_m(
    *,
    level: int,
    p: float,
    m: int,
    filter_threshold: float,
    solver: str,
    trial_seeds: list[int],
    args: argparse.Namespace,
) -> tuple[bool, int, list[TrialResult]]:
    results: list[TrialResult] = []
    successes = 0

    for trial_idx, seed in enumerate(trial_seeds):
        result = run_trial(
            level=level,
            p=p,
            m=m,
            filter_threshold=filter_threshold,
            solver=solver,
            trial_seed=seed,
            args=args,
        )
        results.append(result)
        successes += int(result.success)

        if args.verbose:
            status = "Success" if result.success else "Failure"
            suffix = f", error={result.error}" if result.error else ""
            print(
                f"[{solver}] level={level}, p={p:g}, m={m}, "
                f"trial={trial_idx + 1}/{len(trial_seeds)}, {status}, "
                f"recovered={result.recovered}/{result.total}, time={result.elapsed:.3f}s{suffix}",
                flush=True,
            )

        failures_allowed = len(trial_seeds) - args.success_trials
        failures = (trial_idx + 1) - successes
        if failures > failures_allowed:
            break
        if successes >= args.success_trials:
            break

    return successes >= args.success_trials, successes, results


def summarize_trials(results: list[TrialResult]) -> tuple[float, float, str]:
    if not results:
        return 0.0, 0.0, ""
    avg_recovered = float(np.mean([r.recovered for r in results]))
    avg_time = float(np.mean([r.elapsed for r in results]))
    errors = "; ".join(sorted({r.error for r in results if r.error}))
    return avg_recovered, avg_time, errors


def find_min_m(
    *,
    level: int,
    p: float,
    filter_threshold: float,
    solver: str,
    trial_seeds: list[int],
    args: argparse.Namespace,
) -> SearchResult:
    params = CilweParams.from_level(level)
    low = 0
    high = int(args.start_m if args.start_m is not None else params.n)
    high = max(1, high)
    max_m = int(args.max_m)
    last_results: list[TrialResult] = []
    last_successes = 0

    print(f"\n=== level={level}, p={p:g}, filter={filter_threshold:g}, solver={solver} ===", flush=True)

    found = False
    while high <= max_m:
        ok, successes, results = success_at_m(
            level=level,
            p=p,
            m=high,
            filter_threshold=filter_threshold,
            solver=solver,
            trial_seeds=trial_seeds,
            args=args,
        )
        last_results = results
        last_successes = successes

        print(f"[{solver}] m={high}: successes={successes}/{len(trial_seeds)}", flush=True)
        if ok:
            found = True
            break

        low = high
        if high == max_m:
            break
        high = min(max_m, max(high + 1, high * 2))

    if not found:
        avg_recovered, avg_time, errors = summarize_trials(last_results)
        return SearchResult(
            level=level,
            p=p,
            filter_threshold=filter_threshold,
            solver=solver,
            min_m=None,
            status="not_found",
            success_trials=last_successes,
            total_trials=len(trial_seeds),
            avg_recovered=avg_recovered,
            avg_time=avg_time,
            last_m=high,
            errors=errors,
        )

    resolution = max(1, int(args.resolution))
    best_results = last_results
    best_successes = last_successes

    while high - low > resolution:
        mid = ((low + high) // 2 // resolution) * resolution
        if mid <= low:
            mid = low + resolution
        if mid >= high:
            mid = high - resolution
        if mid <= low or mid >= high:
            break

        ok, successes, results = success_at_m(
            level=level,
            p=p,
            m=mid,
            filter_threshold=filter_threshold,
            solver=solver,
            trial_seeds=trial_seeds,
            args=args,
        )
        print(f"[{solver}] m={mid}: successes={successes}/{len(trial_seeds)}", flush=True)

        if ok:
            high = mid
            best_results = results
            best_successes = successes
        else:
            low = mid

    avg_recovered, avg_time, errors = summarize_trials(best_results)
    return SearchResult(
        level=level,
        p=p,
        filter_threshold=filter_threshold,
        solver=solver,
        min_m=high,
        status="success",
        success_trials=best_successes,
        total_trials=len(trial_seeds),
        avg_recovered=avg_recovered,
        avg_time=avg_time,
        last_m=high,
        errors=errors,
    )


def skipped_after_lower_p_failure(
    *,
    level: int,
    p: float,
    filter_threshold: float,
    solver: str,
    lower_p: float,
    total_trials: int,
    max_m: int,
) -> SearchResult:
    return SearchResult(
        level=level,
        p=p,
        filter_threshold=filter_threshold,
        solver=solver,
        min_m=None,
        status="skipped_after_lower_p_not_found",
        success_trials=0,
        total_trials=total_trials,
        avg_recovered=0.0,
        avg_time=0.0,
        last_m=max_m,
        errors=f"lower p={lower_p:g} was not_found for the same level/filter/solver",
    )


def write_result(path: Path, result: SearchResult, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not append) or (not path.exists())
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level",
                "p",
                "filter_threshold",
                "solver",
                "min_m",
                "status",
                "success_trials",
                "total_trials",
                "avg_recovered",
                "avg_time",
                "last_m",
                "errors",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(result.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-search minimal sample counts for CILWE solvers.")

    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=[2, 3, 5],
        default=None,
        help=f"Defaults to SECURITY_LEVEL_CANDIDATES={SECURITY_LEVEL_CANDIDATES}",
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=None,
        help=f"Defaults to P_CANDIDATES={P_CANDIDATES}",
    )
    parser.add_argument("--filter-thresholds", nargs="+", type=float, default=None,
                        help="If omitted, each level uses threshold=tau.")
    parser.add_argument("--solvers", nargs="+", choices=SOLVER_CHOICES, default=SOLVER_CHOICES)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--success-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-m", type=int, default=None)
    parser.add_argument("--max-m", type=int, default=100000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("CLWE_Solve/min_sample_results.csv"))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Solver parameters mirrored from build_and_solve_cilwe.py.
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

    args = parser.parse_args()
    if args.levels is None:
        args.levels = list(SECURITY_LEVEL_CANDIDATES)
    if args.p_values is None:
        args.p_values = list(P_CANDIDATES)

    if args.num_trials <= 0:
        parser.error("--num-trials must be positive")
    if args.success_trials is None:
        args.success_trials = args.num_trials
    if not 1 <= args.success_trials <= args.num_trials:
        parser.error("--success-trials must be in [1, num-trials]")
    if args.max_m <= 0:
        parser.error("--max-m must be positive")
    if args.start_m is not None and args.start_m <= 0:
        parser.error("--start-m must be positive")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    for p in args.p_values:
        if not 0.0 <= p <= 1.0:
            parser.error("--p-values must be in [0, 1]")
    args.p_values = sorted(args.p_values)

    return args


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.append:
        args.output.unlink()

    print("=== CILWE batch minimal-sample search ===")
    print(f"levels         = {args.levels}")
    print(f"p_values       = {args.p_values}")
    print(f"solvers        = {args.solvers}")
    print(f"num_trials     = {args.num_trials}")
    print(f"success_trials = {args.success_trials}")
    print(f"start_m        = {args.start_m}")
    print(f"max_m          = {args.max_m}")
    print(f"resolution     = {args.resolution}")
    print(f"output         = {args.output}")

    trial_seeds = [args.seed + 1000003 * idx for idx in range(args.num_trials)]
    results: list[SearchResult] = []

    for level in args.levels:
        params = CilweParams.from_level(level)
        thresholds = args.filter_thresholds if args.filter_thresholds is not None else [float(params.tau)]
        for threshold in thresholds:
            if threshold <= 0:
                raise ValueError("filter thresholds must be positive")
            failed_lower_p_by_solver: dict[str, float] = {}
            for p in args.p_values:
                for solver in args.solvers:
                    lower_p = failed_lower_p_by_solver.get(solver)
                    if lower_p is not None:
                        result = skipped_after_lower_p_failure(
                            level=level,
                            p=p,
                            filter_threshold=float(threshold),
                            solver=solver,
                            lower_p=lower_p,
                            total_trials=len(trial_seeds),
                            max_m=int(args.max_m),
                        )
                        print(
                            f"[skip] level={level}, p={p:g}, filter={threshold:g}, solver={solver}: "
                            f"lower p={lower_p:g} was not_found",
                            flush=True,
                        )
                    else:
                        result = find_min_m(
                            level=level,
                            p=p,
                            filter_threshold=float(threshold),
                            solver=solver,
                            trial_seeds=trial_seeds,
                            args=args,
                        )
                        if result.status == "not_found":
                            failed_lower_p_by_solver[solver] = p

                    results.append(result)
                    write_result(args.output, result, append=True)
                    print(
                        f"[result] level={level}, p={p:g}, filter={threshold:g}, "
                        f"solver={solver}, min_m={result.min_m}, status={result.status}",
                        flush=True,
                    )

    print("\n=== Summary ===")
    for result in results:
        print(
            f"level={result.level}, p={result.p:g}, filter={result.filter_threshold:g}, "
            f"{result.solver:10s}: {result.min_m} ({result.status})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
