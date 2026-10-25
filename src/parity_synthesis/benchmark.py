"""Seeded experiments and resource summaries, without competition grading."""

import json
from pathlib import Path
import time

import numpy as np

from .api import compile_unitary
from .targets import haar_unitary


def load_cases(path):
    """Read a small JSON experiment configuration; execute no source code."""
    config = json.loads(Path(path).read_text())
    for key in ("seed_groups", "sizes"):
        if not config.get(key) or any(type(value) is not int for value in config[key]):
            raise ValueError(f"{key} must be a nonempty list of integers")
    if any(value < 1 for value in config["sizes"]):
        raise ValueError("benchmark sizes must be positive")
    samples = config.get("samples_per_size", 2)
    if type(samples) is not int or samples < 1:
        raise ValueError("samples_per_size must be positive")
    return config


def run_benchmark(config, *, epsilon=1e-9, backend="cyclosynth",
                  method="phase11", fold_common=True, error_budget=None,
                  cache_dir=None, cache_only=False, verbose=False,
                  tolerance=1e-8, rotation_weight=44.0):
    """Synthesize the configured targets and report measured errors and costs.

    Each seed group is an experiment, not a grade. Reference costs are only
    compared when their rotation weight matches the requested reporting metric.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(rotation_weight) or rotation_weight < 0:
        raise ValueError("rotation_weight must be finite and nonnegative")
    started = time.monotonic()
    reports = []
    for group_seed in config["seed_groups"]:
        samples = []
        means = {}
        group_start = time.monotonic()
        for num_qubits in config["sizes"]:
            costs = []
            for sample in range(config.get("samples_per_size", 2)):
                seed = group_seed + 1000 * num_qubits + sample
                if verbose:
                    print(f"n={num_qubits}, group={group_seed}, sample={sample}", flush=True)
                target = haar_unitary(num_qubits, seed)
                circuit = compile_unitary(
                    target, epsilon=epsilon, backend=backend, method=method,
                    fold_common=fold_common, error_budget=error_budget,
                    cache_dir=cache_dir, cache_only=cache_only, verbose=verbose,
                )
                row = circuit.statistics(rotation_weight)
                row.update(target_seed=seed, sample_index=sample,
                           max_entrywise_error=circuit.error_against(target))
                samples.append(row)
                costs.append(row["weighted_cost"])
            means[str(num_qubits)] = float(np.mean(costs))
        reference = config.get("reference_costs", {})
        same_weight = rotation_weight == config.get("reference_rotation_weight", 44)
        savings = {
            n: 100 * (1 - cost / reference[n]) for n, cost in means.items()
            if same_weight and n in reference and reference[n] > 0
        }
        largest_error = max(row["max_entrywise_error"] for row in samples)
        reports.append({
            "group_seed": group_seed,
            "samples": samples,
            "mean_weighted_cost_by_size": means,
            "saving_against_reference_percent_by_size": savings,
            "max_entrywise_error": largest_error,
            "within_requested_tolerance": bool(largest_error <= tolerance),
            "elapsed_seconds": time.monotonic() - group_start,
        })
    return {
        "settings": {
            "epsilon": epsilon, "backend": backend, "method": method,
            "fold_common": fold_common, "error_budget": error_budget,
            "entrywise_tolerance": tolerance, "rotation_weight": rotation_weight,
        },
        "cases": reports,
        "elapsed_seconds": time.monotonic() - started,
        "note": "Checkpoint reuse is recorded per sample; this is not necessarily a cold-cache run.",
    }
