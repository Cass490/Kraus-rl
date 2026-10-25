"""Command-line compilation and seeded experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from .api import compile_unitary
from .benchmark import load_cases, run_benchmark
from .precision import synthesis_word_counts, word_epsilon
from .targets import haar_unitary


def _common(parser):
    parser.add_argument("--backend", choices=("cyclosynth", "pygridsynth"), default="cyclosynth")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="per-word tolerance cap")
    parser.add_argument("--method", choices=("baseline", "phase11"), default="phase11")
    parser.add_argument("--no-fold-common", action="store_true")
    parser.add_argument("--error-budget", type=float)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/parity-synthesis"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parity-based quantum unitary synthesis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="compile a matrix or a seeded example")
    source = compile_parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="unitary matrix saved as a .npy file")
    source.add_argument("--qubits", type=int, help="random target size; default 2")
    compile_parser.add_argument("--seed", type=int, default=7)
    compile_parser.add_argument("--check", action="store_true", help="simulate the compiled matrix")
    compile_parser.add_argument("--output", type=Path, help="write the compiled circuit as JSON")
    compile_parser.add_argument("--dry-run", action="store_true", help="print the precision plan only")
    _common(compile_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help="run the cases in a JSON configuration")
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, help="write the complete measured report")
    benchmark_parser.add_argument("--tolerance", type=float, default=1e-8)
    benchmark_parser.add_argument("--rotation-weight", type=float, default=44)
    _common(benchmark_parser)
    args = parser.parse_args(argv)
    if args.no_cache and args.cache_only:
        parser.error("--no-cache and --cache-only cannot be combined")
    options = {
        "epsilon": args.epsilon, "backend": args.backend,
        "method": args.method, "fold_common": not args.no_fold_common,
        "error_budget": args.error_budget,
        "cache_dir": None if args.no_cache else args.cache_dir,
        "cache_only": args.cache_only, "verbose": args.verbose,
    }
    try:
        if args.command == "benchmark":
            config = load_cases(args.config)
            report = run_benchmark(config, **options, tolerance=args.tolerance,
                                   rotation_weight=args.rotation_weight)
            if args.output:
                _write_json(args.output, report)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if all(case["within_requested_tolerance"] for case in report["cases"]) else 1

        target = None
        if args.input:
            target = np.load(args.input, allow_pickle=False)
            if target.ndim != 2 or target.shape[0] != target.shape[1]:
                raise ValueError("input must be a square matrix")
            dimension = len(target)
            if dimension < 2 or dimension & (dimension - 1):
                raise ValueError("input dimension must be a power of two")
            num_qubits = dimension.bit_length() - 1
        else:
            num_qubits = 2 if args.qubits is None else args.qubits
            if num_qubits < 1:
                raise ValueError("--qubits must be positive")
        if args.dry_run:
            planned, worst = synthesis_word_counts(num_qubits, args.method)
            cap = word_epsilon(num_qubits, args.epsilon, args.error_budget)
            print(json.dumps({
                "num_qubits": num_qubits, "planned_words": planned,
                "fallback_maximum_words": worst, "per_word_epsilon": cap,
                "sum_of_word_tolerance_caps": planned * cap,
                "note": "No synthesis was performed; final accuracy requires matrix validation.",
            }, indent=2))
            return 0
        if target is None:
            target = haar_unitary(num_qubits, args.seed)
        circuit = compile_unitary(target, **options)
        report = circuit.statistics()
        if args.check:
            report["max_entrywise_error"] = circuit.error_against(target)
        if args.output:
            _write_json(args.output, circuit.to_dict())
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
