# Parity-Based Quantum Unitary Synthesis

A research implementation of quantum unitary compilation using parity-controlled
paired words, diagonal-phase carry, and reduced multiplexor factorizations.
Given a unitary matrix, the package produces a gate sequence, retains its
global phase, and provides direct matrix verification and resource counts.

The project focuses on the synthesis method. It contains no competition
submission function or grading harness.

## Research focus

- **Reduced parity factorization:** an ordered 11-mask model for four-control
  multiplexors, with output Z phases carried into subsequent circuit blocks.
- **Common-word folding:** joint compilation of two full branch words in place
  of a split pair followed by a separate common gate.
- **Native even-T synthesis:** a patch to cyclosynth that requests parity-compatible
  words directly rather than replacing odd-T results with longer fallback words.
- **Phase-aware compilation:** explicit control and scalar phase corrections,
  exact gate cleanup, and matrix-based validation of the resulting circuit.

These implementation choices build on established cosine-sine and paired-word
constructions. Global coverage of the reduced model and novelty relative to all
prior literature are not established. See [the method](docs/method.md).

## Recorded benchmark

The predecessor implementation recorded the following results for seed group 7,
with two targets at each size:

| Qubits | Mean weighted cost | Reference cost | Displayed reduction |
| ---: | ---: | ---: | ---: |
| 2 | 315 | 660 | 52.2% |
| 3 | 1,425 | 3,168 | 55.0% |
| 4 | 5,869.5 | 13,728 | 57.2% |
| 5 | 18,048 | 57,024 | **68.3%** |

The reported maximum entrywise error across this seed group was **5.4e-9**,
including global phase, with no auxiliary wires. The final five-qubit sample
contained **66,326 operations**, including GlobalPhase. The seed group took
**1,543 seconds**; this is not the total for both groups, and no cold-cache
timing protocol was recorded.

The metric is `N_T + 44*N_rot`. Five-qubit outputs retain **15 arbitrary RZ
rotations**, so the result is not a pure Clifford+T T-count comparison. The
reference cost is a fixed historical benchmark, not the best available compiler
at matched accuracy. The modular distribution has component validation; the
full native benchmark has **not** been rerun for this refactor.

[Measurement data](benchmarks/results/reference_seed7.json) ·
[Experiment protocol and limitations](docs/benchmarking.md)

## Installation

Tested with Python **3.10.19**. After extracting the archive or cloning the
repository:

```bash
cd parity-unitary-synthesis
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[native]"
bash scripts/setup_cyclosynth.sh
```

Native compilation also requires Git, Rust/rustup, a C build toolchain, `make`,
`m4`, and GMP/MPFR development packages. The setup script uses a pinned upstream
revision and the included even-T patch. It does not run synthesis or benchmarks.
If Rust is missing from the shell path, run `source "$HOME/.cargo/env"` first.

PennyLane is optional: install it with `python -m pip install -e ".[pennylane]"`
when using the recording integration. The core compiler and matrix simulator
do not require it. `requirements.txt` retains the full original measured
Python environment for reference.

## Python API

```python
from parity_synthesis import compile_unitary
from parity_synthesis.targets import haar_unitary

target = haar_unitary(num_qubits=2, seed=7)
circuit = compile_unitary(target, epsilon=1e-9, cache_dir=".cache/parity-synthesis")

print(circuit.statistics())
print(circuit.error_against(target))
```

`compile_unitary` accepts an arbitrary finite unitary matrix with power-of-two
dimension. The numerical implementation has primarily been studied at two
through five qubits. Its default settings enable the reduced model and folding.
The native synthesis backend is required for the reported word lengths; the
`pygridsynth` stand-in is useful for compatibility experiments but is costlier.

`epsilon` is a **per-word** cap. It does not guarantee the full circuit error;
use `error_against` to measure the compiled matrix. The optional `error_budget`
policy allocates more conservative local tolerances and may increase cost.

## Command-line use

Inspect the five-qubit plan without starting a synthesis search:

```bash
parity-synthesis compile --qubits 5 --dry-run
```

Compile and verify a small random target, or compile your own `.npy` matrix:

```bash
parity-synthesis compile --qubits 2 --seed 7 --check
parity-synthesis compile --input unitary.npy --check --output circuit.json
```

Run the two benchmark seed groups:

```bash
parity-synthesis benchmark --config benchmarks/cases.json --output .cache/benchmark.json
```

`python -m parity_synthesis` provides the same CLI. Compilation and the full
benchmark are explicit, potentially expensive operations. Use `--cache-only`
to refuse new synthesis on a checkpoint miss, or `--no-cache` for a fresh run.
Checkpoints contain completed raw circuits; there is no recovery point inside
an unfinished synthesis.

## Source organization

```text
src/parity_synthesis/
  api.py                  Public matrix-to-circuit interface
  compiler.py             Recursive circuit assembly and phase carry
  decompositions.py       Cosine-sine and diagonal primitives
  multiplexors.py          Parity-factor and common-word compilation
  parity.py               Full parity-product factorization
  phase_factorization.py  Reduced 11-mask numerical solver
  pairs.py                Dynamic-programming paired-word alignment
  words.py                Exact gate-word algebra and phase conventions
  backends.py             Single-qubit synthesis and even-T routing
  cleanup.py              Exact gate reduction
  simulation.py           Direct circuit-matrix evaluation
  precision.py            Word counts and tolerance allocation
  cache.py                Validated JSON checkpoints
  targets.py              Reproducible random-unitary generation
  benchmark.py / cli.py   Experiment reporting and command-line interface
tests/                    Component and integration checks
examples/                 Small API and factorization examples
benchmarks/               Two seed groups and historical measurements
native/                   Pinned cyclosynth revision and even-T patch
scripts/                  Native setup and API verification
docs/                     Method, architecture, and experiment protocol
third_party/              Upstream native license
```

See [architecture and code-reading order](docs/architecture.md) for module
boundaries and [examples](examples/compile_random_unitary.py) for a minimal run.
The [continuous factorization example](examples/factorize_multiplexor.py) does
not invoke a gate-synthesis backend.

## Validation

The default suite uses bounded matrix, exact-algebra, and mocked-backend checks:

```bash
python -m unittest discover -s tests -v
```

Native search tests are opt-in and require the patched backend:

```bash
PARITY_SYNTHESIS_NATIVE_TESTS=1 python -m unittest tests.test_backends -v
```

The suite checks phase retention, folded pairs, fallback behavior, CSD and
matrix reconstruction, precision allocation, gate cleanup, checkpoint isolation,
the public API, and optional PennyLane recording. These checks do not replace
a fresh end-to-end native benchmark.

## References and provenance

- Yamazaki, Akibue, and Sano:
  [Multi-qubit controlled gate synthesis without T-count overhead in the small-error limit](https://arxiv.org/abs/2603.14202v3).
- Morisaki, Sano, and Akibue:
  [Optimal ancilla-free Clifford+T synthesis for general single-qubit unitaries](https://arxiv.org/abs/2510.05816).
- Shende, Bullock, and Markov:
  [Synthesis of Quantum Logic Circuits](https://arxiv.org/abs/quant-ph/0406176v5).
- Bergholm et al.:
  [Quantum circuits with uniformly controlled one-qubit gates](https://arxiv.org/abs/quant-ph/0410066v2).

The project began with the QHack 2026 unitary-synthesis task. Its two seed groups
are retained as numerical examples; the competition harness is not included.
The native backend is developed independently at
[mtweiden/cyclosynth](https://github.com/mtweiden/cyclosynth). Attribution and
the retained third-party license are described in [NOTICE.md](NOTICE.md).
