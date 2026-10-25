# Architecture and reading guide

## Public entry point

Start with `api.compile_unitary` and `CompiledCircuit`. The input is a NumPy
matrix; no template, submission callback, simulator device, or benchmark
configuration is required. Input validation rejects nonfinite, nonsquare,
non-power-of-two, or nonunitary matrices before backend construction.

`CompiledCircuit` stores the gate list and scalar phase. Its `matrix`,
`error_against`, `statistics`, and optional `queue` methods support inspection
without coupling the compiler to a particular experiment or framework.

## Algorithm boundaries

| Module | Responsibility |
| --- | --- |
| `decompositions` | Numerical CSD, block ordering, diagonal primitives, wire indices |
| `compiler` | Recursive circuit assembly and incoming/outgoing phase carry |
| `multiplexors` | Factor selection, paired-word targets, folding, phase corrections |
| `parity` | Full parity-product continuation solver |
| `phase_factorization` | Four-control reduced model and bounded analytic-Jacobian solve |
| `pairs` | Exact skeleton alignment, Clifford conjugations, pair emission |
| `words` | Matrix-order words, inverses, normal form, scalar phase, word error |
| `backends` | Joint native synthesis, even-T routing, compatibility backend |
| `precision` | Planned/worst word counts and optional error-budget allocation |
| `cleanup` | Exact local Clifford reductions and phase accounting |
| `emission` | Concrete time-ordered gates and optional PennyLane recording |
| `simulation` | Direct matrix evaluation, independent of synthesis and grading |

The extracted numerical routines retain the predecessor algorithm. Modules
import their collaborators explicitly; there is no shared monolithic runner
namespace or source-text execution. The public API handles single-qubit inputs
directly and uses the recursive compiler for larger matrices.

## Checkpoint lifecycle

The API fingerprints the target, installed backend, source files, precision,
method, folding setting, and budget. A checkpoint stores a completed raw
circuit before cleanup. Loading a valid hit avoids constructing a synthesizer;
cleanup and later matrix evaluation can still be performed.

The new package uses its own cache identity and does not silently load the
predecessor's checkpoint directory. Callable experimental backends cannot use
persistent caching because their identity cannot be safely inferred. No
checkpoint is written midway through an unfinished word or multiplexor solve.

## Experiments are separate

`targets` generates random matrices. `benchmark` iterates seed groups and
reports per-sample measurements and aggregate statistics. `cli` exposes those
operations but is not imported by the compiler. Configurations contain data,
not executable source. There are no competition grades or gate/wire acceptance
policies in the package.

The optional weighted resource metric has a configurable rotation weight.
Changing that weight disables comparison to reference costs recorded under a
different weight. Physical counts, T counts, and remaining rotations are
reported separately.

## Suggested reading order

1. Read the short API example in the README and inspect `CompiledCircuit`.
2. Read `words`, then the branchwise folding identity in `multiplexors`.
3. Trace incoming and outgoing carry in `compiler.synthesize`.
4. Compare `parity.model` with `phase_factorization.reconstruct`.
5. Study the reduced solver's Jacobian and continuation controls.
6. Read the paired-word and carry tests before changing phase conventions.
7. Use the benchmark protocol only after the bounded tests pass.

The exact gate cleanup is a cohesive subsystem kept in its own module. The
native implementation remains a pinned upstream dependency plus a patch,
rather than being duplicated in the Python package.
