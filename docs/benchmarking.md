# Experiment protocol and evidence

## Two reproducible seed groups

`benchmarks/cases.json` defines groups 2026 and 7. Each group contains two
samples at each size n in 2,3,4,5, giving sixteen distinct target circuits.
The target seed is `group_seed + 1000*n + sample_index`. Complex Gaussian QR
generates the Haar-random unitary. These are independent inputs to the compiler,
not a lookup table of precomputed answers.

Run the configured experiment explicitly:

```bash
parity-synthesis benchmark --config benchmarks/cases.json --output .cache/benchmark.json
```

This can take tens of minutes or longer. The setup script and default unit
tests do not launch it. `--cache-only` refuses fresh searches; `--no-cache`
requests a cold synthesis run. The report records checkpoint reuse per sample.
For a reproducible timing study, also record CPU, core count, thread settings,
memory, backend build, and whether operating-system/build caches were warm.

## Metrics

Report the actual T count, remaining rotation count, and physical operation
count separately. The historical weighted resource measure is

```text
C = N_T + 44*(N_RX + N_RY + N_RZ).
```

The 44 is a reporting convention, not a universal hardware cost. Historical
reference values are 660, 3168, 13728, and 57024. They are preserved as data,
not used as a grading rule or a claim about the strongest available compiler.
The CLI allows a different rotation weight; savings against the historical
reference are omitted when the weights do not match.

Accuracy is the largest absolute matrix-entry difference, including global
phase. This is distinct from operator norm or diamond distance. The benchmark
records whether it meets a requested tolerance, default 1e-8, but assigns no
competition grade. A per-word epsilon is not a full-circuit error guarantee.

## Preserved historical result

`benchmarks/results/reference_seed7.json` transcribes the final measured report
from the predecessor implementation. Its settings were phase11, common-word
folding, and per-word epsilon 1e-9. At five qubits the mean cost was 18048
against 57024: about 68.3502% reduction, displayed as 68.3% by the original
reporting convention. Across that seed group the largest reported entry error
was 5.4e-9. No auxiliary wires were used.

The last individual five-qubit sample used 17388 T gates and fifteen RZ
rotations, giving `17388 + 44*15 = 18048`, with 66326 physical operations
including GlobalPhase. Individual sample gate counts are not averages over
both targets. The recorded elapsed time, 1543 seconds, covers only seed group
7. The complete final report for group 2026, raw terminal logs, full-suite
timing, and a documented cold-cache hardware protocol were not preserved.

These measurements came before the modular reorganization. The package has
bounded component, API, and extraction-equivalence checks; a fresh native
end-to-end run of this distribution remains to be performed. Do not relabel
the historical data as new measurements.

## What would support stronger conclusions

- Compare baseline, folding-only, reduced-model-only, and combined variants on
  identical targets, backends, and cleanup settings.
- Distinguish equal per-word epsilon from matched final circuit error.
- Evaluate more random and structured targets and retain failures, fallback
  rates, maximum costs, and errors, not just successful means.
- Synthesize the remaining RZ rotations before making a pure Clifford+T
  T-count comparison.
- Compare contemporary compilers using the same gate set, error metric, and
  hardware/timing protocol.
- Review local expressivity and global reachability separately; a local-rank
  observation or finite collection of fits is not a universal coverage proof.
