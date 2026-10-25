# Attribution and provenance

The paired-word and controlled-gate construction builds on Yamazaki, Akibue,
and Sano, arXiv:2603.14202. The single-qubit backend is the independently
developed cyclosynth implementation associated with the synthesis framework
of Morisaki, Sano, and Akibue, arXiv:2510.05816.

The local native modifications add an even-T search constraint. They are
distributed as a patch against the revision in `native/cyclosynth-base.txt`.
The upstream copyright and license notice is retained verbatim in
`third_party/cyclosynth-LICENSE`. Do not remove that notice when redistributing
the patched native source or compiled backend.

Cosine-sine decomposition and uniformly controlled gates are established
linear-algebra and circuit-synthesis tools. The numerical CSD helper was
retained from the supplied QHack 2026 task, together with its reconstruction
convention; this project does not claim to introduce that decomposition.

The seed groups 2026 and 7 and the reference weighted costs originated in
that task. Historical measurements were transcribed from the completed
predecessor implementation, commit
`d6656b585afc92d0da9960db503624760adf8a6b`. They are not fresh measurements
of the modular package, and neither the task's grading code nor precomputed
target circuits are distributed here.

The research focus is the phase-gauged reduced parity model, its numerical
integration and fallback, common-word folding, even-T backend constraint,
and phase-aware implementation. Independent proof review, broader experiments,
and a complete novelty assessment remain open.
