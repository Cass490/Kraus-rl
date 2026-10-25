# Synthesis method

## Problem and conventions

The input is a unitary U of dimension 2^n. The output is a time-ordered list
of small gates and a scalar angle phi such that the circuit matrix is
`exp(i*phi) * G_last * ... * G_first`. Wire 0 is the most significant bit.

A gate word has the opposite ordering convention: the string `ABC` means
the matrix product `A @ B @ C`. Reversing the word produces application-time
order. `words.py` makes these conventions explicit and retains exact scalar
phases rather than identifying matrices only projectively.

## Recursive decomposition

Cosine-sine decomposition writes

```text
U = blockdiag(L0,L1) [[C,-S],[S,C]] blockdiag(R0,R1).
```

Here C and S are diagonal cosine and sine matrices. The middle part is a
uniformly controlled Y rotation; recursive decomposition of the side blocks
leaves a network of one-target multiplexors. The construction used here has
2^n-1 multiplexors, including 31 at five qubits. This is an established
synthesis skeleton, not the proposed reduction itself.

## Parity-product representation

For control pattern x and nonzero mask m, let
`p_m(x)=popcount(x & m) mod 2`. A full parity model is

```text
V_x = R product_j beta_j^p_j(x),    R, beta_j in SU(2).
```

The product is ordered: SU(2) factors generally do not commute. A parity can
be computed into an existing control wire with a CNOT ladder, used to select
a branch, then uncomputed. The implementation needs no additional wire.

At four controls, the full model uses all 15 nonzero masks. Its numerical
solver is retained as the fallback in `parity.py`.

## Output-phase freedom and the reduced family

The experimental reduced representation is

```text
V_x = D(z_x) R product_j beta_j^p_j(x),
D(z) = exp(-i*z*Z) = diag(exp(-i*z), exp(i*z)).
```

An output diagonal can be combined with the input of the next multiplexor.
This is a change in the circuit representation, not permission to drop phase.
The ordered masks used for sixteen blocks are
`(1,2,4,8,3,5,6,9,10,12,7)`.

The solver starts from a target-seeded generic model, then follows a blockwise
SU(2) path to the desired target. Its analytic Jacobian is used in least-squares
Newton updates with line search. Starting at identity is avoided because the
reduced parameterization is rank-deficient there.

Two starts share a five-second limit. The final reconstructed blocks must
have operator-norm error at most 1e-12. A declared numerical failure falls
back to the full model before any words are synthesized. Full local rank
and successful finite experiments do not establish global coverage or
universal solver convergence.

## Paired-word compilation and even T parity

A pair `P(A,B)` applies A or B according to the parity control. Joint
compilation can share aligned word layers. For example, `P(T,T)` is simply
an unconditional target T gate, rather than two independent controlled gates.
General pairs need alignment and may add controlled-H overhead.

For a relative word W=X*Y, set A=X_inverse, giving A*W=Y. A balanced split
therefore supplies two branches with roughly half the original T count.
The selected split construction needs compatible parity. The native patch
searches for even-T words directly instead of returning an odd result and
replacing it with a long stand-in.

`pairs.py` uses dynamic-programming alignment and searches simultaneous
Clifford conjugations. It tracks the exact scalar and control phases of the
result. A phase that is globally irrelevant to an isolated one-qubit gate
can become a relative phase when that gate is controlled.

## Common-word folding

Let G be the accumulated exact compiler gauge, W=G*beta*G_dagger, and let
the split pair be P(A,A*W). Its separate common gate is
`C=R*G_dagger*A_dagger`. Branchwise multiplication gives

```text
C P(A,A W) = P(R G_dagger, R beta G_dagger).
```

The compiler synthesizes those two full branch targets and compiles them
jointly. This replaces a split pair plus a common word, while retaining the
independent approximation phases of both new words. It changes joint emitted
cost, not the number of backend word requests by itself.

## Carry and the final boundary

If determinant phase is phi_x, deferred control phase is theta_x, and output
angle is z_x, the outgoing diagonal angles are

```text
d[x,0] = phi_x + theta_x - z_x
d[x,1] = phi_x + theta_x + z_x.
```

These entries are indexed by the actual target and control wire positions.
The next block multiplies the incoming diagonal on its input side. A separate
scalar phase is retained throughout; the optional PennyLane adapter queues
`GlobalPhase(-phi)` because its convention is `exp(-i*angle)`.

The final multiplexor has no subsequent block into which it can carry new
output Z freedom. It deliberately uses the full model, keeping the last
diagonal independent of one wire. At five qubits this leaves 15 arbitrary RZ
rotations rather than a generic diagonal with 31. With no reduced-model
fallbacks, the planned word count is `30*12+16=376`, versus `31*16=496`.

## Precision and exact cleanup

Per-word tolerance is distinct from whole-circuit error. For unitary factors,
the telescoping operator-norm bound is
`||product U_j - product V_j|| <= sum_j ||U_j-V_j||`. The measured entrywise
error can be much smaller than this sum, but cancellation is not a guarantee.

Word checks subtract phase-aligned matrices directly. Trace overlap alone
loses small discrepancies quadratically and can round to an exact-looking
value near the required precision. The final matrix comparison includes
global phase. The optional budget policy is conservative, not an
interval-arithmetic certificate of all floating-point operations.

Cleanup applies exact Clifford rewrites, dependency-respecting cancellations,
Pauli-frame propagation, and short two-qubit Clifford replacements. Removed
scalar phases are returned explicitly. This changes physical gate count
without intentionally adding approximation error. Floating-point matrix
evaluation and the diagonal emitter's tiny-angle threshold still require
numerical validation.
