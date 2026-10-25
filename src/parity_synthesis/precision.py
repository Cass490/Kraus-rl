"""Per-word precision caps and conservative circuit-budget allocation."""

import numpy as np


def synthesis_word_counts(n, mux_method="baseline"):
    """Planned source words and safe worst case if every reduced solve falls back."""
    if mux_method not in ("baseline", "phase11"):
        raise ValueError(f"unknown multiplexor method: {mux_method}")
    worst = (2 ** n - 1) * 2 ** (n - 1)
    planned = 30 * 12 + 16 if mux_method == "phase11" and n == 5 else worst
    return planned, worst


def word_epsilon(n, eps, error_budget=None):
    """Allocate optional circuit budget over (2**n-1)*2**(n-1) words.

    Unitary products satisfy ||prod U - prod V|| <= sum ||U_j-V_j||.
    Reserve half the budget for numerical decomposition/roundoff and allow
    the words to consume the other half. This is conservative, not a cost
    optimum or a rigorous floating-point certificate.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if error_budget is None:
        return float(eps)
    if not np.isfinite(error_budget) or error_budget <= 0:
        raise ValueError("error budget must be finite and positive")
    words = (2 ** n - 1) * 2 ** (n - 1)
    return min(float(eps), float(error_budget) / (2 * words))
