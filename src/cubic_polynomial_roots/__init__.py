"""Cubic Polynomial Roots.

Calculates the real and complex roots of a cubic equation using
Cardano's method.

Exported names:
    CubicPolynomial
    solve_cubic
"""

from .core import CubicPolynomial, solve_cubic

__all__ = ["CubicPolynomial", "solve_cubic"]
