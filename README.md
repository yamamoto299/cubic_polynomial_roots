# Cubic Polynomial Roots

Calculates the real and complex roots of a cubic equation using Cardano's method. Returns exactly three roots (counted with multiplicity) as complex numbers.

```python
from cubic_polynomial_roots import CubicPolynomial, solve_cubic

poly = CubicPolynomial(1.0, -6.0, 11.0, -6.0)
print(poly.roots())   # ((1+0j), (2+0j), (3+0j))

print(solve_cubic(1.0, 0.0, 0.0, -1.0))   # roots of x^3 - 1 = 0
```

## Why this exists

Solving a cubic with real coefficients by hand is error-prone, and the standard-library `cmath` module does not expose a cubic root-finder. This library packages Cardano's method with the three discriminant cases (one real plus conjugate pair, repeated root, three distinct reals) handled separately so that each branch stays numerically stable and readable. The trade-off is simplicity over sophistication: this is float arithmetic, not arbitrary precision, and it targets general engineering use rather than symbolic exactness.

## The awkward edge

In the "three distinct real roots" case (the *casus irreducibilis*) Cardano's radicals are complex, so this library switches to the trigonometric form `t_k = 2*sqrt(-p/3)*cos((theta - 2*pi*k)/3)` to stay in real arithmetic. The three roots come back as plain real complex numbers, but their ordering is not contractually guaranteed — match greedily by distance if order matters to you.

Roots are always returned as `complex`, even when the imaginary part is zero. This keeps the return type uniform across all three discriminant branches. Coefficients must be real; passing complex coefficients is not supported.

Exported names: `CubicPolynomial`, `solve_cubic`.
