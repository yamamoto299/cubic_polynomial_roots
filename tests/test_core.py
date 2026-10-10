"""Tests for the cubic-polynomial Cardano solver.

All float comparisons use a tolerance rather than ``==`` because the
underlying cube-root and trigonometric primitives are not bit-exact.
Roots are compared as complex numbers with the same tolerance.
"""

import math
import unittest

from cubic_polynomial_roots import CubicPolynomial, solve_cubic


def cabs(z):
    """Complex absolute value that also accepts ints/floats."""
    return abs(complex(z))


def roots_close(roots, expected, tol=1e-9):
    """Return True if the two multisets of roots match within ``tol``.

    We match greedily: for each expected root we find the closest actual
    root and require it to be within tolerance, then remove that actual
    root so it cannot be matched twice. This avoids assuming any
    particular ordering of the roots returned by the solver.
    """
    if len(roots) != len(expected):
        return False
    pool = list(roots)
    for e in expected:
        best_i = -1
        best_d = math.inf
        for i, r in enumerate(pool):
            d = cabs(complex(r) - complex(e))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0 or best_d > tol:
            return False
        pool.pop(best_i)
    return True


class TestSolveCubic(unittest.TestCase):
    def test_three_distinct_real_roots(self):
        # x^3 - 6 x^2 + 11 x - 6 = (x-1)(x-2)(x-3)
        roots = solve_cubic(1.0, -6.0, 11.0, -6.0)
        self.assertTrue(roots_close(roots, [1.0, 2.0, 3.0]))

    def test_one_real_and_complex_pair(self):
        # x^3 - 1 = 0 => roots 1, (-1 +/- i*sqrt(3))/2
        roots = solve_cubic(1.0, 0.0, 0.0, -1.0)
        expected = [1.0, complex(-0.5, math.sqrt(3.0) / 2.0), complex(-0.5, -math.sqrt(3.0) / 2.0)]
        self.assertTrue(roots_close(roots, expected))

    def test_double_root(self):
        # x^3 - 3 x + 2 = (x-1)^2 (x+2)
        roots = solve_cubic(1.0, 0.0, -3.0, 2.0)
        self.assertTrue(roots_close(roots, [1.0, 1.0, -2.0]))

    def test_triple_root(self):
        # x^3 = 0
        roots = solve_cubic(1.0, 0.0, 0.0, 0.0)
        self.assertTrue(roots_close(roots, [0.0, 0.0, 0.0]))

    def test_triple_root_shifted(self):
        # (x-5)^3 = x^3 - 15 x^2 + 75 x - 125
        roots = solve_cubic(1.0, -15.0, 75.0, -125.0)
        self.assertTrue(roots_close(roots, [5.0, 5.0, 5.0]))

    def test_all_complex(self):
        # x^3 + 1 = 0 => roots -1, (1 +/- i*sqrt(3))/2
        roots = solve_cubic(1.0, 0.0, 0.0, 1.0)
        expected = [-1.0, complex(0.5, math.sqrt(3.0) / 2.0), complex(0.5, -math.sqrt(3.0) / 2.0)]
        self.assertTrue(roots_close(roots, expected))

    def test_negative_leading(self):
        # -2 x^3 + 12 x^2 - 22 x + 12 = -2 (x-1)(x-2)(x-3)
        roots = solve_cubic(-2.0, 12.0, -22.0, 12.0)
        self.assertTrue(roots_close(roots, [1.0, 2.0, 3.0]))

    def test_pure_linear(self):
        # x^3 - 5x = x(x - sqrt5)(x + sqrt5): p != 0, q == 0
        roots = solve_cubic(1.0, 0.0, -5.0, 0.0)
        s = math.sqrt(5.0)
        self.assertTrue(roots_close(roots, [0.0, s, -s]))

    def test_integer_inputs_accepted(self):
        # Coercion from int to float should not raise and should give the
        # same answer as the float version.
        roots = solve_cubic(1, -6, 11, -6)
        self.assertTrue(roots_close(roots, [1.0, 2.0, 3.0]))

    def test_depressed_three_real(self):
        # x^3 - 7 x + 6 = (x-1)(x-2)(x+3): depressed cubic, three real roots
        roots = solve_cubic(1.0, 0.0, -7.0, 6.0)
        self.assertTrue(roots_close(roots, [1.0, 2.0, -3.0]))

    def test_large_shift(self):
        # (x-1000)(x-1001)(x-1002): b is large, exercises the depression shift
        a3 = 1.0
        b3 = -(1000.0 + 1001.0 + 1002.0)
        c3 = 1000.0 * 1001.0 + 1000.0 * 1002.0 + 1001.0 * 1002.0
        d3 = -(1000.0 * 1001.0 * 1002.0)
        roots = solve_cubic(a3, b3, c3, d3)
        self.assertTrue(roots_close(roots, [1000.0, 1001.0, 1002.0], tol=1e-6))


class TestCubicPolynomial(unittest.TestCase):
    def test_roots_method_matches_function(self):
        poly = CubicPolynomial(1.0, -6.0, 11.0, -6.0)
        self.assertTrue(roots_close(poly.roots(), [1.0, 2.0, 3.0]))

    def test_stores_coefficients(self):
        poly = CubicPolynomial(2.0, 3.0, 5.0, 7.0)
        self.assertEqual(poly.a, 2.0)
        self.assertEqual(poly.b, 3.0)
        self.assertEqual(poly.c, 5.0)
        self.assertEqual(poly.d, 7.0)

    def test_int_coercion(self):
        poly = CubicPolynomial(1, -6, 11, -6)
        self.assertEqual(poly.a, 1.0)
        self.assertTrue(roots_close(poly.roots(), [1.0, 2.0, 3.0]))

    def test_rejects_zero_leading(self):
        with self.assertRaises(ValueError):
            CubicPolynomial(0.0, 1.0, 2.0, 3.0)

    def test_rejects_zero_leading_function(self):
        with self.assertRaises(ValueError):
            solve_cubic(0.0, 1.0, 2.0, 3.0)


class TestRootShape(unittest.TestCase):
    def test_returns_three_complex(self):
        roots = solve_cubic(1.0, 0.0, 0.0, -1.0)
        self.assertEqual(len(roots), 3)
        for r in roots:
            self.assertIsInstance(r, complex)

    def test_real_root_is_complex(self):
        # The real root of x^3 - 1 = 0 should still be returned as a
        # complex number with zero imaginary part, not as a float.
        roots = solve_cubic(1.0, 0.0, 0.0, -1.0)
        # The real root is 1+0j; the other two are complex.
        real_roots = [r for r in roots if abs(r.imag) < 1e-12]
        self.assertEqual(len(real_roots), 1)
        self.assertAlmostEqual(real_roots[0].real, 1.0)
        self.assertAlmostEqual(real_roots[0].imag, 0.0)


if __name__ == "__main__":
    unittest.main()
