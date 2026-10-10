"""Cardano's method for finding the roots of a cubic polynomial."""

from __future__ import annotations

import cmath
import math

__all__ = ["CubicPolynomial", "solve_cubic"]


def _as_cplx(z):
    """Return ``z`` as a complex number, normalising a -0.0 imaginary part
    to +0.0.

    Cardano's method multiplies by cube roots of unity, which often
    introduces a tiny imaginary component on what should be a real root.
    When we snap such a value back to a clean real number we still want it
    to compare equal to a plain float root, and Python treats ``-0.0 ==
    0.0`` as True — but a negative-zero imaginary part in a complex
    number is noisy when printed. Normalising here keeps the output tidy
    without changing equality semantics.
    """
    if isinstance(z, complex):
        return complex(z.real, 0.0 if z.imag == 0.0 else z.imag)
    return complex(z)


class CubicPolynomial:
    """A cubic polynomial ``a x^3 + b x^2 + c x + d``.

    Parameters
    ----------
    a, b, c, d : float
        Coefficients. ``a`` must be non-zero; everything else is free.
        Integers are accepted and coerced to floats.
    """

    def __init__(self, a, b, c, d):
        a = float(a)
        b = float(b)
        c = float(c)
        d = float(d)
        if a == 0.0:
            raise ValueError("coefficient 'a' must be non-zero for a cubic")
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def roots(self):
        """Return the three roots as a tuple of complex numbers.

        Pure-real roots are returned as plain ``complex`` numbers with a
        zero imaginary part. We always return exactly three roots counted
        with multiplicity, which is what the fundamental theorem of
        algebra guarantees for a degree-three polynomial with non-zero
        leading coefficient.
        """
        return solve_cubic(self.a, self.b, self.c, self.d)


def solve_cubic(a, b, c, d):
    """Return the three roots of ``a x^3 + b x^2 + c x + d``.

    Parameters
    ----------
    a, b, c, d : float
        Real coefficients; ``a`` must be non-zero.

    Returns
    -------
    tuple of (complex, complex, complex)
        The three roots, counted with multiplicity. Pure-real roots are
        returned as complex numbers with a zero imaginary part.

    Notes
    -----
    The implementation is Cardano's method applied to the depressed cubic
    ``t^3 + p t + q = 0`` obtained by the substitution ``x = t - b/(3a)``.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    if a == 0.0:
        raise ValueError("coefficient 'a' must be non-zero for a cubic")

    # Depress: x = t - b/(3a). The resulting cubic in t is
    #   t^3 + p t + q = 0
    # with p and q as below. Working in floats keeps arithmetic simple and
    # is adequate for the accuracy this library targets.
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)

    shift = -b / (3.0 * a)

    return _depressed_roots(p, q, shift)


def _depressed_roots(p, q, shift):
    """Roots of ``t^3 + p t + q = 0`` then shifted by ``shift``.

    ``p`` and ``q`` are real. The classical discriminant is
    ``Delta = -4 p^3 - 27 q^2``; for the numerics below it is more
    convenient to work with its scaled, sign-flipped form
    ``D = (q/2)^2 + (p/3)^3``, which is positive when there is one real
    root and a complex-conjugate pair, zero at a repeated root, and
    negative when there are three distinct real roots (the "casus
    irreducibilis").
    """
    # The truly-degenerate cubic: t^3 + q = 0, i.e. p = 0 exactly. We
    # short-circuit here because the general formula divides by p and
    # because this branch naturally captures the triple-root case
    # q = 0 (giving three zeros).
    if p == 0.0:
        if q == 0.0:
            r = _as_cplx(shift)
            return (r, r, r)
        # t^3 = -q  =>  one real cube root, two complex ones obtained by
        # multiplying by the non-trivial cube roots of unity.
        r0 = math.copysign(abs(q) ** (1.0 / 3.0), -q)
        w = cmath.exp(2j * math.pi / 3.0)
        w2 = w * w
        t0 = complex(r0, 0.0)
        t1 = r0 * w
        t2 = r0 * w2
        return (
            _as_cplx(t0 + shift),
            _as_cplx(t1 + shift),
            _as_cplx(t2 + shift),
        )

    half_q = q / 2.0
    third_p = p / 3.0
    disc = half_q * half_q + third_p * third_p * third_p

    if disc > 0.0:
        # One real root and a complex-conjugate pair. Cardano's formula
        # gives the real root directly from real cube roots; the pair comes
        # from the non-real cube roots of the same radicands.
        sqrt_disc = math.sqrt(disc)
        u_cubed = -half_q + sqrt_disc
        v_cubed = -half_q - sqrt_disc
        u = math.copysign(abs(u_cubed) ** (1.0 / 3.0), u_cubed)
        v = math.copysign(abs(v_cubed) ** (1.0 / 3.0), v_cubed)

        t1 = u + v
        real_part = -(u + v) / 2.0
        imag_part = (u - v) * math.sqrt(3.0) / 2.0

        return (
            _as_cplx(complex(t1 + shift, 0.0)),
            _as_cplx(complex(real_part + shift, imag_part)),
            _as_cplx(complex(real_part + shift, -imag_part)),
        )

    if disc == 0.0:
        # Repeated root. When q != 0 we have one simple and one double
        # root; when q == 0 (and p != 0, guaranteed by the early branch) we
        # have a triple root at the origin.
        if q == 0.0:
            r = _as_cplx(shift)
            return (r, r, r)
        # With disc == 0 we have u = v = cube_root(-q/2), so Cardano's
        # formulas collapse to a double root at -u and a simple root at
        # 2u.
        u = math.copysign(abs(-half_q) ** (1.0 / 3.0), -half_q)
        double = -u
        single = 2.0 * u
        return (
            _as_cplx(complex(single + shift, 0.0)),
            _as_cplx(complex(double + shift, 0.0)),
            _as_cplx(complex(double + shift, 0.0)),
        )

    # disc < 0: three distinct real roots (casus irreducibilis). Cardano's
    # radicals are complex here, so we use the trigonometric form instead,
    # which keeps everything in real arithmetic and is numerically stable
    # for this case. The pi/3 spacing of the three arccosine branches is the
    # standard choice and yields a strictly ordered triple of real roots.
    mp = -p / 3.0
    sqrt_mp = math.sqrt(mp)
    ratio = -q / (2.0 * mp * sqrt_mp)
    ratio = max(-1.0, min(1.0, ratio))
    theta = math.acos(ratio)

    t_k = []
    for k in range(3):
        angle = (theta - 2.0 * math.pi * k) / 3.0
        tk = 2.0 * sqrt_mp * math.cos(angle)
        t_k.append(complex(tk + shift, 0.0))

    return (_as_cplx(t_k[0]), _as_cplx(t_k[1]), _as_cplx(t_k[2]))
