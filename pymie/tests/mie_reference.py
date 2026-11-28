# Copyright 2026, Vinothan N. Manoharan
#
# This file is part of the python-mie python package.
#
# This package is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This package is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this package. If not, see <https://www.gnu.org/licenses/>.
"""
Reference Mie scattering implementation using arbitrary-precision special
functions. The goal of this implementation is to provide *accurate* results for
Mie calculations with code that can be readily verified to match theoretical
formulas.

To this end, I've written the functions to correspond, as closely as possible,
to the relevant equations in B&H, taking advantage of mpmath's special function
implementations, rather than computing successive orders by recurrence.
Production Mie codes use recurrence relations, which are fast and (when run in
the right direction) numerically stable in double precision. But with
arbitrary-precision functions, we don't need to use them, so long as we
maintain sufficiently high precision. Also, writing the code in terms of
mpmath's special functions should make it easier to verify that the theory is
implemented correctly.

Some caveats:

1. We are relying on the correctness and accuracy of mpmath's special function
   implementations.
2. Some care is needed to ensure that prefactors like
       (2*n + 1)/(n*(n+1))
   yield arbitrary precision numbers rather than ordinary floats. In this case,
   the prefactor should be written as
       mpmath.mpf(2*n + 1)/(n*(n+1)).
   It's important to check for cases like this where precision may be lost.
3. This module does not set the working precision. The calling function or
   script should do so using a context manager such as
       `with mpmath.workdps(30):`.
   mpmath's default precision is only 15 digits, which is equivalent to double
   precision.
4. This implementation is not fast, and it's not meant to be. Accuracy and
   transparency are the primary goals. Be warned that calculating coefficients
   for large size parameters (order 1000) can take minutes.

Note that the functions in this module are used only to generate reference
values for checking the accuracy of pymie; they are not used in pymie itself.
Running code that uses these functions requires installing mpmath, which is not
installed in the environment specified by the root-level environment.yml file.
Install mpmath manually to use these functions.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
   (with assistance from Claude Opus 5.5)
"""
import mpmath


def riccati_psi(z, n):
    """Riccati-Bessel function psi_n(z) using mpmath's Bessel function. From
    B&H p. 101,

        psi_n(z) = z j_n(z)

    where j_n(z) is the spherical Bessel function of order n. We use B&H
    equation 4.9 to express psi_n(z) in terms of the Bessel function J_nu(z):

        j_n(z) = sqrt(pi / (2z)) J_nu(z) where nu = n+1/2

    """
    # prefactor includes factor of z in z*j_n(z)
    return mpmath.sqrt(mpmath.pi*z/2) * mpmath.besselj(n + mpmath.mpf(1)/2, z)


def riccati_xi(z, n):
    """Riccati-Bessel function xi_n(z) using mpmath's Hankel function. From B&H
    p. 101,

        xi_n(z) = z h1_n(z)

    where h1_n(z) is the spherical Hankel function of the first kind of order
    n.  The spherical Hankel function is defined analogously to the spherical
    Bessel function:

        h1_n(z) = sqrt(pi / (2z)) H1_nu(z) where nu = n+1/2

    where H1_nu(z) is the Hankel function of the first kind. Though B&H does
    not show this equation, it follows from eq. 4.13, 4.9, 4.10, and the
    definition of H1_n at the bottom of page 197 (H1_n = J_n + iY_n).

    Using mpmath's Hankel function is useful because it reduces numerical error
    for complex z with large imaginary part. J_nu and Y_nu grow exponentially
    with Im(z), whereas H1_nu decays, so the sum J_nu + iY_nu would lose digits
    due to cancellation. mpmath's implementation of H1_nu does not have this
    problem.

    """
    # prefactor includes factor of z in z*h1_n(z)
    return mpmath.sqrt(mpmath.pi*z/2) * mpmath.hankel1(n + mpmath.mpf(1)/2, z)


def riccati_psi_xi(z, nstop):
    """Calculates Riccati-Bessel functions for n = 0, ..., nstop.  This
    function is currently used only for convenience in comparing the results to
    the pymie function of the same name.

    """
    psi = []
    xi = []
    for n in range(nstop + 1):
        psi.append(riccati_psi(z, n))
        xi.append(riccati_xi(z, n))

    return psi, xi


def scatcoeffs(m, x, nstop):
    """Calculates Mie scattering coefficients a_n and b_n for n = 1, ..., nstop
    from Bohren and Huffman eqs. 4.56 and 4.57. As B&H note, these equations
    are not typically used in computing the scattering coefficients because of
    round-off error accumulation.  But with mpmath we can increase the
    precision to reduce round-off error.

            m psi_n(mx) psi'_n(x) - psi_n(x) psi'_n(mx)
      a_n = -------------------------------------------
            m psi_n(mx) xi'_n(x)  - xi_n(x) psi'_n(mx)

            psi_n(mx) psi'_n(x) - m psi_n(x) psi'_n(mx)
      b_n = ---------------------------------------------
            psi_n(mx) xi'_n(x)  - m xi_n(x) psi'_n(mx)

    m and x should be specified as mpmath float or complex.

    """
    a = []
    b = []
    for n in range(1, nstop + 1):
        psi_x = riccati_psi(x, n)
        xi_x = riccati_xi(x, n)
        psi_mx = riccati_psi(m*x, n)

        # calculate derivatives using mpmath.diff, which adjusts its working
        # precision to keep finite differences from losing digits
        dpsi_x = mpmath.diff(lambda z: riccati_psi(z, n), x)
        dxi_x = mpmath.diff(lambda z: riccati_xi(z, n), x)
        dpsi_mx = mpmath.diff(lambda z: riccati_psi(z, n), m*x)

        an = ((m * psi_mx * dpsi_x - psi_x * dpsi_mx)/
              (m * psi_mx * dxi_x  - xi_x * dpsi_mx))
        bn = ((psi_mx * dpsi_x - m * psi_x * dpsi_mx)/
              (psi_mx * dxi_x  - m * xi_x * dpsi_mx))
        a.append(an)
        b.append(bn)
    return a, b


def internal_coeffs(m, x, nstop):
    """Calculates Mie internal-field coefficients c_n and d_n for
    n = 1, ..., nstop from Bohren and Huffman eq. 4.52 and taking mu_1 = mu
    (same magnetic permeability in particle and medium)

              j_n(x) [x h_n(x)]' - h_n(x) [x j_n(x)]'
      c_n = ------------------------------------------
            j_n(mx) [x h_n(x)]' - h_n(x) [mx j_n(mx)]'

              m j_n(x) [x h_n(x)]' - m h_n(x) [x j_n(x)]'
      d_n = ----------------------------------------------,
            m^2 j_n(mx) [x h_n(x)]' - h_n(x) [mx j_n(mx)]'

    where j_n is the spherical Bessel function of the first kind, h_n is the
    spherical Hankel function of the first kind, and the primes denote
    derivatives with respect to the argument in brackets. We write the
    spherical functions in terms of the Riccati-Bessel functions, j_n(x) =
    psi_n(x)/x and h_n(x) = xi_n(x)/x, so that [x j_n(x)]' = psi'_n(x),
    [x h_n(x)]' = xi'_n(x), and [mx j_n(mx)]' = psi'_n(mx).  Then

              j_n(x) xi'_n(x) - h_n(x) psi'_n(x)
      c_n = --------------------------------------
             j_n(mx) xi'_n(x) - h_n(x) psi'_n(mx)

              m j_n(x) xi'_n(x) - m h_n(x) psi'_n(x)
      d_n = ------------------------------------------,
             m^2 j_n(mx) xi'_n(x) - h_n(x) psi'_n(mx)

    m and x should be specified as mpmath float or complex.

    """
    c = []
    d = []
    for n in range(1, nstop + 1):
        j_x = riccati_psi(x, n)/x
        h_x = riccati_xi(x, n)/x
        j_mx = riccati_psi(m*x, n)/(m*x)

        # derivatives of x j_n(x), x h_n(x), and mx j_n(mx) with respect to
        # their arguments, using mpmath.diff() as in scatcoeffs()
        dpsi_x = mpmath.diff(lambda z: riccati_psi(z, n), x)
        dxi_x = mpmath.diff(lambda z: riccati_xi(z, n), x)
        dpsi_mx = mpmath.diff(lambda z: riccati_psi(z, n), m*x)

        cn = ((j_x  * dxi_x - h_x * dpsi_x) /
              (j_mx * dxi_x - h_x * dpsi_mx))
        dn = ((m    * j_x * dxi_x  - m * h_x * dpsi_x) /
              (m**2 * j_mx * dxi_x  -   h_x * dpsi_mx))
        c.append(cn)
        d.append(dn)

    return c, d


def efficiencies(a, b, x):
    """Calculates scattering, extinction, and backscattering efficiencies from
    the scattering coefficients, using B&H eqs. 4.61 and 4.62, plus the
    equation at the top of p. 122 for the radar backscattering efficiency.
    Following the convention in pymie, we divide this value by (4 pi) to get
    the backscattering efficiency.  Also calculates the asymmetry parameter g
    from formula on B&H page 120.

    """
    qsca = 0
    qext = 0
    sum_back = 0
    for n, (an, bn) in enumerate(zip(a, b), start=1):
        qsca += (2*n + 1) * (abs(an)**2 + abs(bn)**2)
        qext += (2*n + 1) * (an + bn).real
        sum_back += (2*n + 1) * (-1)**n * (an - bn)
    qsca = 2/x**2 * qsca
    qext = 2/x**2 * qext
    qback = 1/x**2 * abs(sum_back)**2/(4*mpmath.pi)
    asym = asymmetry_parameter_unnormalized(a, b)
    g = 4 * asym/(x**2 * qsca)
    return qsca, qext, qback, g


def asymmetry_parameter_unnormalized(a, b):
    """Calculates the unnormalized sum for the asymmetry parameter from the
    scattering coefficients using equation on B&H page 120. To recover g,
    multiply by 4/(x^2 Q_sca).

    """
    # term for n = 1
    asym = 3/2 * (a[0]*mpmath.conj(b[0])).real
    for n in range(2, len(a)+1):
        # remember that a[n-1] (python indexing) = a_n (math notation)
        k = n-1
        # wrap numerators so that division yields arbitrary precision floats
        asym += (mpmath.mpf(k*(k + 2))/(k + 1)
                 * (a[k-1]*mpmath.conj(a[k]) + b[k-1]*mpmath.conj(b[k])).real)
        asym += (mpmath.mpf(2*n + 1)/(n*(n+1))
                 * (a[n-1]*mpmath.conj(b[n-1])).real)
    return asym


def pis_and_taus(theta, nstop):
    """Calculate the angular functions pi_n and tau_n for n = 1, ..., nstop at
    scattering angle theta (in radians) from Legendre polynomials P_n.  From
    B&H eq. 4.46,
                   P1_n(mu)
        pi_n =  ---------------
                  sin(theta)

        tau_n = dP1_n(mu)/d(theta)

    where mu = cos(theta) and P1_n = -dP_n/d(theta) (eq. 4.30), where P_n is
    the Legendre polynomial of degree n.  We rewrite the equations in terms of
    mu to avoid dividing by sin(theta) at theta = 0 and pi. Then

        P1_n(mu) = sqrt(1 - mu^2) dP_n(mu)/d(mu)

    (cf. B&H eq 4.25) and

        pi_n = dP_n/d(mu)

    since d(mu)/d(theta) = -sin(theta), and

        tau_n = mu dP_n/d(mu) - (1 - mu^2) d^2P_n/d(mu)^2

    by the chain rule.

    B&H use recurrence relations (eq. 4.47) to compute pi_n and tau_n but we
    use numerical derivatives with arbitrary precision to write the functions
    in a simpler way.

    Note that we don't insert pi_0 into the returned pi values (since it's
    zero). So pi[0] = pi_1 and tau[0] = tau_1.

    """
    mu = mpmath.cos(theta)
    pi = [ ]
    tau = [ ]
    for n in range(1, nstop+1):
        # calculate first and second derivatives of P_n(mu) and evaluate at mu
        # (note first output is the value of P_n(mu), which we don't need)
        _, dP, d2P = mpmath.diffs(lambda z: mpmath.legendre(n, z), mu, 2)
        pi.append(dP)
        tau.append(mu * dP - (1-mu**2) * d2P)
    return pi, tau


def amplitude_scattering_matrix(a, b, theta):
    """Calculates the amplitude scattering matrix elements S1 and S2 at
    scattering angle theta (in radians) from the scattering coefficients a_n
    and b_n, using B&H 4.74:

        S_1 = sum_n (2n+1)/(n(n+1)) (a_n pi_n + b_n tau_n)
        S_2 = sum_n (2n+1)/(n(n+1)) (a_n tau_n + b_n pi_n)

    """
    pi, tau = pis_and_taus(theta, len(a))
    S1 = 0
    S2 = 0
    for n in range(1, len(a) + 1):
        prefactor = mpmath.mpf(2*n+1)/(n*(n+1))
        # index of arrays here is n-1 because a_1 is the first element in the
        # array (a[0]=a_1) and likewise for the other arrays
        S1 += prefactor * (a[n-1] * pi[n-1] + b[n-1] * tau[n-1])
        S2 += prefactor * (a[n-1] * tau[n-1] + b[n-1] * pi[n-1])
    return S1, S2
