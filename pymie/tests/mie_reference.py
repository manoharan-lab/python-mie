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
functions. The functions in this module are used only to generate reference
values for checking the accuracy of pymie; they are not used in pymie itself.
Running code that uses these functions requires installing mpmath, which is not
installed in the environment specified by the root-level environment.yml file.

The functions are based on conventions and formulas from Bohren and Huffman.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
   (with assistance from Claude Opus 5.5)
"""
import mpmath


def riccati_psi_xi(z, nstop):
    """Calculates Riccati-Bessel functions for n = 0, ..., nstop using
    mpmath's Bessel functions.  From Bohren and Huffman p. 101,

    psi_n(z) = z j_n(z)
    xi_n(z) = z h1_n(z)

    where j_n(z) is the spherical Bessel function of order n (B&H equation 4.9)
    and h1_n(z) is the spherical Hankel function of the first kind (B&H
    equation 4.13).

    """
    psi = []
    xi = []
    for n in range(nstop + 1):
        # B&H 4.9 gives j_n(z) = sqrt(pi / (2z)) J_nu(z) where nu = n+1/2
        #           and y_n(z) = sqrt(pi / (2z)) Y_nu(z)
        # we'll need both kinds of Bessel functions to calculate h1_n(x)
        nu = n + mpmath.mpf(1)/2
        prefactor = mpmath.sqrt(mpmath.pi*z/2)  # including factor of z
        j_nu = mpmath.besselj(nu, z)
        y_nu = mpmath.bessely(nu, z)
        psi.append(prefactor * j_nu)
        # B&H 4.13 gives h1_n(z) = j_n(z) + iy_n(z)
        xi.append(prefactor * (j_nu + y_nu*1j))

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
    psi_x, xi_x = riccati_psi_xi(x, nstop)
    psi_mx, _ = riccati_psi_xi(m*x, nstop)
    a = []
    b = []
    for n in range(1, nstop + 1):
        # calculate derivatives using recurrence relations (equations are below
        # B&H eq 4.88):
        #     psi'_n(x) = psi_{n-1}(x) - n psi_n(x)/x
        #     xi'_n(x) = xi_{n-1}(x) - n xi_n(x)/x
        dpsi_x = psi_x[n-1] - n * psi_x[n]/x
        dxi_x = xi_x[n-1] - n * xi_x[n]/x
        dpsi_mx = psi_mx[n-1] - n * psi_mx[n]/(m*x)

        an = ((m * psi_mx[n] * dpsi_x - psi_x[n] * dpsi_mx)/
              (m * psi_mx[n] * dxi_x  - xi_x[n] * dpsi_mx))
        bn = ((psi_mx[n] * dpsi_x - m * psi_x[n] * dpsi_mx)/
              (psi_mx[n] * dxi_x  - m * xi_x[n] * dpsi_mx))
        a.append(an)
        b.append(bn)
    return a, b
