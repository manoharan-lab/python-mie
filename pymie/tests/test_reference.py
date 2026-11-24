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
Tests the Mie reference implementation built using the arbitrary-precision
special functions in mpmath.  Requires mpmath to be installed.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
   (with assistance from Claude Opus 5.5)
"""
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest

mpmath = pytest.importorskip("mpmath")
from . import mie_reference     # noqa: E402

# highest order calculated
NSTOP = 30

# values of mx to test, with complex values corresponding to absorbing
# particles
ZVALS = [0.1, 1.0, 10.0, 100.0, 1000.0, 1.2 + 1e-3j, 12.0 + 0.5j]


# calculate once at all orders and for all values of z
@pytest.fixture(scope="module")
def reference_psi_xi():
    """Calculate Riccati-Bessel functions from reference implementation and
    convert to double precision.

    """
    values = {}
    # raise precision to 30 digits
    with mpmath.workdps(30):
        for z in ZVALS:
            psi, xi = mie_reference.riccati_psi_xi(mpmath.mpmathify(z), NSTOP)
            values[z] = (np.array(psi, dtype=complex),
                         np.array(xi, dtype=complex))
    return values


@pytest.mark.parametrize("z", ZVALS)
def test_riccati_psi_xi_zeroth_order(reference_psi_xi, z):
    """
    Test that zeroth-order Riccati-Bessel functions agree with analytical forms
        psi_0 = sin(z)
        xi_0 = -i exp(iz)
    """
    psi, xi = reference_psi_xi[z]
    assert_allclose(psi[0], np.sin(z), rtol=1e-14)
    assert_allclose(xi[0], -1j * np.exp(1j * z), rtol=1e-14)


@pytest.mark.parametrize("z", ZVALS)
def test_riccati_psi_xi_first_order(reference_psi_xi, z):
    """
    Test that first-order Riccati-Bessel functions agree with analytical forms
        psi_1 = sin(z)/z - cos(z)
        xi_1 = -exp(iz) (1 + i/z)
    """
    psi, xi = reference_psi_xi[z]
    assert_allclose(psi[1], np.sin(z)/z - np.cos(z), rtol=1e-14)
    assert_allclose(xi[1], -np.exp(1j * z) * (1 + 1j/z), rtol=1e-14)


@pytest.mark.parametrize("z", ZVALS)
def test_riccati_psi_xi_recurrence(reference_psi_xi, z):
    """
    Tests that the spherical Bessel functions satisfy B&H eq. 4.11 and 4.12:
        j_(n-1)(z) + j_(n+1)(z) = (2n+1)/z j_n(z)
        (analogous for y)
    We test psi_n and xi_n which are z times spherical Bessel functions
    """
    psi, xi = reference_psi_xi[z]
    n = np.arange(1, NSTOP)
    for f in (psi, xi):
        f_nm1 = f[:-2]                  # f_{n-1}
        f_np1 = f[2:]                   # f_{n+1}
        rhs = (2*n + 1)/z * f[1:-1]     # (2n+1)/z f_n
        residual = rhs - f_nm1 - f_np1
        # normalize by the value at each n.  Note that this scaling will not
        # work if z puts psi(z) or xi(z) near a zero of the function.
        assert_array_less(np.abs(residual/f[1:-1]), 1e-12)
