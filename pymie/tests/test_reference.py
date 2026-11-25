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

# values of z to test, with complex values corresponding to absorbing
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
    """Test that zeroth-order Riccati-Bessel functions agree with analytical
    forms
        psi_0 = sin(z)
        xi_0 = -i exp(iz)
    """
    psi, xi = reference_psi_xi[z]
    assert_allclose(psi[0], np.sin(z), rtol=1e-14)
    assert_allclose(xi[0], -1j * np.exp(1j * z), rtol=1e-14)


@pytest.mark.parametrize("z", ZVALS)
def test_riccati_psi_xi_first_order(reference_psi_xi, z):
    """Test that first-order Riccati-Bessel functions agree with analytical
    forms
        psi_1 = sin(z)/z - cos(z)
        xi_1 = -exp(iz) (1 + i/z)
    """
    psi, xi = reference_psi_xi[z]
    assert_allclose(psi[1], np.sin(z)/z - np.cos(z), rtol=1e-14)
    assert_allclose(xi[1], -np.exp(1j * z) * (1 + 1j/z), rtol=1e-14)


@pytest.mark.parametrize("z", ZVALS)
def test_riccati_psi_xi_recurrence(reference_psi_xi, z):
    """Tests that the spherical Bessel functions satisfy B&H eq. 4.11 and 4.12:
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


def test_scatcoeffs_index_matched():
    """Test that scattering coefficients go to zero when the particle is
    index-matched to the medium

    """
    with mpmath.workdps(30):
        # index ratio is 1 for index matching
        m = mpmath.mpf(1)
        # choose reasonably large size parameter
        x = mpmath.mpf(10)
        a, b = mie_reference.scatcoeffs(m, x, NSTOP)
    assert_allclose(np.array(a, dtype=complex), 0, atol=1e-14)
    assert_allclose(np.array(b, dtype=complex), 0, atol=1e-14)


@pytest.mark.parametrize("m, x", [(1.5, 0.1), (1.5, 10.0), (0.8, 10.0)])
def test_scatcoeffs_no_absorption(m, x):
    """Test that scattering coefficients obey the relations
        Re(a_n) = |a_n|^2
        Re(b_n) = |b_n|^2
    which follow from the equality of C_sca and C_ext when there is no
    absorption (see B&H eqs. 4.61 and 4.62)

    """
    m = mpmath.mpmathify(m)
    x = mpmath.mpmathify(x)
    with mpmath.workdps(30):
        a, b = mie_reference.scatcoeffs(m, x, NSTOP)
    a = np.array(a, dtype=complex)
    b = np.array(b, dtype=complex)
    assert_allclose(np.real(a), np.abs(a)**2, rtol=1e-14)
    assert_allclose(np.real(b), np.abs(b)**2, rtol=1e-14)


@pytest.mark.parametrize("m", [1.5, 0.8, 1.5 + 0.1j])
def test_scatcoeffs_small_x(m):
    """Test that scattering coefficient a_1 reduces to Rayleigh result in limit
    of small x. From B&H eq. 5.4,
        a_1 = -i2x^3/3 (m^2 - 1)/(m^2 + 2)

    """
    m = mpmath.mpmathify(m)
    x = mpmath.mpmathify(1e-7)
    with mpmath.workdps(30):
        a, _ = mie_reference.scatcoeffs(m, x, 1)
        a1_expected = -1j * 2 * x**3 / 3 * (m**2 - 1)/(m**2 + 2)
    assert_allclose(complex(a[0]), complex(a1_expected), rtol=1e-12)


@pytest.mark.parametrize("m, x", [(1.5, 0.1), (1.5, 10.0), (0.8, 10.0)])
def test_efficiencies_no_absorption(m, x):
    """Test that in the absence of absorption, Q_ext = Q_sca"""
    with mpmath.workdps(30):
        m = mpmath.mpmathify(m)
        x = mpmath.mpmathify(x)
        a, b = mie_reference.scatcoeffs(m, x, NSTOP)
        qsca, qext, _, _ = mie_reference.efficiencies(a, b, x)
    assert_allclose(float(qsca), float(qext), rtol=1e-14)


@pytest.mark.parametrize("m", [1.5, 0.8, 1.5 + 0.1j])
def test_efficiencies_small_x(m):
    """Test that efficiencies match results in Rayleigh limit for small x.

    From B&H eq 5.8,
        Q_sca = 8/3 x^4 |(m^2 - 1) / (m^2 + 2)|^2
    From B&H eq 5.11
        Q_abs = 4 x Im{(m^2 - 1) / (m^2 + 2)}
    From B&H eq 5.9 (incorporating the 1/(4 pi) factor from pymie convention)
        Q_back = x^4/pi |(m^2 - 1) / (m^2 + 2)|^2
    and for a Rayleigh particle, scattering should be symmetric between front
    and back hemispheres:
        g = 0
    """
    with mpmath.workdps(30):
        m = mpmath.mpmathify(m)
        x = mpmath.mpmathify(1e-7)
        a, b = mie_reference.scatcoeffs(m, x, NSTOP)
        qsca, qext, qback, g = mie_reference.efficiencies(a, b, x)
        cm_factor = (m**2 - 1)/(m**2 + 2)
        qsca_expected = mpmath.mpf(8)/3 * x**4 * abs(cm_factor)**2
        qabs_expected = 4 * x * cm_factor.imag
        qext_expected = qsca_expected + qabs_expected
        qback_expected = x**4 / mpmath.pi * abs(cm_factor)**2
    assert_allclose(float(qsca), float(qsca_expected), rtol=1e-12)
    assert_allclose(float(qext), float(qext_expected), rtol=1e-12)
    assert_allclose(float(qback), float(qback_expected), rtol=1e-12)
    assert_allclose(float(g), 0, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.0, 2.5, np.pi])
def test_pis_and_taus_low_order(theta):
    """Test that the angular functions pi_n and tau_n agree with the analytical
    forms at low order (see B&H 4.47 and initial conditions for the recurrence
    relations on p. 95):

        pi_1 = 1
        pi_2 = 3 mu
        tau_1 = mu pi_1 = mu
        tau_2 = 2 mu pi_2 - 3 pi_1 = 6 mu^2 - 3

    where mu = cos(theta)
    """
    with mpmath.workdps(30):
        pi, tau = mie_reference.pis_and_taus(mpmath.mpmathify(theta), 2)
    mu = np.cos(theta)
    assert_allclose(np.array(pi, dtype=float), [1, 3*mu], rtol=1e-14)
    assert_allclose(np.array(tau, dtype=float), [mu, 6*mu**2 - 3], rtol=1e-14)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.0, 2.5, np.pi])
def test_pis_and_taus_recurrence(theta):
    """Test that the angular functions pi_n and tau_n satisfy the recurrence
    relations in B&H 4.47:

        pi_n = (2n-1)/(n-1) mu pi_{n-1} - n/(n-1) pi_{n-2}
        tau_n = n mu pi_n - (n+1) pi_{n-1}

    where mu = cos(theta).
    """
    with mpmath.workdps(30):
        mu = mpmath.cos(theta)
        pi, tau = mie_reference.pis_and_taus(mpmath.mpmathify(theta), NSTOP)
        # insert the zeroth coefficient (pi_0 = 0) at the front of the list
        pi.insert(0, mpmath.mpf(0))

        # calculate the values by recurrence
        pi_recurrence = [0, 1]
        for n in range(2, NSTOP+1):
            pi_recurrence.append(mpmath.mpf(2*n-1)/(n-1) * mu * pi[n-1]
                                 - mpmath.mpf(n)/(n-1) * pi[n-2])
        tau_recurrence = []
        for n in range(1, NSTOP+1):
            tau_recurrence.append(n * mu * pi[n]
                                  - (n+1) * pi[n-1])

        # we use relative tolerance here because the angles chosen aren't close
        # to zeros of either angular function. Would need to change the tests
        # to check angles that are close to zeros.
        assert_allclose(np.array(pi, dtype=float),
                        np.array(pi_recurrence, dtype=float),
                        rtol=1e-14)
        assert_allclose(np.array(tau, dtype=float),
                        np.array(tau_recurrence, dtype=float),
                        rtol=1e-14)
