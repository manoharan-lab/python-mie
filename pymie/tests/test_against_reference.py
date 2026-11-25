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
Tests the pymie results against the reference (arbitrary-precision) Mie
implementation.  These tests are run only if mpmath is installed.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
   (with assistance from Claude Opus 5.5)
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from .. import mie, mie_specfuncs

mpmath = pytest.importorskip("mpmath")
from . import mie_reference     # noqa: E402


@pytest.mark.parametrize("z", [0.1, 1.0, 10.0, 100.0, 1000.0])
def test_riccati_psi_xi_matches_reference(z):
    """Tests that pymie Riccati-Bessel functions return results that match
    those of the reference (mpmath) implementation.

    """
    # use pymie's stopping criterion (Allardice & Le Ru)
    nstop = mie._nstop(z)
    # set 30 digits of precision for calculations
    with mpmath.workdps(30):
        psi, xi = mie_reference.riccati_psi_xi(mpmath.mpf(z), nstop)
    psi_pymie, xi_pymie = mie_specfuncs.riccati_psi_xi(z, nstop)

    assert_allclose(psi_pymie, np.array(psi, dtype=complex), rtol=1e-12)
    assert_allclose(xi_pymie, np.array(xi, dtype=complex), rtol=1e-12)


@pytest.mark.parametrize("m, x", [(1.5, 0.1), (1.5, 10.0), (0.8, 10.0),
                                  (1.5 + 0.01j, 100.0), (1.5 + 1j, 100.0),
                                  (1.5 + 1e-3j, 1000.0)])
def test_scatcoeffs_match_reference(m, x):
    """Tests that pymie's scattering coefficients match those of the reference
    (mpmath) implementation.

    """
    nstop = mie._nstop(x)
    with mpmath.workdps(30):
        a, b = mie_reference.scatcoeffs(mpmath.mpmathify(m),
                                        mpmath.mpmathify(x), nstop)
    a_pymie, b_pymie = mie._scatcoeffs(np.array([m]), x, nstop)[:, 0]
    assert_allclose(a_pymie, np.array(a, dtype=complex),
                    rtol=1e-10, atol=1e-13)
    assert_allclose(b_pymie, np.array(b, dtype=complex),
                    rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("m, x", [(1.5, 0.1), (1.5, 10.0), (0.8, 100.0),
                                  (1.5 + 0.01j, 100.0), (1.5 + 1j, 100.0),
                                  (1.5 + 1e-3j, 1000.0)])
def test_efficiencies_match_reference(m, x):
    """Tests that pymie's scattering efficiencies (including asymmetry
    parameter) match those of the mpmath reference implementation

    """
    nstop = mie._nstop(x)
    m_pymie = np.array([m])
    x_pymie = np.array([x])
    m = mpmath.mpmathify(m)
    x = mpmath.mpmathify(x)
    with mpmath.workdps(30):
        a, b = mie_reference.scatcoeffs(m, x, nstop)
        qsca, qext, qback, g = mie_reference.efficiencies(a, b, x)
    qsca_pymie, qext_pymie, qback_pymie = mie.calc_efficiencies(m_pymie,
                                                                x_pymie)
    g_pymie = mie.calc_g(m_pymie, x_pymie)
    assert_allclose(qsca_pymie, np.array(qsca, dtype=float), rtol=1e-12)
    assert_allclose(qext_pymie, np.array(qext, dtype=float), rtol=1e-12)
    assert_allclose(qback_pymie, np.array(qback, dtype=float), rtol=1e-12)
    # looser tolerance for g because pymie calculates b_1 with a larger
    # relative error than a_1 using B&H 4.88 (perhaps because of cancellation
    # in the numerator at small x)
    assert_allclose(g_pymie, np.array(g, dtype=float), rtol=1e-10)


def test_pis_and_taus_match_reference():
    """Tests that pymie's angular functions pi_n and tau_n evaluated over a
    range of angles match the results from the mpmath reference implementation.

    """
    # test up to high n
    nstop = 200
    thetas = np.linspace(0, np.pi, 7)
    with mpmath.workdps(30):
        reference = [mie_reference.pis_and_taus(mpmath.mpmathify(theta), nstop)
                     for theta in thetas]
    pi = np.array([p for p, _ in reference], dtype=float)
    tau = np.array([t for _, t in reference], dtype=float)
    pi_pymie, tau_pymie = mie._pis_and_taus(nstop, thetas)
    # In the forward direction (mu=1), pi_n and tau_n are n(n+1)/2 (see B&H
    # page 112).  We normalize by this value so that all orders are on the same
    # scale, making it possible to test all orders using a single absolute
    # tolerance.  We disable relative tolerance checking in case theta is
    # near a zero of one of the angular functions.
    n = np.arange(1, nstop + 1)
    bound = n*(n + 1)/2
    assert_allclose(pi_pymie/bound, pi/bound, rtol=0, atol=1e-12)
    assert_allclose(tau_pymie/bound, tau/bound, rtol=0, atol=1e-12)
