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
