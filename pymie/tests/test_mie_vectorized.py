# Copyright 2016, Vinothan N. Manoharan
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
# this package. If not, see <http://www.gnu.org/licenses/>.
"""
Tests vectorization behavior of the mie module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, index_ratio, size_parameter, np, mie
from numpy.testing import assert_almost_equal, assert_equal, assert_approx_equal
import pytest

class TestVectorized():
    num_wavelen = 10
    num_angle = 19
    wavelen = Quantity(np.linspace(400, 800, num_wavelen), 'nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00, '')
    # let index be the same at all wavelengths
    n_particle = Quantity(np.ones(num_wavelen)*1.59, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    angles = Quantity(np.linspace(0, 180., 19), 'deg')

    def test_vectorized_nstop(self):
        # Just checks that the shape of nstop is correct
        # (should scale with number of wavelengths)
        nstop = mie._nstop(self.x)
        assert nstop.shape[0] == self.num_wavelen

    def test_vectorized_scatcoeffs(self):
        # tests that scattering coefficient calculation vectorizes properly
        nstop = mie._nstop(self.x.max())
        m = self.m[:, np.newaxis]
        x = self.x[:, np.newaxis]
        coeffs = mie._scatcoeffs(m, x, nstop)
        expected_shape = (2, self.num_wavelen, nstop)
        assert coeffs.shape == expected_shape

        coeffs_loop = np.zeros(expected_shape, dtype=complex)
        for i in range(self.m.shape[0]):
            coeffs_loop[:, i] = mie._scatcoeffs(m[i].squeeze(), x[i].squeeze(),
                                                nstop)
        assert_equal(coeffs, coeffs_loop)
