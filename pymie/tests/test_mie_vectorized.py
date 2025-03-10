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
from numpy.testing import assert_allclose, assert_equal
#import pytest

class TestVectorized():
    """Test vectorization of the Mie calculations over wavelength for solid
    (one layer) spheres.

    """
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

    def calc_coeffs(self):
        nstop = mie._nstop(self.x.max())
        m = self.m[:, np.newaxis]
        x = self.x
        coeffs = mie._scatcoeffs(m, x, nstop)

        return nstop, coeffs

    def test_vectorized_nstop(self):
        # Just checks that the shape of nstop is correct
        # (should scale with number of wavelengths)
        nstop = mie._nstop(self.x)
        assert nstop.shape[0] == self.num_wavelen

    def test_vectorized_scatcoeffs(self):
        # tests that mie._scatcoeffs() vectorizes properly
        nstop, coeffs = self.calc_coeffs()

        # make sure shape is correct
        expected_shape = (2, self.num_wavelen, nstop)
        assert coeffs.shape == expected_shape

        # we should get same value from loop
        coeffs_loop = np.zeros(expected_shape, dtype=complex)
        for i in range(self.m.shape[0]):
            coeffs_loop[:, i] = mie._scatcoeffs(self.m[i], self.x[i], nstop)
        assert_equal(coeffs, coeffs_loop)

    def test_vectorized_asymmetry_parameter(self):
        # tests that mie._asymmetry_parameter() vectorizes properly
        nstop, coeffs = self.calc_coeffs()

        # make sure shape is [num_wavelen]
        g = mie._asymmetry_parameter(coeffs[0], coeffs[1])
        expected_shape = (self.num_wavelen,)
        assert g.shape == expected_shape

        # we should get same values from loop
        g_loop = np.zeros(expected_shape, dtype=float)
        for i in range(self.num_wavelen):
            albl = mie._scatcoeffs(self.m[i], self.x[i], nstop)
            g_loop[i] = mie._asymmetry_parameter(albl[0], albl[1])
        assert_equal(g, g_loop)

    def test_vectorized_cross_sections(self):
        # tests that mie._cross_sections() vectorizes properly
        nstop, coeffs = self.calc_coeffs()

        cscat, cext, cback = mie._cross_sections(coeffs[0], coeffs[1])

        # test shape
        expected_shape = (self.num_wavelen,)
        for cs in [cscat, cext, cback]:
            assert cs.shape == expected_shape

        # we should get same values from loop
        cscat_loop = np.zeros(expected_shape, dtype=float)
        cext_loop = np.zeros(expected_shape, dtype=float)
        cback_loop = np.zeros(expected_shape, dtype=float)
        for i in range(self.num_wavelen):
            albl = mie._scatcoeffs(self.m[i], self.x[i], nstop)
            cs = mie._cross_sections(albl[0], albl[1])
            cscat_loop[i], cext_loop[i], cback_loop[i] = cs
        assert_equal(cscat, cscat_loop)
        assert_equal(cext, cext_loop)
        assert_equal(cback, cback_loop)

def test_parameter_shapes():
    """Test to make sure vectorized size_parameter() and index_ratio() have the
    right shapes"""

    num_wavelen = 8
    num_layer = 3
    wavelen = Quantity(np.linspace(400, 800, num_wavelen), 'nm')
    radius = Quantity(np.linspace(0.85, 1.0, num_layer), 'um')
    n_matrix = Quantity(1.00, '')
    # let index be the same at all wavelengths, but different at each layer
    n_particle = np.linspace(1.33, 1.59, num_layer)
    n_particle = np.repeat(np.array([n_particle]), num_wavelen, axis=0)
    n_particle = Quantity(n_particle, '')

    # multiple wavelengths, multiple layers. m and x should have shape
    # [num_wavelen, num_layer].
    expected_shape = (num_wavelen, num_layer)
    # The following should be true by construction of n_particle, but we test
    # anyway to make sure that index_ratio() doesn't change shape
    m = index_ratio(n_particle, n_matrix)
    assert m.shape == expected_shape
    # x should have shape [num_wavelen, num_layer]
    x = size_parameter(wavelen, n_matrix, radius)
    assert x.shape == expected_shape

    # one wavelength, multiple layers; index specified as 1D array.  Should
    # return a 1D index ratio and a 2D size parameter
    wavelen = Quantity(400, 'nm')
    num_layer = 6
    radius = Quantity(np.linspace(0.85, 1.0, num_layer), 'um')
    n_particle = Quantity(np.linspace(1.33, 1.59, num_layer), '')
    m = index_ratio(n_particle, n_matrix)
    assert m.shape == (num_layer, )
    x = size_parameter(wavelen, n_matrix, radius)
    assert x.shape == (1, num_layer)

    # one wavelength, multiple layers; index specified as 2D array with shape
    # [1, num_layers]. Should return a 2D index ratio and a 2D size parameter
    wavelen = Quantity(400, 'nm')
    num_layer = 6
    radius = Quantity(np.linspace(0.85, 1.0, num_layer), 'um')
    n_particle = Quantity(np.linspace(1.33, 1.59, num_layer)[np.newaxis,:], '')
    m = index_ratio(n_particle, n_matrix)
    assert m.shape == (1, num_layer)
    x = size_parameter(wavelen, n_matrix, radius)
    assert x.shape == (1, num_layer)

    # multiple wavelengths, one layer; index specified as a 2D array with shape
    # [num_wavelen, 1].  Should return a 2D index ratio and 2D size parameter
    num_wavelen = 8
    wavelen = Quantity(np.linspace(400, 800, num_wavelen), 'nm')
    radius = Quantity(0.85, 'um')
    n_particle = Quantity(np.linspace(1.33, 1.59, num_wavelen)[:,np.newaxis],
                          '')
    m = index_ratio(n_particle, n_matrix)
    assert m.shape == (num_wavelen, 1)
    x = size_parameter(wavelen, n_matrix, radius)
    assert x.shape == (num_wavelen, 1)


class TestVectorizedMultilayer():
    """Test vectorization of the Mie calculations over wavelength for
    multilayer spheres.

    """
    num_wavelen = 10
    num_angle = 19
    num_layer = 5
    wavelen = Quantity(np.linspace(400, 800, num_wavelen), 'nm')
    radius = Quantity(np.linspace(0.85, 1.0, num_layer), 'um')
    n_matrix = Quantity(1.00, '')
    # let index be the same at all wavelengths, but different at each layer
    n_particle = np.linspace(1.33, 1.59, num_layer)
    n_particle = np.repeat(np.array([n_particle]), num_wavelen, axis=0)
    n_particle = Quantity(n_particle, '')
    # m should have shape [num_wavelen, num_layer]
    m = index_ratio(n_particle, n_matrix)
    # x should have shape [num_wavelen, num_layer]
    x = size_parameter(wavelen, n_matrix, radius)
    angles = Quantity(np.linspace(0, 180., num_angle), 'deg')

    def calc_coeffs(self):
        nstop = mie._nstop(self.x.max())
        m = self.m
        x = self.x
        coeffs = mie._scatcoeffs(m, x, nstop)

        return nstop, coeffs

    def test_vectorized_parameters(self):
        expected_shape = (self.num_wavelen, self.num_layer)
        assert self.x.shape == expected_shape
        assert self.m.shape == expected_shape

    def test_vectorized_scatcoeffs_multi(self):
        """Tests that mie._scatcoeffs_multi() vectorizes properly

        """
        # first check that _scatcoeffs is actually calling the multilayer code
        nstop, coeffs = self.calc_coeffs()
        coeffs_direct = mie._scatcoeffs_multi(self.m, self.x)
        assert_equal(coeffs, coeffs_direct)

        # make sure shape is correct
        expected_shape = (2, self.num_wavelen, nstop)
        assert coeffs.shape == expected_shape

        # we should get same values from loop
        coeffs_loop = np.zeros(expected_shape, dtype=complex)
        for i in range(self.m.shape[0]):
            # need to specify nstop here; otherwise we will get a different
            # number of scattering coefficients for each wavelength, since
            # _scatcoeffs_multi() picks the largest x for each wavelength.
            c = mie._scatcoeffs_multi(self.m[i], self.x[i], nstop)
            coeffs_loop[:, i] = c

        # the vectorized version differs from the loop version in a few
        # elements by more than floating point uncertainty, perhaps due to
        # tolerances in the continued fraction algorithm.  So we
        # test for allclose instead of equal
        assert_allclose(coeffs, coeffs_loop, rtol=1e-14)
