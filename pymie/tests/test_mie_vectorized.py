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
from .. import mie_specfuncs
from numpy.testing import assert_allclose, assert_array_max_ulp, assert_equal
import pytest

class TestVectorizedSpecialFuncs():
    """Tests that simplifying/removing loops from Mie special functions
    produces same results as using loops.

    """
    num_wavelen = 10
    num_angle = 19
    wavelen = Quantity(np.linspace(400, 800, num_wavelen), 'nm')
    radius = Quantity('0.25 um')
    n_matrix = Quantity(1.00, '')
    # let index be the same at all wavelengths
    n_particle = Quantity(np.ones(num_wavelen)*1.59, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    angles = Quantity(np.linspace(0, 180., num_angle), 'deg')

    # vectorizing functions may lead to small differences from loops, due to
    # floating point precision.  We set 16 ULP as a tolerance for differences,
    # given that floating point errors tend to accumulate with multiple
    # operations.
    maxulp = 16

    def test_lentz_dn1(self):
        """Test whether Lentz continued fraction approximation for nth order
        logarithmic derivative parallelizes properly with both wavelengths and
        layers

        """
        # First test across wavelengths for single layer.
        # vectorized:
        nstop = mie._nstop(self.x.max())
        n = nstop + 1
        z = self.m[:, np.newaxis] * self.x
        lentz_vec = mie_specfuncs.lentz_dn1(z, n)
        # loop:
        lentz = np.zeros((self.num_wavelen, 1))
        for i in range(self.num_wavelen):
            z = self.m[i] * self.x[i]
            lentz[i] = mie_specfuncs.lentz_dn1(z, n)
        # these should be exactly the same because the iteration count should
        # be determined individually for each z, even in the vectorized version
        assert_equal(lentz_vec, lentz)

        # Next test across single wavelength for multiple layers
        num_layer = 5
        wavelen = Quantity(400, 'nm')
        radius = Quantity(np.linspace(0.1, 0.5, num_layer), 'um')
        # let index be different at each layer; we'll use a complex index here
        n_particle = np.linspace(1.33+0.01j, 1.59+0.03j, num_layer)[np.newaxis, :]
        n_particle = Quantity(n_particle, '')
        m = index_ratio(n_particle, self.n_matrix)
        x = size_parameter(wavelen, self.n_matrix, radius)
        # quick check on shapes
        assert m.shape == (1, num_layer)
        assert x.shape == (1, num_layer)
        # vectorized:
        nstop = mie._nstop(x.max())
        n = nstop + 1
        z = m * x
        lentz_vec = mie_specfuncs.lentz_dn1(z, n)
        assert lentz_vec.shape == (1, num_layer)
        # loop:
        lentz = np.zeros((1, num_layer), dtype=complex)
        for i in range(num_layer):
            z = m[:, i] * x[:, i]
            lentz[:, i] = mie_specfuncs.lentz_dn1(z, n)
        assert_equal(lentz_vec, lentz)

        # finally test multiple wavelengths, multiple layers
        num_wavelen = 8
        wavelen = Quantity(np.linspace(400, 800, num_wavelen), 'nm')
        radius = Quantity(np.linspace(0.1, 0.5, num_layer), 'um')
        # let index be the same at all wavelengths, but different at each
        # layer; we use a complex index here
        n_particle = np.linspace(1.33+0.1j, 1.59+0.5j, num_layer)
        n_particle = np.repeat(np.array([n_particle]), num_wavelen, axis=0)
        n_particle = Quantity(n_particle, '')
        m = index_ratio(n_particle, self.n_matrix)
        x = size_parameter(wavelen, self.n_matrix, radius)
        # quick check on shapes
        assert m.shape == (num_wavelen, num_layer)
        assert x.shape == (num_wavelen, num_layer)
        # vectorized:
        nstop = mie._nstop(x.max())
        n = nstop + 1
        z = m * x
        lentz_vec = mie_specfuncs.lentz_dn1(z, n)
        assert lentz_vec.shape == (num_wavelen, num_layer)
        # loop
        lentz = np.zeros((num_wavelen, num_layer), dtype=complex)
        for i in range(num_wavelen):
            for j in range(num_layer):
                z = m[i, j] * x[i, j]
                lentz[i, j] = mie_specfuncs.lentz_dn1(z, n).item()
        assert_equal(lentz_vec, lentz)

        # check result against Lentz (1976) equation 9, which gives the ratio
        # of Bessel functions of order nu = 9.5 at x = 1.  Converting nu to n
        # gives n = 9, and then noting that A_n = -n/z + ratio of Bessel
        # functions, we add n/z to the result:
        expected_ratio = 18.95228198
        assert_allclose(mie_specfuncs.lentz_dn1(1.0, 9) + 9, expected_ratio)


    def test_dn_1_down(self):
        """Tests that down-recurrence for logarithmic derivatives can be
        vectorized over wavelengths.

        """
        nstop = mie._nstop(self.x.max())
        nmx = nstop + 1

        z = self.m[:, np.newaxis] * self.x
        start_val = mie_specfuncs.lentz_dn1(z, nmx)
        # loop version of dn_1_down
        dn = np.zeros(start_val.shape + (nmx+1,), dtype=complex)
        dn[..., nmx] = start_val
        for i in np.arange(nmx-1, -1, -1):
            dn[..., i] = (i+1.)/z - 1.0/(dn[..., i+1] + (i+1.)/z)
        dn = dn[..., 0:nstop+1]

        dn_vec = mie_specfuncs.dn_1_down(z, nmx, nstop, start_val)
        # currently these differ at the 8.8*10-16 level (max) for some
        # elements. The following test should pass:
        np.testing.assert_array_max_ulp(dn_vec.imag, dn.imag,
                                        maxulp=self.maxulp)
        np.testing.assert_array_max_ulp(dn_vec.real, dn.real,
                                        maxulp=self.maxulp)

    def test_Qratio(self):
        """Tests that vectorized version of Qratio (without loop for up
        recursion) works the same as loop version

        """
        num_layer = 5
        radius = Quantity(np.linspace(0.85, 1.0, num_layer), 'um')
        n_matrix = Quantity(1.00, '')
        # let index be the same at all wavelengths, but different at each layer
        n_particle = np.linspace(1.33, 1.59, num_layer)
        n_particle = np.repeat(np.array([n_particle]), self.num_wavelen, axis=0)
        n_particle = Quantity(n_particle, '')
        # m should have shape [num_wavelen, num_layer]
        m = index_ratio(n_particle, n_matrix)
        # x should have shape [num_wavelen, num_layer]
        x = size_parameter(self.wavelen, n_matrix, radius)

        marray = m.astype(complex)
        xarray = x.astype(complex)

        nstop = mie._nstop(xarray.max())

        for lay in np.arange(1, num_layer):
            # m_l x_{l-1}
            z1 = marray[..., lay]*xarray[..., lay-1]
            # m_l x_l
            z2 = marray[..., lay]*xarray[..., lay]

            z1 = np.atleast_2d(z1).transpose()
            z2 = np.atleast_2d(z2).transpose()

            # calculate logarithmic derivatives D_n^1 and D_n^3
            derz1s = mie_specfuncs.log_der_13(z1, nstop)
            derz2s = mie_specfuncs.log_der_13(z2, nstop)

            # calculate ratio Q_n^l for this layer
            Qnl = mie_specfuncs.Qratio(z1, z2, nstop, dns1 = derz1s, dns2 =
                                       derz2s)

            # do same calculation with loop
            d1z1 = derz1s[0]
            d3z1 = derz1s[1]
            d1z2 = derz2s[0]
            d3z2 = derz2s[1]

            # initialize according to Yang eqn. 34
            a1 = np.real(z1)
            a2 = np.real(z2)
            b1 = np.imag(z1)
            b2 = np.imag(z2)
            qns = np.zeros(z1.shape + (nstop+1,), dtype=complex)
            qns[..., 0] = (np.exp(-2.*(b2-b1)) * (np.exp(-1j*2.*a1)
                                                  -np.exp(-2.*b1))
                           / (np.exp(-1j*2.*a2) - np.exp(-2.*b2)))
            for i in np.arange(1, nstop+1):
                qns[..., i] = qns[..., i-1]* (((d3z1[..., i] + i/z1)
                                               * (d1z2[..., i] + i/z2))
                                              / ((d3z2[..., i] + i/z2)
                                                 * (d1z1[..., i] + i/z1)))
            assert_array_max_ulp(Qnl.real, qns.real, maxulp=self.maxulp)
            # Use a different test to look at imaginary elements because the
            # differences are pretty close to zero but can vary a lot in their
            # magnitude.
            assert_allclose(np.abs(Qnl.imag - qns.imag), 0, atol=1e-10)

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
    angles = Quantity(np.linspace(0, 180., num_angle), 'deg')

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
        """Tests that mie._scatcoeffs() vectorizes properly.

        """
        nstop, coeffs = self.calc_coeffs()

        # make sure shape is correct
        expected_shape = (2, self.num_wavelen, nstop)
        assert coeffs.shape == expected_shape

        # we should get same value from loop
        coeffs_loop = np.zeros(expected_shape, dtype=complex)
        for i in range(self.m.shape[0]):
            coeffs_loop[:, i] = mie._scatcoeffs(self.m[i], self.x[i], nstop)
        assert_equal(coeffs, coeffs_loop)

    def test_vectorized_internal_coeffs(self):
        """Tests that mie._internal_coeffs() vectorizes properly

        """
        nstop = mie._nstop(self.x.max())

        # should not work for a layered sphere
        m = self.m[:, np.newaxis]
        x = self.x * np.ones((1, 5))

        with pytest.raises(ValueError, match="Internal Mie coefficients"):
            coeffs = mie._internal_coeffs(m, x, nstop)

        m = self.m[:, np.newaxis]
        x = self.x
        coeffs = mie._internal_coeffs(m, x, nstop)

        # make sure shape is correct
        expected_shape = (2, self.num_wavelen, nstop)
        assert coeffs.shape == expected_shape

        # we should get same values from loop
        coeffs_loop = np.zeros(expected_shape, dtype=complex)
        for i in range(self.m.shape[0]):
            c = mie._internal_coeffs(self.m[i], self.x[i], nstop)
            coeffs_loop[:, i] = c

        assert_equal(coeffs, coeffs_loop)

    def test_vectorized_asymmetry_parameter(self):
        """Tests that mie.calc_g() vectorizes properly. Also implicitly checks
        that mie._asymmetry_parameter() vectorizes properly

        """
        m = self.m[:, np.newaxis]
        x = self.x
        # make sure shape is [num_wavelen]
        g = mie.calc_g(m,x)
        expected_shape = (self.num_wavelen,)
        assert g.shape == expected_shape

        # we should get same values from loop. Need to set nstop to the same
        # value as used in the vectorized calculation.
        g_loop = np.zeros(expected_shape, dtype=float)
        nstop = mie._nstop(x.max())
        for i in range(self.num_wavelen):
            g_loop[i] = mie.calc_g(m[i], x[i], nstop=nstop)
        assert_equal(g, g_loop)

    def test_vectorized_cross_sections(self):
        """Tests that mie.calc_cross_sections() vectorizes properly. Also
        implicitly checks that _cross_sections() vectorizes properly

        """
        m = self.m[:, np.newaxis]
        x = self.x
        # wavelength in medium
        wavelen = self.wavelen/self.n_matrix
        cscat, cext, cback, cabs, asym = mie.calc_cross_sections(m, x, wavelen)

        # test shape
        expected_shape = (self.num_wavelen,)
        for cs in [cscat, cext, cback, cabs, asym]:
            assert cs.shape == expected_shape

        # we should get same values from loop
        cscat_loop = np.zeros(expected_shape, dtype=float)
        cext_loop = np.zeros(expected_shape, dtype=float)
        cback_loop = np.zeros(expected_shape, dtype=float)
        cabs_loop = np.zeros(expected_shape, dtype=float)
        asym_loop = np.zeros(expected_shape, dtype=float)
        for i in range(self.num_wavelen):
            cs = mie.calc_cross_sections(m[i], x[i], wavelen[i])
            cscat_loop[i], cext_loop[i], cback_loop[i], \
                cabs_loop[i], asym_loop[i] = (c.magnitude for c in cs)
        assert_equal(cscat.magnitude, cscat_loop)
        assert_equal(cext.magnitude, cext_loop)
        assert_equal(cback.magnitude, cback_loop)
        assert_equal(cabs.magnitude, cabs_loop)
        assert_equal(asym.magnitude, asym_loop)

    def test_vectorized_calc_ang_dist(self):
        """Tests that mie.calc_ang_dist() vectorizes properly. Also implicitly
        checks that _amplitude_scattering_matrix() and
        _amplitude_scattering_matrix_RG() vectorize properly.  Also checks for
        correctness of RG calculations by comparing against Mie calculations
        for small refractive index difference.

        """
        m = self.m[:, np.newaxis]
        x = self.x
        form_factor = mie.calc_ang_dist(m, x, self.angles)
        expected_shape = (self.num_wavelen, self.num_angle)
        for pol in form_factor:
            assert pol.shape == expected_shape

        # we should get same values from loop
        ipar_loop = np.zeros(expected_shape, dtype=float)
        iperp_loop = np.zeros(expected_shape, dtype=float)
        for i in range(self.num_wavelen):
            ipar, iperp = mie.calc_ang_dist(self.m[i], self.x[i], self.angles)
            ipar_loop[i] = ipar
            iperp_loop[i] = iperp
        assert_equal(form_factor[0], ipar_loop)
        assert_equal(form_factor[1], iperp_loop)

        # check vectorization for Rayleigh-Gans approximation
        form_factor_RG = mie.calc_ang_dist(m, x, self.angles, mie=False)
        expected_shape = (self.num_wavelen, self.num_angle)
        for pol in form_factor_RG:
            assert pol.shape == expected_shape

        # also check that we recover approximately the same result for RG as we
        # do for Mie in the limit of low refractive index
        radius = Quantity('0.85 um')
        n_matrix = Quantity(1.00, '')
        # let index be the same at all wavelengths.  We look at small index
        # contrast (1 + 1e-8) to be in the RG regime.  If we go smaller we run
        # into numerical issues
        n_particle = Quantity(np.ones(self.num_wavelen)*(1 + 1e-8), '')
        m = index_ratio(n_particle, n_matrix)[:, np.newaxis]
        x = size_parameter(self.wavelen, n_matrix, radius)
        num_angle = 1000
        # 0 degree scattering may give differences between RG and Mie, so we
        # compare at a few degrees and higher; also we do a lot of angles to
        # capture the sharp dips in the form factor
        angles = Quantity(np.linspace(10, 180., num_angle), 'deg')
        form_factor_RG = mie.calc_ang_dist(m, x, angles, mie=False)
        form_factor_mie = mie.calc_ang_dist(m, x, angles)

        # Since we are comparing small numbers at the dips of the form factor,
        # the Mie and RG solutions may have a relative difference of up to a
        # few percent.  But the absolute difference should be very small
        # (smaller than the default atol for this test).
        assert_allclose(form_factor_RG, form_factor_mie, rtol=1e-1)


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

        assert_equal(coeffs, coeffs_loop)
