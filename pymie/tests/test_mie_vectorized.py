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
from numpy.testing import assert_allclose, assert_equal
import pytest

def mx(num_wavelen, num_layer, start_wavelen=400, end_wavelen=800,
       start_radius = 100, end_radius = 1000,
       start_n_particle = 1.33+0j, end_n_particle = 1.59+0j,
       n_matrix = 1.0):
    """Convenience function to set up various combinations of wavelength and
    layer inputs to test functions.

    """
    if num_layer > 1:
        radius = Quantity(np.linspace(start_radius, end_radius, num_layer),
                          'nm')
        n_particle = np.linspace(start_n_particle, end_n_particle, num_layer)
        n_particle = n_particle[np.newaxis, :]
    else:
        radius = Quantity(start_radius, 'nm')
        n_particle = np.atleast_1d(start_n_particle)[np.newaxis, :]
    if num_wavelen > 1:
        wavelen = Quantity(np.linspace(start_wavelen, end_wavelen, num_wavelen),
                           'nm')
        # let index be the same at all wavelengths
        n_particle = Quantity(np.ones((num_wavelen, num_layer))*n_particle, '')
    else:
        wavelen = Quantity(start_wavelen, 'nm')
        n_particle = Quantity(n_particle, '')

    n_matrix = Quantity(n_matrix, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)

    return m, x


def calc_coeffs(m, x):
    """Utility function to calculate scattering coefficients"""
    nstop = mie._nstop(x.max())
    coeffs = mie._scatcoeffs(m, x, nstop)

    return nstop, coeffs


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


# all functions in this class are tested for various combinations of
# wavelengths and layers
@pytest.mark.parametrize("num_wavelen,num_layer",
                         [(1, 1), (10, 1), (1, 5), (10, 5)])
class TestVectorizedSpecialFuncs():
    """Tests that simplifying/removing loops from Mie special functions
    produces same results as using loops.  Special functions and corresponding
    tests of vectorization are as follows:

    riccati_psi_xi() :
        not tested in this class, but tested implicitly in
        `test_vectorized_scatcoeffs()`
    lentz_dn1() :
        tested by `test_lentz_dn1()`
    dn_1_down() :
        tested by `test_dn_1_down()`
    log_der_13() :
        not tested explicitly but tested implicitly in `test_Qratio()` and in
        testing of the multilayer scattering coefficients and internal
        scattering coefficients in `TestVectorizedUserFunctions`
    Qratio() :
        tested by `test_Qratio()`
    R_psi() :
        tested by `test_R_psi()`

    """
    mxargs = {"start_wavelen": 400,
              "end_wavelen": 800,
              "start_radius": 100,
              "end_radius": 1000,
              "start_n_particle": 1.59 + 0.001j,
              "end_n_particle": 1.33 + 0.005j,
              "n_matrix": Quantity(1.00, '')}

    # vectorizing functions may lead to small differences from loops, due to
    # floating point precision. We set 10^-14 as a relative tolerance for
    # differences, given that floating point errors tend to accumulate with
    # multiple operations.
    rtol = 1e-14

    def test_lentz_dn1(self, num_wavelen, num_layer):
        """Test whether Lentz continued fraction approximation for nth order
        logarithmic derivative parallelizes properly with both wavelengths and
        layers

        """
        m, x = mx(num_wavelen=num_wavelen, num_layer=num_layer, **self.mxargs)

        # quick check on shapes
        if np.isscalar(m):
            assert (num_wavelen, num_layer) == (1, 1)
        else:
            assert m.shape == (num_wavelen, num_layer)
        if np.isscalar(x):
            assert (num_wavelen, num_layer) == (1, 1)
        else:
            assert x.shape == (num_wavelen, num_layer)

        # vectorized computation
        nstop = mie._nstop(np.array(x).max())
        n = nstop + 1
        z = m * x
        lentz_vec = mie_specfuncs.lentz_dn1(z, n)

        # looped computation
        lentz = np.zeros((num_wavelen, num_layer), dtype=complex)
        for i in range(num_wavelen):
            for j in range(num_layer):
                z = np.atleast_2d(m)[i, j] * np.atleast_2d(x)[i, j]
                lentz[i, j] = mie_specfuncs.lentz_dn1(z, n).item()

        # vectorized and loop results should be exactly the same because the
        # iteration count should be determined individually for each z, even in
        # the vectorized version
        assert_equal(lentz_vec, lentz)

        # check result against Lentz (1976) equation 9, which gives the ratio
        # of Bessel functions of order nu = 9.5 at x = 1.  Converting nu to n
        # gives n = 9, and then noting that A_n = -n/z + ratio of Bessel
        # functions, we add n/z to the result:
        expected_ratio = 18.95228198
        assert_allclose(mie_specfuncs.lentz_dn1(1.0, 9) + 9, expected_ratio)

    def test_dn_1_down(self, num_wavelen, num_layer):
        """Tests that down-recurrence for logarithmic derivatives can be
        vectorized over wavelengths and layers.

        """
        m, x = mx(num_wavelen=num_wavelen, num_layer=num_layer, **self.mxargs)
        nstop = mie._nstop(np.array(x).max())
        nmx = nstop + 1

        z = m * x
        start_val = mie_specfuncs.lentz_dn1(z, nmx)

        # loop version of dn_1_down
        dn = np.zeros(start_val.shape + (nmx+1,), dtype=complex)
        dn[..., nmx] = start_val
        for i in np.arange(nmx-1, -1, -1):
            dn[..., i] = (i+1.)/z - 1.0/(dn[..., i+1] + (i+1.)/z)
        dn = dn[..., 0:nstop+1]

        # vectorized version
        dn_vec = mie_specfuncs.dn_1_down(z, nmx, nstop, start_val)

        # currently these differ at the 10^-15 level for some
        # elements. The following test should pass:
        assert_allclose(dn_vec, dn, rtol=self.rtol)

    def test_Qratio(self, num_wavelen, num_layer):
        """Tests that vectorized version of Qratio (without loop for up
        recursion) works the same as loop version.  This test runs some of the
        same code that is used to calculate the scattering coefficients for
        multilayer particles.

        """
        def Qratio_loop(z1, z2, nstop, dns1, dns2):
            # non-vectorized (loop-based) version of Qratio calculation, from
            # previous version of pymie
            d1z1 = dns1[0]
            d3z1 = dns1[1]
            d1z2 = dns2[0]
            d3z2 = dns2[1]

            # initialize according to Yang eqn. 34
            a1 = np.real(z1)
            a2 = np.real(z2)
            b1 = np.imag(z1)
            b2 = np.imag(z2)
            qns = np.zeros(z1.shape + (nstop+1,), dtype=complex)
            qns[..., 0] = (np.exp(-2.*(b2-b1)) *
                           (np.exp(-1j*2.*a1)-np.exp(-2.*b1))
                           / (np.exp(-1j*2.*a2) - np.exp(-2.*b2)))
            for i in np.arange(1, nstop+1):
                qns[..., i] = qns[..., i-1]* (((d3z1[..., i] + i/z1)
                                               * (d1z2[..., i] + i/z2))
                                              / ((d3z2[..., i] + i/z2)
                                                 * (d1z1[..., i] + i/z1)))
            return qns

        m, x = mx(num_wavelen=num_wavelen, num_layer=num_layer,
                  start_wavelen=self.mxargs["start_wavelen"],
                  end_wavelen=self.mxargs["end_wavelen"], start_radius=850,
                  end_radius=1000,
                  start_n_particle=1.33,
                  end_n_particle=1.59)

        marray = np.atleast_1d(m).astype(complex)
        xarray = np.atleast_1d(x).astype(complex)

        nstop = mie._nstop(xarray.max())
        # m_l x_{l-1}
        z1 = marray[..., 1:] * xarray[..., :-1]
        # # m_l x_l
        z2 = marray[..., 1:] * xarray[..., 1:]

        # pre-calculate logarithmic derivatives for all layers
        derz1s = mie_specfuncs.log_der_13(z1, nstop)
        derz2s = mie_specfuncs.log_der_13(z2, nstop)

        # vectorized calculation of Q_n^l for all layers
        Qnl_vec = mie_specfuncs.Qratio(z1, z2, nstop, dns1 = derz1s,
                                       dns2 = derz2s)
        # non-vectorized (loop-based) calculation of Q_n^l
        Qnl_loop = Qratio_loop(z1, z2, nstop, derz1s, derz2s)

        assert_allclose(Qnl_vec.real, Qnl_loop.real, rtol=self.rtol)
        # Use a different test to look at imaginary elements because the
        # differences are pretty close to zero but can vary a lot in their
        # magnitudes.
        assert_allclose(np.abs(Qnl_vec.imag - Qnl_loop.imag), 0, atol=1e-10)

    def test_R_psi(self, num_wavelen, num_layer):
        """Tests that the up-recurrence in the calculation of the ratio of
        Riccati-Bessel functions can be done without a loop

        """
        # The R_psi calculation shows up in the calculation of internal
        # coefficients, which is only valid (for now) for non-multilayer
        # spheres. Nonetheless, the R_psi calculation can be vectorized over
        # layers, and we test that it vectorizes properly over both wavelength
        # and layers here
        m, x = mx(num_wavelen, num_layer)
        z1 = x
        z2 = m * x
        nstop = mie._nstop(np.array(x).max())
        nmax = nstop + 1

        # note that inputs to R_psi must be 2-dimensional
        if np.isscalar(z1):
            z1 = z1 * np.ones((1,1))
            z2 = z2 * np.ones((1,1))

        # vectorized version
        output_vec = mie_specfuncs.R_psi(z1, z2, nmax)

        # loop version of R_psi (from a previous version of pymie)
        output = np.zeros(z1.shape + (nmax + 1,), dtype=complex)
        output[..., 0] = np.sin(z1) / np.sin(z2)
        dnz1 = mie_specfuncs.dn_1_down(z1, nmax + 1, nmax,
                                       mie_specfuncs.lentz_dn1(z1, nmax + 1))
        dnz2 = mie_specfuncs.dn_1_down(z2, nmax + 1, nmax,
                                       mie_specfuncs.lentz_dn1(z2, nmax + 1))
        for i in np.arange(1, nmax + 1):
            output[..., i] = output[..., i-1] * ((dnz2[..., i] + i / z2)
                                                 / (dnz1[..., i] + i / z1))

        assert_allclose(output_vec.imag, output.imag, rtol=self.rtol)
        assert_allclose(output_vec.real, output.real, rtol=self.rtol)

class TestVectorizedInternalFunctions():
    """Test vectorization of the internal Mie calculation functions (the ones
    starting with an underscore) over wavelength and layers. These tests check
    primarily that the functions return the same values for array arguments as
    they do when we loop over the arrays. They do not check for correctness
    of the results.

    Internal functions and corresponding tests of vectorization are as follows:

    _pis_and_taus() :
        not tested explicitly here, but tested implicitly in
        `test_vectorized_calc_ang_dist()`.  Vectorization over angles is tested
        in `test_mie.py::test_pis_taus()`
    _scatcoeffs() :
        tested by `test_vectorized_scatcoeffs()`
    _scatcoeffs_multi() :
        tested by `test_vectorized_scatcoeffs()`
    _internal_coeffs() :
        tested by `test_vectorized_internal_coeffs()`
    _trans_coeffs() :
        * vectorization not yet tested
    _time_coeffs() :
        * vectorization not yet tested
    _W0() :
        * vectorization not yet tested
    _nstop() :
        tested by `test_vectorized_nstop()`
    _asymmetry_parameter() :
        not tested explicitly here, but tested implicitly in
        `test_vectorized_asymmetry_parameter()`, which tests the user-facing
        function for calculating asymmetry parameters
    _cross_sections() :
        not tested explicitly here, but tested implicitly in
        `test_vectorized_cross_sections()`, which tests the user-facing
        function for calculating cross-sections
    _cross_sections_complex_medium_fu() :
        * vectorization not yet tested
    _cross_sections_complex_medium_sudiarta() :
        * vectorization not yet tested
    _scat_fields_complex_medium() :
        * vectorization not yet tested
    diff_scat_intensity_complex_medium() :
        * vectorization not yet tested
    integrate_intensity_complex_medium() :
        * vectorization not yet tested
    diff_abs_intensity_complex_medium() :
        * vectorization not yet tested
    amplitude_scattering_matrix() :
        * vectorization not yet tested
    vector_scattering_amplitude() :
        * vectorization not yet tested
    _amplitude_scattering_matrix() :
        not tested explicitly here, but tested implicitly in
        `test_vectorized_calc_ang_dist()`
    _amplitude_scattering_matrix_RG() :
        not tested explicitly here, but tested implicitly in
        `test_vectorized_calc_ang_dist()`

    """
    mxargs = {"start_wavelen": 400,
              "end_wavelen": 800,
              "start_radius": 100,
              "end_radius": 1000,
              "start_n_particle": 1.59 + 0.001j,
              "end_n_particle": 1.33 + 0.005j,
              "n_matrix": Quantity(1.00, '')}

    @pytest.mark.parametrize("num_wavelen", [1, 10, 100])
    def test_vectorized_nstop(self, num_wavelen):
        # Just checks that the shape of nstop is correct
        # (should scale with number of wavelengths)
        m, x = mx(num_wavelen=num_wavelen, num_layer=1, **self.mxargs)
        nstop = mie._nstop(x)
        assert np.atleast_1d(nstop).shape == (num_wavelen,)

    @pytest.mark.parametrize("num_wavelen,num_layer",
                             [(10, 1), (1, 5), (10, 5)])
    def test_vectorized_scatcoeffs(self, num_wavelen, num_layer):
        """Tests that mie._scatcoeffs() and mie._scatcoeffs_multi() vectorize
        properly.

        """
        m, x = mx(num_wavelen=num_wavelen, num_layer=num_layer, **self.mxargs)
        nstop, coeffs = calc_coeffs(m, x)

        # if multilayer, check that _scatcoeffs is actually calling the
        # multilayer code
        if num_layer > 1:
            coeffs_direct = mie._scatcoeffs_multi(m, x)
            assert_equal(coeffs, coeffs_direct)

        # make sure shape is correct
        if num_wavelen == 1:
            expected_shape = (2, nstop)
            assert coeffs.shape == expected_shape
            # no further test since no loop required in this case
        else:
            expected_shape = (2, num_wavelen, nstop)
            assert coeffs.shape == expected_shape

            # we should get same value from loop
            coeffs_loop = np.zeros(expected_shape, dtype=complex)
            for i in range(m.shape[0]):
                if num_layer == 1:
                    c = mie._scatcoeffs(m[i], x[i], nstop)
                else:
                    # need to specify nstop here; otherwise we will get a
                    # different number of scattering coefficients for each
                    # wavelength, since _scatcoeffs_multi() picks the largest x
                    # for each wavelength.
                    c = mie._scatcoeffs_multi(m[i], x[i], nstop)
                coeffs_loop[:, i] = c
            assert_equal(coeffs, coeffs_loop)

    @pytest.mark.parametrize("num_wavelen,num_layer",
                             [(1, 1), (10, 1), (1, 5), (10, 5)])
    def test_vectorized_internal_coeffs(self, num_wavelen, num_layer):
        """Tests that mie._internal_coeffs() vectorizes properly

        """
        m, x = mx(num_wavelen=num_wavelen, num_layer=num_layer, **self.mxargs)
        m = np.atleast_1d(m)
        x = np.atleast_1d(x)
        nstop = mie._nstop(x.max())

        # should not work for a layered sphere
        if np.atleast_2d(m).shape[-1] > 1:
            with pytest.raises(ValueError, match="Internal Mie coefficients"):
                coeffs = mie._internal_coeffs(m, x, nstop)

        else:
            coeffs = mie._internal_coeffs(m, x, nstop)

            # make sure shape is correct
            if num_wavelen == 1:
                expected_shape = (2, nstop)
                assert coeffs.shape == expected_shape
                # no further test since no loop required in this case
            else:
                expected_shape = (2, num_wavelen, nstop)
                assert coeffs.shape == expected_shape

                # we should get same values from loop
                coeffs_loop = np.zeros(expected_shape, dtype=complex)
                for i in range(m.shape[0]):
                    c = mie._internal_coeffs(m[i], x[i], nstop)
                    coeffs_loop[:, i] = c

                assert_equal(coeffs, coeffs_loop)

class TestVectorizedUserFunctions():
    """Test vectorization of the user-facing Mie calculation functions over
    wavelength for solid (one layer) spheres.  These tests check primarily that
    the functions return the same values for array arguments as they do when
    we loop over the arrays.  They do not check for correctness of the
    results.

    User functions and corresponding tests of vectorization are as follows:

    calc_ang_dist() :
        tested by test_vectorized_calc_ang_dist()
    calc_cross_sections() :
        tested by test_vectorized_cross_sections()
    calc_efficiencies() :
        tested by test_vectorized_calc_efficiencies()
    calc_g() :
        tested by test_vectorized_asymmetry_parameter
    calc_integrated_cross_section() :
        * vectorization not yet tested
    calc_energy() :
        * vectorization not yet tested
    calc_dwell_time() :
        * vectorization not yet tested
    calc_reflectance() :
        * vectorization not yet tested

    """
    mxargs = {"start_wavelen": 400,
              "end_wavelen": 800,
              "start_radius": 100,
              "end_radius": 1000,
              "start_n_particle": 1.59 + 0.001j,
              "end_n_particle": 1.33 + 0.005j,
              "n_matrix": Quantity(1.00, '')}

    num_angle = 19
    angles = Quantity(np.linspace(0, 180., num_angle), 'deg')

    @pytest.mark.parametrize("num_wavelen, num_layer",
                             [(10, 1), (1, 5), (10, 5)])
    def test_vectorized_asymmetry_parameter(self, num_wavelen, num_layer):
        """Tests that mie.calc_g() vectorizes properly. Also implicitly checks
        that mie._asymmetry_parameter() vectorizes properly

        """
        m, x = mx(num_wavelen, num_layer, **self.mxargs)
        # make sure shape is [num_wavelen]
        g = mie.calc_g(m,x)
        if num_wavelen == 1:
            expected_shape = ()
            assert g.shape == expected_shape
            # no further test needed since no loop is required in this case
        else:
            expected_shape = (num_wavelen,)
            assert g.shape == expected_shape

            # we should get same values from loop. Need to set nstop to the
            # same value as used in the vectorized calculation.
            g_loop = np.zeros(expected_shape, dtype=float)
            nstop = mie._nstop(x.max())
            for i in range(num_wavelen):
                g_loop[i] = mie.calc_g(m[i], x[i], nstop=nstop)
            assert_equal(g, g_loop)

    @pytest.mark.parametrize("num_wavelen, num_layer",
                             [(10, 1), (1, 5), (10, 5)])
    def test_vectorized_cross_sections(self, num_wavelen, num_layer):
        """Tests that mie.calc_cross_sections() vectorizes properly. Also
        implicitly checks that _cross_sections() vectorizes properly

        """
        m, x = mx(num_wavelen, num_layer, **self.mxargs)
        # wavelength in medium
        wavelen = Quantity(np.linspace(self.mxargs["start_wavelen"],
                                       self.mxargs["end_wavelen"],
                                       num_wavelen),
                           "nm")
        wavelen_med = wavelen/self.mxargs["n_matrix"]
        cscat, cext, cback, cabs, asym = mie.calc_cross_sections(m, x,
                                                                 wavelen_med)

        # test shape
        expected_shape = (num_wavelen,)
        for cs in [cscat, cext, cback, cabs, asym]:
            assert cs.shape == expected_shape

        # we should get same values from loop
        cscat_loop = np.zeros(expected_shape, dtype=float)
        cext_loop = np.zeros(expected_shape, dtype=float)
        cback_loop = np.zeros(expected_shape, dtype=float)
        cabs_loop = np.zeros(expected_shape, dtype=float)
        asym_loop = np.zeros(expected_shape, dtype=float)
        for i in range(num_wavelen):
            cs = mie.calc_cross_sections(m[i], x[i], wavelen[i])
            cscat_loop[i], cext_loop[i], cback_loop[i], \
                cabs_loop[i], asym_loop[i] = (c.magnitude for c in cs)
        assert_equal(cscat.magnitude, cscat_loop)
        assert_equal(cext.magnitude, cext_loop)
        assert_equal(cback.magnitude, cback_loop)
        assert_equal(cabs.magnitude, cabs_loop)
        assert_equal(asym.magnitude, asym_loop)

    @pytest.mark.parametrize("num_wavelen, num_layer",
                             [(10, 1), (1, 5), (10, 5)])
    def test_vectorized_calc_efficiencies(self, num_wavelen, num_layer):
        """Tests that mie.calc_efficiencies() vectorizes properly with
        wavelength, including with multi-layered particles.

        """
        m, x = mx(num_wavelen, num_layer, **self.mxargs)
        qscat, qext, qback = mie.calc_efficiencies(m, x)

        # test shape
        if num_wavelen == 1:
            expected_shape = ()
            # no further test because no loop is needed in this case
        else:
            expected_shape = (num_wavelen,)
        for q in [qscat, qext, qback]:
            assert q.shape == expected_shape

        if num_wavelen > 1:
            # we should get same values from loop
            qscat_loop = np.zeros(expected_shape, dtype=float)
            qext_loop = np.zeros(expected_shape, dtype=float)
            qback_loop = np.zeros(expected_shape, dtype=float)
            for i in range(num_wavelen):
                qs = mie.calc_efficiencies(m[i], x[i])
                qscat_loop[i], qext_loop[i], qback_loop[i] = (q for q in qs)
            assert_equal(qscat, qscat_loop)
            assert_equal(qext, qext_loop)
            assert_equal(qback, qback_loop)

    @pytest.mark.parametrize("num_wavelen, num_layer",
                             [(10, 1), (1, 5), (10, 5)])
    def test_vectorized_calc_ang_dist(self, num_wavelen, num_layer):
        """Tests that mie.calc_ang_dist() vectorizes properly. Also implicitly
        checks that _amplitude_scattering_matrix() and
        _amplitude_scattering_matrix_RG() vectorize properly.  Also checks for
        correctness of RG calculations by comparing against Mie calculations
        for small refractive index difference.

        """
        m, x = mx(num_wavelen, num_layer, **self.mxargs)
        form_factor = mie.calc_ang_dist(m, x, self.angles)
        if num_wavelen == 1:
            expected_shape = (self.num_angle,)
            for pol in form_factor:
                assert pol.shape == expected_shape
            # no further test required since there is only one wavelength
            return
        else:
            expected_shape = (num_wavelen, self.num_angle)
        for pol in form_factor:
            assert pol.shape == expected_shape

        # we should get same values from loop
        ipar_loop = np.zeros(expected_shape, dtype=float)
        iperp_loop = np.zeros(expected_shape, dtype=float)
        for i in range(num_wavelen):
            ipar, iperp = mie.calc_ang_dist(m[i], x[i], self.angles)
            ipar_loop[i] = ipar
            iperp_loop[i] = iperp
        assert_equal(form_factor[0], ipar_loop)
        assert_equal(form_factor[1], iperp_loop)

        # check vectorization for Rayleigh-Gans approximation
        if num_layer > 1:
            with pytest.raises(ValueError,
                               match="Rayleigh-Gans approximation cannot"):
                form_factor_RG = mie.calc_ang_dist(m, x, self.angles,
                                                   mie=False)
            return

        form_factor_RG = mie.calc_ang_dist(m, x, self.angles,
                                               mie=False)

        expected_shape = (num_wavelen, self.num_angle)
        for pol in form_factor_RG:
            assert pol.shape == expected_shape

        # also check that we recover approximately the same result for RG as we
        # do for Mie in the limit of low refractive index
        radius = Quantity('0.85 um')
        n_matrix = Quantity(1.00, '')
        # let index be the same at all wavelengths. We look at small index
        # contrast (1 + 1e-8) to be in the RG regime. If we go smaller we run
        # into numerical issues
        n_particle = Quantity(np.ones(num_wavelen)*(1 + 1e-8), '')
        wavelen = Quantity(np.linspace(self.mxargs["start_wavelen"],
                                       self.mxargs["end_wavelen"],
                                       num_wavelen),
                           'nm')
        m = index_ratio(n_particle, n_matrix)[:, np.newaxis]
        x = size_parameter(wavelen, n_matrix, radius)
        num_angle = 1000
        # 0 degree scattering may give differences between RG and Mie, so we
        # compare at a few degrees and higher; also we do a lot of angles to
        # capture the sharp dips in the form factor
        angles = Quantity(np.linspace(10, 180., num_angle), 'deg')
        form_factor_RG = mie.calc_ang_dist(m, x, angles, mie=False)
        form_factor_mie = mie.calc_ang_dist(m, x, angles)

        # Since we are comparing small numbers at the dips of the form factor,
        # the Mie and RG solutions may have a relative difference of up to a
        # few percent. But the absolute difference should be very small
        # (smaller than the default atol for this test).
        assert_allclose(form_factor_RG, form_factor_mie, rtol=1e-1)
