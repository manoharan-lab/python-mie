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
Tests for the multilayer_sphere_lib module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Victoria Hwang <vhwang@g.harvard.edu>
"""
import os
from .. import Quantity, size_parameter, np, mie
from .. import multilayer_sphere_lib as msl
from numpy.testing import assert_array_almost_equal, assert_allclose
import yaml

def test_scatcoeffs_multi():
    # test that the scattering coefficients are the same for a non-multilayer
    # particle and for an equivalent multilayer particle

    # calculate coefficients for the non-multilayer
    m = 1.15
    n_sample = Quantity(1.5, '')
    wavelen = Quantity('500.0 nm')
    radius = Quantity('100.0 nm')
    x = size_parameter(wavelen, n_sample, radius)
    nstop = mie._nstop(x)
    coeffs = mie._scatcoeffs(m, x, nstop)

    # calculate coefficients for a multilayer particle with a core that
    # is the same as the non-multilayer and a shell thickness of zero
    marray = [1.15, 1.15]  # layer index ratios, innermost first
    multi_radius = Quantity(np.array([100.0, 100.0]),'nm')
    xarray = size_parameter(wavelen, n_sample, multi_radius)
    coeffs_multi = msl.scatcoeffs_multi(marray, xarray)

    assert_array_almost_equal(coeffs, coeffs_multi)

    # calculate coefficients for a 3-layer particle with a core that
    # is the same as the non-multilayer and shell thicknesses of zero
    marray2 = [1.15, 1.15, 1.15]  # layer index ratios, innermost first
    multi_radius2 = Quantity(np.array([100.0, 100.0, 100.0]),'nm')
    xarray2 = size_parameter(wavelen, n_sample, multi_radius2)
    coeffs_multi2 = msl.scatcoeffs_multi(marray2, xarray2)

    assert_array_almost_equal(coeffs, coeffs_multi2)

def test_scatcoeffs_multi_absorbing_particle():
    # test that the scattering coefficients are the same for a real index ratio
    # and a complex index ratio with a 0 imaginary component.
    marray_real = [1.15, 1.2]
    marray_imag = [1.15 + 0j, 1.2 + 0j]
    n_sample = Quantity(1.5, '')
    wavelen = Quantity('500.0 nm')
    multi_radius = Quantity(np.array([100.0, 110.0]),'nm')
    xarray = size_parameter(wavelen, n_sample, multi_radius)

    coeffs_multi_real = msl.scatcoeffs_multi(marray_real, xarray)
    coeffs_multi_imag = msl.scatcoeffs_multi(marray_imag, xarray)

    assert_array_almost_equal(coeffs_multi_real, coeffs_multi_imag)

def test_sooty_particles():
    '''
    Test multilayered sphere scattering coefficients by comparison of
    radiometric quantities.

    We will use the data in [Yang2003]_ Table 3 on  p. 1717, cases
    2, 3, and 4 as our gold standard.
    '''
    x_L = 100
    m_med = 1.33
    m_abs = 2. + 1.j
    f_v = 0.1

    def efficiencies_from_scat_units(m, x):
        asbs = msl.scatcoeffs_multi(m, x)
        qs = np.array(mie._cross_sections(*asbs)) * 2 / x_L**2
        # there is a factor of 2 conventional difference between
        # "backscattering" and "radar backscattering" efficiencies.
        return np.array([qs[1], qs[0], qs[2]*4*np.pi/2.])

    # first case: absorbing core
    x_ac = np.array([f_v**(1./3.) * x_L, x_L])
    m_ac = np.array([m_abs, m_med])

    # second case: absorbing shell
    x_as = np.array([(1. - f_v)**(1./3.), 1.]) * x_L
    m_as = np.array([m_med, m_abs])

    # third case: smooth distribution (900 layers)
    n_layers = 900
    x_sm = np.arange(1, n_layers + 1) * x_L / n_layers
    beta = (m_abs**2 - m_med**2) / (m_abs**2 + 2. * m_med**2)
    f = 4./3. * (x_sm / x_L) * f_v
    m_sm = m_med * np.sqrt(1. + 3. * f * beta / (1. - f * beta))

    location = os.path.split(os.path.abspath(__file__))[0]
    gold_name = os.path.join(location, 'gold',
                             'gold_multilayer')
    gold_file = open(gold_name + '.yaml')
    gold = np.array(yaml.safe_load(gold_file))
    gold_file.close()

    assert_allclose(efficiencies_from_scat_units(m_ac, x_ac), gold[0],
                    rtol = 1e-3)
    assert_allclose(efficiencies_from_scat_units(m_as, x_as), gold[1],
                    rtol = 2e-5)
    assert_allclose(efficiencies_from_scat_units(m_sm, x_sm), gold[2],
                    rtol = 1e-3)
