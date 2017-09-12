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
from .. import Quantity, size_parameter, np, mie
from .. import multilayer_sphere_lib as msl
from numpy.testing import assert_array_almost_equal

def test_scatcoeffs_multi():
    # test that the scattering coefficients are the same for a non-multilayer
    # particle and for an equivalent multilayer particle
    
    # calculate coefficients for the non-multilayer
    m = 1.15
    n_sample = Quantity(1.5, '')
    wavelen = Quantity('500 nm')
    radius = Quantity('100 nm')    
    x = size_parameter(wavelen, n_sample, radius) 
    nstop = mie._nstop(x)
    coeffs = mie._scatcoeffs(m, x, nstop)
    
    # calculate coefficients for a multilayer particle with a core that 
    # is the same as the non-multilayer and a shell thickness of zero
    marray = [1.15, 1.15]  # layer index ratios, innermost first
    multi_radius = Quantity(np.array([100, 100]),'nm')   
    xarray = size_parameter(wavelen, n_sample, multi_radius)
    coeffs_multi = msl.scatcoeffs_multi(marray, xarray)

    assert_array_almost_equal(coeffs, coeffs_multi)
       
    # calculate coefficients for a 3-layer particle with a core that 
    # is the same as the non-multilayer and shell thicknesses of zero
    marray2 = [1.15, 1.15, 1.15]  # layer index ratios, innermost first
    multi_radius2 = Quantity(np.array([100, 100, 100]),'nm')   
    xarray2 = size_parameter(wavelen, n_sample, multi_radius2)
    coeffs_multi2 = msl.scatcoeffs_multi(marray2, xarray2)

    assert_array_almost_equal(coeffs, coeffs_multi2)
    
