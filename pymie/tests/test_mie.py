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
Tests for the mie module

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

from .. import Quantity, ureg, q, index_ratio, size_parameter, np, mie
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pint.errors import DimensionalityError
import pytest

def test_cross_sections():
    # Test cross sections against values calculated from BHMIE code (originally
    # calculated for testing fortran-based Mie code in holopy)

    # test case is PS sphere in water
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.33, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    qscat, qext, qback = mie.calc_efficiencies(m, x)
    g = mie.calc_g(m,x)   # asymmetry parameter

    qscat_std, qext_std, g_std = 3.6647, 3.6677, 0.92701
    assert_almost_equal(qscat, qscat_std, decimal=4)
    assert_almost_equal(qext, qext_std, decimal=4)
    assert_almost_equal(g, g_std, decimal=4)

    # test to make sure calc_cross_sections returns the same values as
    # calc_efficiencies and calc_g
    cscat = qscat * np.pi * radius**2
    cext = qext * np.pi * radius**2
    cback  = qback * np.pi * radius**2
    cscat2, cext2, _, cback2, g2 = mie.calc_cross_sections(m, x, wavelen/n_matrix)
    assert_almost_equal(cscat.to('m^2').magnitude, cscat2.to('m^2').magnitude)
    assert_almost_equal(cext.to('m^2').magnitude, cext2.to('m^2').magnitude)
    assert_almost_equal(cback.to('m^2').magnitude, cback2.to('m^2').magnitude)
    assert_almost_equal(g, g2)

    # test that calc_cross_sections throws an exception when given an argument
    # with the wrong dimensions
    assert_raises(DimensionalityError, mie.calc_cross_sections,
                  m, x, Quantity('0.25 J'))
    assert_raises(DimensionalityError, mie.calc_cross_sections,
                  m, x, Quantity('0.25'))

def test_form_factor():
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)

    angles = Quantity(np.linspace(0, 180., 19), 'deg')
    # these values are calculated from MiePlot
    # (http://www.philiplaven.com/mieplot.htm), which uses BHMIE
    iperp_bhmie = np.array([2046.60203864487, 1282.28646423634, 299.631502275208,
                            7.35748912156671, 47.4215270799552, 51.2437259188946,
                            1.48683515673452, 32.7216414263307, 1.4640166361956,
                            10.1634538431238, 4.13729254895905, 0.287316587318158,
                            5.1922111829055, 5.26386476102605, 1.72503962851391,
                            7.26013963969779, 0.918926070270738, 31.5250813730405,
                            93.5508557840006])
    ipar_bhmie = np.array([2046.60203864487, 1100.18673543798, 183.162880455348,
                           13.5427093640281, 57.6244243689505, 35.4490544770251,
                           41.0597781235887, 14.8954859951121, 34.7035437764261,
                           5.94544441735711, 22.1248452485893, 3.75590232882822,
                           10.6385606309297, 0.881297551245856, 16.2259629218812,
                           7.24176462105438, 76.2910238480798, 54.1983836607738,
                           93.5508557840006])

    ipar, iperp = mie.calc_ang_dist(m, x, angles)
    assert_array_almost_equal(ipar, ipar_bhmie)
    assert_array_almost_equal(iperp, iperp_bhmie)
    
def test_efficiencies():
    x = np.array([0.01, 0.01778279, 0.03162278, 0.05623413, 0.1, 0.17782794,
                  0.31622777, 0.56234133, 1, 1.77827941, 3.16227766, 5.62341325,
                  10, 17.7827941, 31.6227766, 56.23413252, 100, 177.827941,
                  316.22776602, 562.34132519, 1000])
    # these values are calculated from MiePlot
    # (http://www.philiplaven.com/mieplot.htm), which uses BHMIE
    qext_bhmie = np.array([1.86E-06, 3.34E-06, 6.19E-06, 1.35E-05, 4.91E-05,
                           3.39E-04, 3.14E-03, 3.15E-02, 0.2972833954,
                           1.9411047797, 4.0883764682, 2.4192037463, 2.5962875796,
                           2.097410246, 2.1947770304, 2.1470056626, 2.1527225028,
                           2.0380806126, 2.0334715395, 2.0308028599, 2.0248011731])
    qsca_bhmie = np.array([3.04E-09, 3.04E-08, 3.04E-07, 3.04E-06, 3.04E-05,
                           3.05E-04, 3.08E-03, 3.13E-02, 0.2969918262,
                           1.9401873562, 4.0865768252, 2.4153820014,
                           2.5912825599, 2.0891233123, 2.1818510296,
                           2.1221614258, 2.1131226379, 1.9736114111,
                           1.922984002, 1.8490112847, 1.7303694187])
    qback_bhmie = np.array([3.62498741762823E-10, 3.62471372652178E-09,
                            3.623847844672E-08, 3.62110791613906E-07,
                            3.61242786911475E-06, 3.58482008581018E-05,
                            3.49577114878315E-04, 3.19256234186963E-03,
                            0.019955229811329, 1.22543944129328E-02,
                            0.114985907473273, 0.587724020116958,
                            0.780839362788633, 0.17952369257935,
                            0.068204471161473, 0.314128510891842,
                            0.256455963161882, 3.84713481428992E-02,
                            1.02022022710453, 0.51835427781473,
                            0.331000402174976])

    wavelen = Quantity('658 nm')
    n_matrix = Quantity(1.00, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)

    effs = [mie.calc_efficiencies(m, x) for x in x]
    q_arr = np.asarray(effs)
    qsca = q_arr[:,0]
    qext = q_arr[:,1]
    qback = q_arr[:,2]
    # use two decimal places for the small size parameters because MiePlot
    # doesn't report sufficient precision
    assert_array_almost_equal(qsca[0:9], qsca_bhmie[0:9], decimal=2)
    assert_array_almost_equal(qext[0:9], qext_bhmie[0:9], decimal=2)
    # there is some disagreement at 4 decimal places in the cross
    # sections at large x.  Not sure if this points to a bug in the algorithm
    # or improved precision over the bhmie results.  Should be investigated
    # more.
    assert_array_almost_equal(qsca[9:], qsca_bhmie[9:], decimal=3)
    assert_array_almost_equal(qext[9:], qext_bhmie[9:], decimal=3)

    # test backscattering efficiencies (still some discrepancies at 3rd decimal
    # point for large size parameters)
    assert_array_almost_equal(qback, qback_bhmie, decimal=2)

def test_absorbing_materials():
    # test calculations for gold, which has a high imaginary refractive index
    wavelen = Quantity('658 nm')
    n_matrix = Quantity(1.00, '')
    n_particle = Quantity(0.1425812 + 3.6813284 * 1.0j, '')
    m = index_ratio(n_particle, n_matrix)
    x = 10.0

    angles = Quantity(np.linspace(0, 90., 10), 'deg')
    # these values are calculated from MiePlot
    # (http://www.philiplaven.com/mieplot.htm), which uses BHMIE
    iperp_bhmie = np.array([4830.51401095968, 2002.39671236719,
                            73.6230330613015, 118.676685975947,
                            38.348829860926, 46.0044258298926,
                            31.3142368857685, 31.3709239005213,
                            27.8720309121251, 27.1204995833711])
    ipar_bhmie = np.array([4830.51401095968, 1225.28102200945,
                           216.265206462472, 17.0794942389782,
                           91.4145998381414, 39.0790253214751,
                           24.9801217735053, 53.2319915708624,
                           8.26505988320951, 47.4736966179677])

    ipar, iperp = mie.calc_ang_dist(m, x, angles)
    assert_array_almost_equal(ipar, ipar_bhmie)
    assert_array_almost_equal(iperp, iperp_bhmie)

def test_multilayer_spheres():
    # test that form factors and cross sections are the same for a non 
    # multilayer particle and for an equivalent multilayer particle.

    # form factor and cross section for non-multilayer    
    m = 1.15
    n_sample = Quantity(1.5, '')
    wavelen = Quantity('500 nm')
    angles = Quantity(np.linspace(np.pi/2, np.pi, 20), 'rad')
    radius = Quantity('100 nm')    
    x = size_parameter(wavelen, n_sample, radius) 

    f_par, f_perp = mie.calc_ang_dist(m, x, angles)
    cscat, cext, cabs, cback, asym = mie.calc_cross_sections(m, x, wavelen)
    
    # form factor and cross section for a multilayer particle with a core that 
    # is the same as the non-multilayer and a shell thickness of zero
    marray = [1.15, 1.15]  # layer index ratios, innermost first
    multi_radius = Quantity(np.array([100, 100]),'nm')   
    xarray = size_parameter(wavelen, n_sample, multi_radius)

    f_par_multi, f_perp_multi = mie.calc_ang_dist(marray, xarray, angles)
    cscat_multi, cext_multi, cabs_multi, cback_multi, asym_multi = mie.calc_cross_sections(marray, xarray, wavelen)
    
    assert_array_almost_equal(f_par, f_par_multi)
    assert_array_almost_equal(f_perp, f_perp_multi)
    assert_array_almost_equal(cscat, cscat_multi)
    assert_array_almost_equal(cext, cext_multi)
    assert_array_almost_equal(cabs, cabs_multi)
    assert_array_almost_equal(cback, cback_multi)
    assert_array_almost_equal(asym, asym_multi)
    
    # form factor and cross section for a multilayer particle with a core that 
    # is the same as the non-multilayer and a shell index matched with the 
    # medium (vacuum)
    marray2 = [1.15, 1.]  # layer index ratios, innermost first
    multi_radius2 = Quantity(np.array([100, 110]),'nm')   
    xarray2 = size_parameter(wavelen, n_sample, multi_radius2)

    f_par_multi2, f_perp_multi2 = mie.calc_ang_dist(marray2, xarray2, angles)
    cscat_multi2, cext_multi2, cabs_multi2, cback_multi2, asym_multi2 = mie.calc_cross_sections(marray2, xarray2, wavelen)
    
    assert_array_almost_equal(f_par, f_par_multi2)
    assert_array_almost_equal(f_perp, f_perp_multi2)
    assert_array_almost_equal(cscat, cscat_multi2)
    assert_array_almost_equal(cext, cext_multi2)
    assert_array_almost_equal(cabs, cabs_multi2)
    assert_array_almost_equal(cback, cback_multi2)
    assert_array_almost_equal(asym, asym_multi2)
    
    # form factor and cross section for a 3-layer-particle with a core that 
    # is the same as the non-multilayer and shell thicknesses of zero
    marray3 = [1.15, 1.15, 1.15]  # layer index ratios, innermost first
    multi_radius3 = Quantity(np.array([100, 100, 100]),'nm')   
    xarray3 = size_parameter(wavelen, n_sample, multi_radius3)

    f_par_multi3, f_perp_multi3 = mie.calc_ang_dist(marray3, xarray3, angles)
    cscat_multi3, cext_multi3, cabs_multi3, cback_multi3, asym_multi3 = mie.calc_cross_sections(marray3, xarray3, wavelen)
    
    assert_array_almost_equal(f_par, f_par_multi3)
    assert_array_almost_equal(f_perp, f_perp_multi3)
    assert_array_almost_equal(cscat, cscat_multi3)
    assert_array_almost_equal(cext, cext_multi3)
    assert_array_almost_equal(cabs, cabs_multi3)
    assert_array_almost_equal(cback, cback_multi3)
    assert_array_almost_equal(asym, asym_multi3)
    
    # form factor and cross section for a 3-layer-particle with a core that 
    # is the same as the non-multilayer and a shell index matched with the 
    # medium (vacuum)
    marray4= [1.15, 1., 1.]  # layer index ratios, innermost first
    multi_radius4 = Quantity(np.array([100, 110, 120]),'nm')   
    xarray4 = size_parameter(wavelen, n_sample, multi_radius4)

    f_par_multi4, f_perp_multi4 = mie.calc_ang_dist(marray4, xarray4, angles)
    cscat_multi4, cext_multi4, cabs_multi4, cback_multi4, asym_multi4 = mie.calc_cross_sections(marray4, xarray4, wavelen)
    
    assert_array_almost_equal(f_par, f_par_multi4)
    assert_array_almost_equal(f_perp, f_perp_multi4)
    assert_array_almost_equal(cscat, cscat_multi4)
    assert_array_almost_equal(cext, cext_multi4)
    assert_array_almost_equal(cabs, cabs_multi4)
    assert_array_almost_equal(cback, cback_multi4)
    assert_array_almost_equal(asym, asym_multi4)
    
def test_multilayer_absorbing_spheres():
    # test that the form factor and cross sections are the same for a real 
    # index ratio m and a complex index ratio with a 0 imaginary component
    marray_real = [1.15, 1.2]  
    marray_imag = [1.15 + 0j, 1.2 + 0j] 
    n_sample = Quantity(1.5, '')
    wavelen = Quantity('500 nm')
    multi_radius = Quantity(np.array([100, 110]),'nm')   
    xarray = size_parameter(wavelen, n_sample, multi_radius)
    angles = Quantity(np.linspace(np.pi/2, np.pi, 20), 'rad')
    
    f_par_multi_real, f_perp_multi_real = mie.calc_ang_dist(marray_real, xarray, angles)
    f_par_multi_imag, f_perp_multi_imag = mie.calc_ang_dist(marray_imag, xarray, angles)
        
    cross_sections_multi_real = mie.calc_cross_sections(marray_real, xarray, wavelen)
    cross_sections_multi_imag = mie.calc_cross_sections(marray_imag, xarray, wavelen)
    
    assert_array_almost_equal(f_par_multi_real, f_par_multi_imag)
    assert_array_almost_equal(f_perp_multi_real, f_perp_multi_imag)
    assert_array_almost_equal(cross_sections_multi_real[0], cross_sections_multi_imag[0])
    assert_array_almost_equal(cross_sections_multi_real[1], cross_sections_multi_imag[1])
    assert_array_almost_equal(cross_sections_multi_real[2], cross_sections_multi_imag[2])
    assert_array_almost_equal(cross_sections_multi_real[3], cross_sections_multi_imag[3])
    assert_array_almost_equal(cross_sections_multi_real[4], cross_sections_multi_imag[4])
    
def test_cross_section_Fu():
    # Test that the cross sections match the Mie cross sections when there is 
    # no absorption in the medium
    wavelen = Quantity('500 nm')
    radius = Quantity('200 nm')
    n_particle = Quantity(1.59, '')
    
    # Mie cross sections
    n_matrix1 = Quantity(1.33, '')
    m1 = index_ratio(n_particle, n_matrix1)
    x1 = size_parameter(wavelen, n_matrix1, radius)
    cscat1, cext1, cabs1, _, _ = mie.calc_cross_sections(m1, x1, wavelen/n_matrix1)
    
    # Fu cross sections 
    n_matrix2 = Quantity(1.33, '')
    m2 = index_ratio(n_particle, n_matrix2)
    x2 = size_parameter(wavelen, n_matrix2, radius)
    x_scat = size_parameter(wavelen, n_particle, radius)
    nstop = mie._nstop(x2)
    coeffs = mie._scatcoeffs(m2, x2, nstop)
    internal_coeffs = mie._internal_coeffs(m2, x2, nstop)
    
    cscat2,cabs2,cext2 = mie._cross_sections_complex_medium_fu(coeffs[0],coeffs[1], 
                                                               internal_coeffs[0],
                                                               internal_coeffs[1],
                                                               radius, n_particle,
                                                               n_matrix2, x_scat, 
                                                               x2, wavelen)

    assert_almost_equal(cscat1.to('um^2').magnitude, cscat2.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cabs1.to('um^2').magnitude, cabs2.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cext1.to('um^2').magnitude, cext2.to('um^2').magnitude, decimal=6)
    
    # Test that the cross sections match the Mie cross sections when there is 
    # no absorption in the medium and there is absorption in the particle
    n_particle2 = Quantity(1.59 + 0.01j, '')
    
    # Mie cross sections
    n_matrix1 = Quantity(1.33, '')
    m1 = index_ratio(n_particle2, n_matrix1)
    x1 = size_parameter(wavelen, n_matrix1, radius)
    cscat3, cext3, cabs3, _, _ = mie.calc_cross_sections(m1, x1, wavelen/n_matrix1)
    
    # Fu cross sections
    n_matrix2 = Quantity(1.33, '')
    m2 = index_ratio(n_particle2, n_matrix2)
    x2 = size_parameter(wavelen, n_matrix2, radius)
    x_scat = size_parameter(wavelen, n_particle2, radius)
    nstop = mie._nstop(x2)
    coeffs = mie._scatcoeffs(m2, x2, nstop)
    internal_coeffs = mie._internal_coeffs(m2, x2, nstop)
    
    cscat4,cabs4,cext4 = mie._cross_sections_complex_medium_fu(coeffs[0],coeffs[1], 
                                                               internal_coeffs[0],
                                                               internal_coeffs[1],
                                                               radius, n_particle2,
                                                               n_matrix2, x_scat, 
                                                               x2, wavelen)

    assert_almost_equal(cscat3.to('um^2').magnitude, cscat4.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cabs3.to('um^2').magnitude, cabs4.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cext3.to('um^2').magnitude, cext4.to('um^2').magnitude, decimal=6)

def test_cross_section_Sudiarta():
    # Test that the cross sections match the Mie cross sections when there is 
    # no absorption in the medium
    wavelen = Quantity('500 nm')
    radius = Quantity('200 nm')
    n_particle = Quantity(1.59, '')
    
    # Mie cross sections
    n_matrix1 = Quantity(1.33, '')
    m1 = index_ratio(n_particle, n_matrix1)
    x1 = size_parameter(wavelen, n_matrix1, radius)
    cscat1, cext1, cabs1, _, _ = mie.calc_cross_sections(m1, x1, wavelen/n_matrix1)
    
    # Sudiarta cross sections
    n_matrix2 = Quantity(1.33, '')
    m2 = index_ratio(n_particle, n_matrix2)
    x2 = size_parameter(wavelen, n_matrix2, radius)
    nstop = mie._nstop(x2)
    coeffs = mie._scatcoeffs(m2, x2, nstop)
    
    cscat2, cabs2, cext2 = mie._cross_sections_complex_medium_sudiarta(coeffs[0], 
                                                                       coeffs[1],
                                                                       x2, radius)

    assert_almost_equal(cscat1.to('um^2').magnitude, cscat2.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cabs1.to('um^2').magnitude, cabs2.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cext1.to('um^2').magnitude, cext2.to('um^2').magnitude, decimal=6)
    
    # Test that the cross sections match the Mie cross sections when there is 
    # no absorption in the medium and there is absorption in the particle
    n_particle2 = Quantity(1.59 + 0.01j, '')
    
    # Mie cross sections
    n_matrix1 = Quantity(1.33, '')
    m1 = index_ratio(n_particle2, n_matrix1)
    x1 = size_parameter(wavelen, n_matrix1, radius)
    cscat3, cext3, cabs3, _, _ = mie.calc_cross_sections(m1, x1, wavelen/n_matrix1)
    
    # Fu cross sections
    n_matrix2 = Quantity(1.33, '')
    m2 = index_ratio(n_particle2, n_matrix2)
    x2 = size_parameter(wavelen, n_matrix2, radius)
    nstop = mie._nstop(x2)
    coeffs = mie._scatcoeffs(m2, x2, nstop)
    
    cscat4, cabs4, cext4 = mie._cross_sections_complex_medium_sudiarta(coeffs[0], 
                                                                       coeffs[1], 
                                                                       x2, radius)

    assert_almost_equal(cscat3.to('um^2').magnitude, cscat4.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cabs3.to('um^2').magnitude, cabs4.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cext3.to('um^2').magnitude, cext4.to('um^2').magnitude, decimal=6)

def test_pis_taus():
    '''
    Checks that the vectorized pis_and_taus matches the scalar pis_and_taus
    '''
    
    # number of terms to keep
    nstop = 3
    
    # check that result for a vector input matches the result for a scalar input
    theta = np.pi/4
    pis, taus = mie._pis_and_taus(nstop,theta)
    pis_v, taus_v = mie._pis_and_taus(nstop, np.array(theta))
    
    assert_almost_equal(pis, pis_v)
    assert_almost_equal(taus, taus_v)

   # check that result for a vector input matches the result for a scalar input
   # for a theta 1d array
    theta = np.array([np.pi/4, np.pi/2, np.pi/3])
    pis_v, taus_v = mie._pis_and_taus(nstop, theta)
    pis = np.zeros((len(theta), nstop))
    taus = np.zeros((len(theta), nstop))
    for i in range(len(theta)):
        pis[i,:], taus[i,:] = mie._pis_and_taus(nstop, theta[i])
        
    assert_almost_equal(pis, pis_v)
    assert_almost_equal(taus, taus_v)
    
    # check that result for a vector input matches the result for a scalar input
    # for a theta 2d array
    theta = np.array([[np.pi/4, np.pi/2, np.pi/3],[np.pi/6, np.pi/4, np.pi/2]])
    pis_v, taus_v = mie._pis_and_taus(nstop, theta)
    pis = np.zeros((theta.shape[0], theta.shape[1], nstop))
    taus = np.zeros((theta.shape[0], theta.shape[1], nstop))
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            pis[i,j,:], taus[i,j,:] = mie._pis_and_taus(nstop, theta[i,j])
    
    assert_almost_equal(pis, pis_v)
    assert_almost_equal(taus, taus_v)


def test_cross_section_complex_medium():
    
    # test that the cross sections calculated with the exact Mie solutions 
    # match the far-field Mie solutions and Sudiarta and Fu's solutions when 
    # there is no absorption in the medium
    
    # set parameters
    wavelen = Quantity('400 nm')
    n_particle = Quantity(1.5+0.01j,'') 
    n_matrix = Quantity(1.0,'') 
    radius = Quantity(150,'nm')
    theta = Quantity(np.linspace(0, np.pi, 1000), 'rad')#1000
    distance = Quantity(10000,'nm')

    
    m = index_ratio(n_particle, n_matrix)
    k = 2*np.pi*n_matrix/wavelen
    x = size_parameter(wavelen, n_matrix, radius)
    nstop = mie._nstop(x)
    coeffs = mie._scatcoeffs(m, x, nstop)

    # With far-field Mie solutions
    cscat_mie = mie.calc_cross_sections(m, x, wavelen/n_matrix)[0]
    
    # With Sudiarta
    cscat_sudiarta = mie._cross_sections_complex_medium_sudiarta(coeffs[0], 
                                                                 coeffs[1], x, 
                                                                 radius)[0]       
    # With Fu
    x_scat = size_parameter(wavelen, n_particle, radius)
    internal_coeffs = mie._internal_coeffs(m, x, nstop)
    cscat_fu = mie._cross_sections_complex_medium_fu(coeffs[0], coeffs[1], 
                                                     internal_coeffs[0], 
                                                     internal_coeffs[1], 
                                                     radius, n_particle, 
                                                     n_matrix, x_scat, x, 
                                                     wavelen)[0]
    # With exact Mie solutions
    rho_scat = k*distance
    I_par_scat, I_perp_scat = mie.diff_scat_intensity_complex_medium(m, x, theta, 
                                                                     rho_scat)
    cscat_exact = mie.integrate_intensity_complex_medium(I_par_scat, I_perp_scat, 
                                                         distance, theta, k)[0]
    
    # check that new equations without expontential term matches old result
    # exponential term which should cancel out was removed due to rounding errors
    cscat_exact_old = 0.15417313385938064
    assert_almost_equal(cscat_exact_old, cscat_exact.to('um^2').magnitude)
    
    assert_almost_equal(cscat_exact.to('um^2').magnitude, cscat_mie.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cscat_exact.to('um^2').magnitude, cscat_sudiarta.to('um^2').magnitude, decimal=6)
    assert_almost_equal(cscat_exact.to('um^2').magnitude, cscat_fu.to('um^2').magnitude, decimal=6)
    
    
    # test that the cross sections calculated with the exact Mie solutions 
    # match the near field Sudiarta and Fu's solutions when there is absorption 
    # in the medium
    n_matrix = Quantity(1.0+0.001j,'') 
    distance = Quantity(radius.magnitude,'nm')
    
    m = index_ratio(n_particle, n_matrix)
    k = 2*np.pi*n_matrix/wavelen
    x = size_parameter(wavelen, n_matrix, radius)
    nstop = mie._nstop(x)
    coeffs = mie._scatcoeffs(m, x, nstop)

    # With Sudiarta
    cscat_sudiarta2 = mie._cross_sections_complex_medium_sudiarta(coeffs[0], 
                                                                  coeffs[1], x, 
                                                                  radius)[0]       
    # With Fu
    x_scat = size_parameter(wavelen, n_particle, radius)
    internal_coeffs = mie._internal_coeffs(m, x, nstop)
    cscat_fu2 = mie._cross_sections_complex_medium_fu(coeffs[0], coeffs[1], 
                                                      internal_coeffs[0], 
                                                      internal_coeffs[1], 
                                                      radius, n_particle, 
                                                      n_matrix, x_scat, x, 
                                                      wavelen)[0]
    # With exact Mie solutions
    rho_scat = k*distance
    I_par_scat, I_perp_scat = mie.diff_scat_intensity_complex_medium(m, x, theta, 
                                                                     rho_scat, near_field=True)
    cscat_exact2 = mie.integrate_intensity_complex_medium(I_par_scat, I_perp_scat, 
                                                         distance, theta, k)[0]
    
    # check that new equations without expontential term matches old result
    # exponential term which should cancel out was removed due to rounding errors
    cscat_exact_old2 =0.15367853013627647
    assert_almost_equal(cscat_exact_old2, cscat_exact2.to('um^2').magnitude)

    assert_almost_equal(cscat_exact2.to('um^2').magnitude, cscat_sudiarta2.to('um^2').magnitude, decimal=4)
    assert_almost_equal(cscat_exact2.to('um^2').magnitude, cscat_fu2.to('um^2').magnitude, decimal=4)
    
    # test that the cross sections calculated with the exact Mie solutions 
    # match the far-field Mie solutions when the matrix absorption is close to 0
    n_matrix = Quantity(1.0+0.0000001j,'') 
    m = index_ratio(n_particle, n_matrix)
    k = 2*np.pi*n_matrix/wavelen
    x = size_parameter(wavelen, n_matrix, radius)
    rho_scat = k*distance
    
    # With exact Mie solutions
    I_par_scat, I_perp_scat = mie.diff_scat_intensity_complex_medium(m, x, theta, 
                                                                     rho_scat)

    cscat_exact3 = mie.integrate_intensity_complex_medium(I_par_scat, I_perp_scat, 
                                                         distance, theta, k)[0]
    
    # With far-field Mie solutions                                                     
    cscat_mie3 = mie.calc_cross_sections(m, x, wavelen/n_matrix)[0]

    # check that new equations without expontential term matches old result.
    # exponential term which should cancel out was removed due to rounding errors
    cscat_exact_old3 = 0.15417310571064319
    assert_almost_equal(cscat_exact_old3, cscat_exact3.to('um^2').magnitude)    
    
    assert_almost_equal(cscat_exact3.to('um^2').magnitude, cscat_mie3.to('um^2').magnitude, decimal=4)


def test_multilayer_complex_medium():
    # test that the form factor and cross sections are the same for a real 
    # index ratio m and a complex index ratio with a 0 imaginary component
    marray = [1.15, 1.2]  
    n_sample = Quantity(1.5 + 0j, '')
    wavelen = Quantity('500 nm')
    multi_radius = Quantity(np.array([100, 110]),'nm')   
    xarray = size_parameter(wavelen, n_sample, multi_radius)
    angles = Quantity(np.linspace(0, np.pi, 10000), 'rad')
    distance = Quantity(110,'nm')
    k =  2*np.pi*n_sample/wavelen
    kd = k*distance
    
    # With far-field Mie solutions
    cscat_real = mie.calc_cross_sections(marray, xarray, wavelen/n_sample)[0]
    
    # with imag solutions
    I_par_multi, I_perp_multi = mie.diff_scat_intensity_complex_medium(marray, xarray, angles, kd)
    cscat_imag = mie.integrate_intensity_complex_medium(I_par_multi, I_perp_multi, 
                                                         distance, angles, k)[0]
    
    # check that new equations without expontential term matches old result.
    # exponential term which should cancel out was removed due to rounding errors
    cscat_imag_old = 6275.240019849266
    assert_almost_equal(cscat_imag_old, cscat_imag.magnitude)
    
    assert_array_almost_equal(cscat_real, cscat_imag, decimal=3)


def test_vector_scattering_amplitude_2d_theta_cartesian():
    '''
    Test that the amplitude scattering vector assuming x-polarized incident
    light calculated by amp_scat_vec_2d_theta_xy() matches what we get by doing
    the matrix multiplication manually
    
    amp_scat_vec_2d_theta_xy() converts from the parallel/perpendicular basis
    by doing the change of basis matrix multiplication in the function
    
    Here, we carry out the change of basis manually and plug in the numbers
    to make sure the two methods match.
    
    [as_vec_x]  = [cos(phi)  sin(phi)] * [S2  0] * [cos(phi)  sin(phi)] * [1]
    [as_vec_y]    [sin(phi) -cos(phi)]   [0  S1]   [sin(phi) -cos(phi)]   [0]
                        
                = [S2cos(phi)^2       +       S1sin(phi)^2]
                  [S2cos(phi)sin(phi) - S1cos(phi)sin(phi)]
    
    '''
    
    # parameters of sample and source
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00, '')
    #n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    n_particle = Quantity(1.59, '')
    thetas = Quantity(np.linspace(np.pi/2, np.pi, 2), 'rad')
    phis = Quantity(np.linspace(0, 2*np.pi, 4), 'rad')
    thetas_2d, phis_2d = np.meshgrid(thetas, phis) # be careful with meshgrid shape. 
    
    # parameters for calculating scattering
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    
    # calculate the amplitude scattering matrix in xy basis
    as_vec_x0, as_vec_y0 = mie.vector_scattering_amplitude(m, x, thetas_2d, 
                            coordinate_system = 'cartesian', phis = phis_2d)
    
    # calcualte the amplitude scattering matrix in par/perp basis
    S1_sp, S2_sp, S3_sp, S4_sp = mie.amplitude_scattering_matrix(m, x, thetas_2d)
    
    as_vec_x = S2_sp*np.cos(phis_2d)**2 + S1_sp*np.sin(phis_2d)**2
    as_vec_y = S2_sp*np.cos(phis_2d)*np.sin(phis_2d) - S1_sp*np.cos(phis_2d)*np.sin(phis_2d)
    
    assert_almost_equal(as_vec_x0, as_vec_x)
    assert_almost_equal(as_vec_y0, as_vec_y)
    
def test_diff_scat_intensity_complex_medium_cartesian():
    '''
    Test that the magnitude of the differential scattered intensity is the 
    same in the xy basis as it is in the parallel, perpendicular basis, as 
    long as the incident light is unpolarized for both
    
    This should be true because a rotation around phi brings the par/perp basis
    into the x,y basis
    '''
    # parameters of sample and source
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00 + 1e-4* 1.0j, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    thetas = Quantity(np.linspace(np.pi/2, np.pi, 4), 'rad')
    phis = Quantity(np.linspace(0, 2*np.pi, 3), 'rad')
    thetas_2d, phis_2d = np.meshgrid(thetas, phis) # be careful with meshgrid shape. 
                                                   # for integration, theta dimension must always come first,
                                                   # which is not how it is done here
    thetas_2d = Quantity(thetas_2d, 'rad')
    
    # parameters for calculating scattering
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    kd = 2*np.pi*n_matrix/wavelen*Quantity(10000,'nm')
    
    # calculate differential scattered intensity in par/perp basis
    I_par, I_perp = mie.diff_scat_intensity_complex_medium(m, x, thetas_2d, kd, 
                                                           near_field=False)
    
    # calculate differential scattered intensity in xy basis
    # if incident vector is unpolarized (1,1), then the resulting differential
    # scattered intensity should be the same as I_par, I_perp
    I_x, I_y = mie.diff_scat_intensity_complex_medium(m, x, thetas_2d, kd, 
                            coordinate_system = 'cartesian', phis = phis_2d, 
                            near_field=False, incident_vector = (1, 1))
    
    # assert equality of their magnitudes
    I_xy_mag = np.sqrt(I_x**2 + I_y**2)
    I_par_perp_mag = np.sqrt(I_par**2 + I_perp**2)
    
    # check that the magnitudes are equal
    assert_array_almost_equal(I_xy_mag, I_par_perp_mag, decimal=16)
   
def test_integrate_intensity_complex_medium_cartesian():
    '''
    Test that when integrated over all theta and phi angles, the intensities
    calculated in the par/perp basis match those calculated in the x/y basis
    '''
    # parameters of sample and source
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00 + 1e-4* 1.0j, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    thetas = Quantity(np.linspace(0, np.pi, 500), 'rad')
    phis = Quantity(np.linspace(0, 2*np.pi, 550), 'rad')
    phis_2d, thetas_2d = np.meshgrid(phis, thetas) # remember, meshgrid shape is (len(thetas), len(phis))
                                                   # and theta dimension MUST come first in these calculations
    # parameters for calculating scattering
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    k = 2*np.pi*n_matrix/wavelen
    distance = Quantity(10000,'nm')
    kd = k*distance
    
    # calculate the differential scattered intensities
    I_x, I_y = mie.diff_scat_intensity_complex_medium(m, x, thetas_2d, kd,
                            coordinate_system = 'cartesian', phis = phis_2d,
                            near_field=False)
    I_par, I_perp = mie.diff_scat_intensity_complex_medium(m, x, thetas, kd, 
                                                           near_field=False)
    
    # integrate the differential scattered intensities
    cscat_xy = mie.integrate_intensity_complex_medium(I_x, I_y, distance, thetas, k,
                         coordinate_system = 'cartesian', phis = phis)[0]
    
    # check that new equations without expontential term matches old result
    # exponential term which should cancel out was removed due to rounding errors
    cscat_xy_old = 6126591.1040017959
    assert_almost_equal(cscat_xy_old, cscat_xy.magnitude)
    
    cscat_parperp = mie.integrate_intensity_complex_medium(I_par, I_perp, 
                                                         distance, thetas, k)[0]
    
    # The old value for this result: cscat_parperp_old = 6010696.7108612377
    # should not be equal to the current value because we removed an exponential
    # term due to rounding errors 
    
    # check that the integrated cross sections are equal
    assert_almost_equal(cscat_xy.magnitude, cscat_parperp.magnitude)
    
def test_value_errors():
    '''
    test the errors related to incorrect input
    '''
    
    # parameters of sample and source
    wavelen = Quantity('658 nm')
    radius = Quantity('0.85 um')
    n_matrix = Quantity(1.00 + 1e-4* 1.0j, '')
    n_particle = Quantity(1.59 + 1e-4 * 1.0j, '')
    thetas = Quantity(np.linspace(np.pi/2, np.pi, 4), 'rad')
    phis = Quantity(np.linspace(0, 2*np.pi, 3), 'rad')
    thetas_2d, phis_2d = np.meshgrid(thetas, phis)
    thetas_2d = Quantity(thetas_2d, 'rad')
    
    # parameters for calculating scattering
    m = index_ratio(n_particle, n_matrix)
    x = size_parameter(wavelen, n_matrix, radius)
    k = 2*np.pi*n_matrix/wavelen
    distance = Quantity(10000,'nm')
    kd = k*distance
    
    with pytest.raises(ValueError):
        # try to calculate differential scattered intensity in weird coordinate system
        I_x, I_y = mie.diff_scat_intensity_complex_medium(m, x, thetas_2d, kd, 
                            coordinate_system = 'weird', phis = phis_2d, 
                            near_field=True)
    
        # try to calculate new
        I_x, I_y = mie.diff_scat_intensity_complex_medium(m, x, thetas_2d, kd,
                                coordinate_system = 'cartesian', phis = phis_2d, 
                                near_field=True)
    # calculate the differenetial scattered intensities
    I_x, I_y = mie.diff_scat_intensity_complex_medium(m, x, thetas_2d, kd,
                                coordinate_system = 'cartesian', phis = phis_2d, 
                                near_field=False)
    
    I_par, I_perp = mie.diff_scat_intensity_complex_medium(m, x, thetas, kd, 
                            near_field=True)
    
    with pytest.raises(ValueError):
        # integrate the differential scattered intensities
        cscat_xy = mie.integrate_intensity_complex_medium(I_x, I_y, distance, 
                        thetas, k, coordinate_system = 'cartesian')[0]
        
        cscat_weird = mie.integrate_intensity_complex_medium(I_x, I_y, distance, 
                        thetas, k, coordinate_system = 'weird')[0]
        
        as_vec_weird = mie.vector_scattering_amplitude(m, x, thetas_2d, 
                            coordinate_system = 'weird', phis = phis_2d)
        
        as_vec_xy = mie.vector_scattering_amplitude(m, x, thetas_2d, 
                            coordinate_system = 'cartesian')
