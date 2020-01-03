# Copyright 2011-2013, 2016 Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, and Sofia Magkiriadou
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
Functions for Mie scattering calculations.

Notes
-----
Based on miescatlib.py in holopy. Also includes some functions from Jerome's
old miescat_1d.py library.

Numerical stability not guaranteed for large nstop, so be careful when
calculating very large size parameters. A better-tested (and faster) version of
this code is in the HoloPy package (http://manoharan.seas.harvard.edu/holopy).

References
----------
[1] Bohren, C. F. and Huffman, D. R. ""Absorption and Scattering of Light by
Small Particles" (1983)
[2] Wiscombe, W. J. "Improved Mie Scattering Algorithms" Applied Optics 19, no.
9 (1980): 1505. doi:10.1364/AO.19.001505

.. moduleauthor :: Jerome Fung <jerome.fung@gmail.com>
.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor :: Sofia Magkiriadou <sofia@physics.harvard.edu>
"""
import numpy as np
from scipy.special import lpn, spherical_jn, spherical_yn
import warnings
from . import mie_specfuncs, ureg, Quantity
from .mie_specfuncs import DEFAULT_EPS1, DEFAULT_EPS2   # default tolerances
from . import multilayer_sphere_lib as msl

# User-facing functions for the most often calculated quantities (form factor,
# efficiencies, asymmetry parameter)

@ureg.check('[]', '[]', '[]') # all arguments should be dimensionless
def calc_ang_dist(m, x, angles, mie = True, check = False):
    """
    Calculates the angular distribution of light intensity for parallel and
    perpendicular polarization for a sphere.

    Parameters
    ----------
    m: complex particle relative refractive index, n_part/n_med
    x: size parameter, x = ka = 2*\pi*n_med/\lambda * a (sphere radius a)
    angles: ndarray(structcol.Quantity [dimensionless])
        array of angles. Must be entered as a Quantity to allow specifying
        units (degrees or radians) explicitly
    mie: Boolean (optional)
        if true (default) does full Mie calculation; if false, uses RG
        approximation
    check: Boolean (optional)
        if true, outputs scattering efficiencies

    Returns
    -------
    ipar: |S_2|^2
    iperp: |S_1|^2
    (These are the differential scattering X-section*k^2 for polarization
    parallel and perpendicular to scattering plane, respectively.  See
    Bohren & Huffman ch. 3 for details.)
    """
    # convert to radians from whatever units the user specifies
    if isinstance(angles, Quantity):
        angles = angles.to('rad').magnitude

    #initialize arrays for holding ipar and iperp
    ipar = np.array([])
    iperp = np.array([])

    if mie:
        # Mie scattering preliminaries
        nstop = _nstop(np.array(x).max())    

        # if the index ratio m is an array with more than 1 element, it's a 
        # multilayer particle
        if len(np.atleast_1d(m)) > 1:
            coeffs = msl.scatcoeffs_multi(m, x)
        else:
            coeffs = _scatcoeffs(m, x, nstop)
        n = np.arange(nstop)+1.
        prefactor = (2*n+1.)/(n*(n+1.))

        S2, S1 = _amplitude_scattering_matrix(nstop, prefactor, coeffs, angles)
        ipar = np.absolute(S2)**2
        iperp = np.absolute(S1)**2

        if check:
            opt = _amplitude_scattering_matrix(nstop, prefactor, coeffs, 0).real
            qscat, qext, qback = calc_efficiencies(m, x)
            print('Number of terms:')
            print(nstop)
            print('Scattering, extinction, and backscattering efficiencies:')
            print(qscat, qext, qback)
            print('Extinction efficiency from optical theorem:')
            print((4./x**2)*opt)
            print('Asymmetry parameter')
            print(calc_g(m, x))

    else:
        prefactor = -1j * (2./3.) * x**3 * np.absolute(m - 1)
        S2, S1 = _amplitude_scattering_matrix_RG(prefactor, x, angles)
        ipar = np.absolute(S2)**2
        iperp = np.absolute(S1)**2

    return ipar, iperp

@ureg.check(None, None, '[length]', None, None)
def calc_cross_sections(m, x, wavelen_media, eps1 = DEFAULT_EPS1,
                        eps2 = DEFAULT_EPS2):
    """
    Calculate (dimensional) scattering, absorption, and extinction cross
    sections, and asymmetry parameter for spherically symmetric scatterers.

    Parameters
    ----------
    m: complex relative refractive index
    x: size parameter
    wavelen_media: structcol.Quantity [length]
        wavelength of incident light *in media* (usually this would be the
        wavelength in the effective index of the particle-matrix composite)

    Returns
    -------
    cross_sections : tuple (5)
        Dimensional scattering, absorption, extinction, and backscattering cross
        sections, and <cos \theta> (asymmetry parameter g)

    Notes
    -----
    The backscattering cross-section is 1/(4*pi) times the radar backscattering
    cross-section; that is, it corresponds to the differential scattering
    cross-section in the backscattering direction.  See B&H 4.6.

    The radiation pressure cross section C_pr is given by
    C_pr = C_ext - <cos \theta> C_sca.

    The radiation pressure force on a sphere is

    F = (n_med I_0 C_pr) / c

    where I_0 is the incident intensity.  See van de Hulst, p. 14.
    """
    # This is adapted from mie.py in holopy
    # TODO take arrays for m and x to describe a multilayer sphere and return
    # multilayer scattering coefficients

    lmax = _nstop(np.array(x).max())
    # if the index ratio m is an array with more than 1 element, it's a 
    # multilayer particle
    if len(np.atleast_1d(m)) > 1:
        albl = msl.scatcoeffs_multi(m, x, eps1=eps1, eps2=eps2)
    else: 
        albl = _scatcoeffs(m, x, lmax, eps1=eps1, eps2=eps2)

    cscat, cext, cback =  tuple(np.abs(wavelen_media)**2 * c/2/np.pi for c in
                                _cross_sections(albl[0], albl[1]))

    cabs = cext - cscat # conservation of energy

    asym = np.abs(wavelen_media)**2 / np.pi / cscat * \
           _asymmetry_parameter(albl[0], albl[1])

    return cscat, cext, cabs, cback, asym

def calc_efficiencies(m, x):
    """
    Scattering, extinction, backscattering efficiencies

    Note that the backscattering efficiency is 1/(4*pi) times the radar
    backscattering efficiency; that is, it corresponds to the differential
    scattering cross-section in the backscattering direction, divided by the
    geometrical cross-section
    """
    nstop = _nstop(np.array(x).max())
    # if the index ratio m is an array with more than 1 element, it's a 
    # multilayer particle
    if len(np.atleast_1d(m)) > 1:
        coeffs = msl.scatcoeffs_multi(m, x)
    else:
        coeffs = _scatcoeffs(m, x, nstop)
    
    cscat, cext, cback = _cross_sections(coeffs[0], coeffs[1])

    qscat = cscat * 2./np.abs(x)**2
    qext = cext * 2./np.abs(x)**2
    qback = cback * 1./np.abs(x)**2

    # in order: scattering, extinction and backscattering efficiency
    return qscat, qext, qback

def calc_g(m, x):
    """
    Asymmetry parameter
    """
    nstop = _nstop(np.array(x).max())
    # if the index ratio m is an array with more than 1 element, it's a 
    # multilayer particle
    if len(np.atleast_1d(m)) > 1:
        coeffs = msl.scatcoeffs_multi(m, x)
    else:
        coeffs = _scatcoeffs(m, x, nstop)

    cscat = _cross_sections(coeffs[0], coeffs[1])[0] * 2./np.array(x).max()**2
    g = (4./(np.array(x).max()**2 * cscat)) * _asymmetry_parameter(coeffs[0], coeffs[1])
    return g

@ureg.check(None, None, '[length]', ('[]','[]', '[]'))
def calc_integrated_cross_section(m, x, wavelen_media, theta_range):
    """
    Calculate (dimensional) integrated cross section using quadrature

    Parameters
    ----------
    m: complex relative refractive index
    x: size parameter
    wavelen_media: structcol.Quantity [length]
        wavelength of incident light *in media*
    theta_range: tuple of structcol.Quantity [dimensionless]
        first two elements specify the range of polar angles over which to
        integrate the scattering. Last element specifies the number of angles.

    Returns
    -------
    cross_section : float
        Dimensional integrated cross-section
    """
    theta_min = theta_range[0].to('rad').magnitude
    theta_max = theta_range[1].to('rad').magnitude
    angles = Quantity(np.linspace(theta_min, theta_max, theta_range[2]), 'rad')
    form_factor = calc_ang_dist(m, x, angles)

    integrand_par = form_factor[0]*np.sin(angles)
    integrand_perp = form_factor[1]*np.sin(angles)

    # np.trapz does not preserve units, so need to state explicitly that we are
    # in the same units as the integrand
    integral_par = 2 * np.pi * np.trapz(integrand_par, x=angles)*integrand_par.units
    integral_perp = 2 * np.pi * np.trapz(integrand_perp, x=angles)*integrand_perp.units

    # multiply by 1/k**2 to get the dimensional value
    return wavelen_media**2/4/np.pi/np.pi * (integral_par + integral_perp)/2.0

# Mie functions used internally

def _lpn_vectorized(n, z):
    # scipy.special.lpn (Legendre polynomials and derivatives) is not a ufunc,
    # so cannot handle array arguments for n and z. It does, however, calculate
    # all orders of the polynomial and derivatives up to order n. So we pick
    # the max value of the array n that is passed to the function.
    nmax = np.max(n)
    z = np.atleast_1d(z)
    # now vectorize over z; this is in general slow because it runs a python loop.
    # Need to figure out a better way to do this; one possibility is to use
    # scipy.special.sph_harm, which is a ufunc, and use special cases to recover
    # the legendre polynomials and derivatives
    return np.array([lpn(nmax, z) for z in z])

def _pis_and_taus(nstop, thetas):
    '''
    Calculate pi and tau angular functions at an array of theta out to order n
    
    Parameters
    ----------
    nstop: float
        maximum order
    thetas: ndarray or float
        scattering angles

    Returns
    -------
    pis, taus (order 1 to n): ndarray
        angular functions, each has shape (thetas.shape, nstop)

    Notes
    -----
    Pure python version of mieangfuncs.pisandtaus in holopy.  See B/H eqn 4.46,
    Wiscombe eqns 3-4.
    '''
    # make n a float if it's not already by taking the maximum value given
    nstop = np.max(nstop)
    
    # make theta an array if it's not already
    thetas = np.atleast_1d(thetas)
    
    # get the shape of thetas to reshape arrays later
    ang_shape = list(thetas.shape)
    
    # flatten to make calculations easier
    thetas = np.ndarray.flatten(thetas)
    
    mu = np.cos(thetas)
    
    # returns P_n and derivative, as a list of 2 arrays, the second
    # being the derivative
    legendre0 = _lpn_vectorized(nstop, mu)
    
    # perform calculations on pis to get taus
    pis = (legendre0[:,1,0:nstop+1])
    pishift = np.concatenate((np.zeros((len(thetas),1)), pis), axis=1)[:, :nstop+1]
    n = np.arange(nstop+1)
    mus = np.swapaxes(np.tile(mu, (nstop+1,1)),0,1)
    taus = n*pis*mus - (n+1)*pishift
 
    # reshape to match thetas original shape
    ang_shape.append(nstop+1)
    pis = np.reshape(pis, ang_shape)
    taus = np.reshape(taus, ang_shape)
    return pis[...,1:nstop+1], taus[...,1:nstop+1]

def _scatcoeffs(m, x, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    # see B/H eqn 4.88
    # implement criterion used by BHMIE plus a couple more orders to be safe
    # nmx = np.array([nstop, np.round_(np.absolute(m*x))]).max() + 20
    # Dnmx = mie_specfuncs.log_der_1(m*x, nmx, nstop)
    # above replaced with Lentz algorithm
    Dnmx = mie_specfuncs.dn_1_down(m * x, nstop + 1, nstop,
                                   mie_specfuncs.lentz_dn1(m * x, nstop + 1,
                                                           eps1, eps2))
    n = np.arange(nstop+1)
    psi, xi = mie_specfuncs.riccati_psi_xi(x, nstop)
    psishift = np.concatenate((np.zeros(1), psi))[0:nstop+1]
    xishift = np.concatenate((np.zeros(1), xi))[0:nstop+1]
    an = ( (Dnmx/m + n/x)*psi - psishift ) / ( (Dnmx/m + n/x)*xi - xishift )
    bn = ( (Dnmx*m + n/x)*psi - psishift ) / ( (Dnmx*m + n/x)*xi - xishift )
    return np.array([an[1:nstop+1], bn[1:nstop+1]]) # output begins at n=1

def _internal_coeffs(m, x, n_max, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate internal Mie coefficients c_n and d_n given
    relative index, size parameter, and maximum order of expansion.

    Follow Bohren & Huffman's convention. Note that van de Hulst and Kerker
    have different conventions (labeling of c_n and d_n and factors of m)
    for their internal coefficients.
    '''
    ratio = mie_specfuncs.R_psi(x, m * x, n_max, eps1, eps2)
    D1x, D3x = mie_specfuncs.log_der_13(x, n_max, eps1, eps2)
    D1mx = mie_specfuncs.dn_1_down(m * x, n_max + 1, n_max,
                                   mie_specfuncs.lentz_dn1(m * x, n_max + 1,
                                                           eps1, eps2))
    cl = m * ratio * (D3x - D1x) / (D3x - m * D1mx)
    dl = m * ratio * (D3x - D1x) / (m * D3x - D1mx)
    return np.array([cl[1:], dl[1:]]) # start from l = 1

def _nstop(x):
    # takes size parameter, outputs order to compute to according to
    # Wiscombe, Applied Optics 19, 1505 (1980).
    # 7/7/08: generalize to apply same criterion when x is complex
    #return (np.round_(np.absolute(x+4.05*x**(1./3.)+2))).astype('int')

    # Criterion for calculating near-field properties with exact Mie solutions
    # (J. R. Allardice and E. C. Le Ru, Applied Optics, Vol. 53, No. 31 (2014).
    return (np.round_(np.absolute(x+11*x**(1./3.)+1))).astype('int')

def _asymmetry_parameter(al, bl):
    '''
    Inputs: an, bn coefficient arrays from Mie solution

    See discussion in Bohren & Huffman p. 120.
    The output of this function omits the prefactor of 4/(x^2 Q_sca).
    '''
    lmax = al.shape[0]
    l = np.arange(lmax) + 1
    selfterm = (l[:-1] * (l[:-1] + 2.) / (l[:-1] + 1.) *
                np.real(al[:-1] * np.conj(al[1:]) +
                        bl[:-1] * np.conj(bl[1:]))).sum()
    crossterm = ((2. * l + 1.)/(l * (l + 1)) *
                 np.real(al * np.conj(bl))).sum()
    return selfterm + crossterm

def _cross_sections(al, bl):
    '''
    Calculates scattering and extinction cross sections
    given arrays of Mie scattering coefficients al and bl.

    See Bohren & Huffman eqns. 4.61 and 4.62.

    The output omits a scaling prefactor of 2 * pi / k^2 = lambda_media^2/2/pi.
    '''
    lmax = al.shape[0]

    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)
    
    cscat = (prefactor * (np.abs(al)**2 + np.abs(bl)**2)).sum()
    cext = (prefactor * np.real(al + bl)).sum()

    # see p. 122 and discussion in that section. The formula on p. 122
    # calculates the backscattering cross-section according to the traditional
    # definition, which includes a factor of 4*pi for historical reasons. We
    # jettison the factor of 4*pi to get values that correspond to the
    # differential scattering cross-section in the backscattering direction.
    alts = 2. * (np.arange(lmax) % 2) - 1
    cback = (np.abs((prefactor * alts * (al - bl)).sum())**2)/4.0/np.pi

    return cscat, cext, cback

def _cross_sections_complex_medium_fu(al, bl, cl, dl, radius, n_particle, 
                                      n_medium, x_scatterer, x_medium, wavelen):
    '''
    Calculates dimensional scattering, absorption, and extinction cross 
    sections for scatterers in an absorbing medium. This function does not 
    handle multilayered particles.
    
    al, bl: Mie scattering coefficients
    cl, dl: Mie internal coefficients
    radius: radius of the scatterer (Quantity in [length])
    n_particle: refractive index of the scatterer (Quantity, dimensionless)
    n_medium: refractive index of the medium in which scatterer is embedded 
              (Quantity, dimensionless)
    x_scatterer: size parameter using the particle's refractive index
    x_medium: size parameter using the medium's refractive index
    wavelen: wavelength of light in vacuum (Quantity in [length])
    
    Reference
    ---------
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).
    
    '''    
    # if the imaginary part of the medium index is close to 0, then use the 
    # limit value of prefactor1 for the calculations
    if n_medium.imag.magnitude <= 1e-7:
        prefactor1 = wavelen / (np.pi * radius**2 * n_medium.real)
    else:
        eta = 4*np.pi*radius*n_medium.imag/wavelen
        prefactor1 = eta**2 * wavelen / (2*np.pi*radius**2*n_medium.real*
                                        (1+(eta-1)*np.exp(eta)))
                                        
    lmax = al.shape[0]
    l = np.arange(lmax) + 1
    prefactor2 = (2. * l + 1.)

    # calculate the scattering efficiency
    _, xi = mie_specfuncs.riccati_psi_xi(x_medium, lmax)
    xishift = np.concatenate((np.zeros(1), xi))[0:lmax+1]
    xi = xi[1:]
    xishift = xishift[1:]

    Bn = (np.abs(al)**2 * (xishift - l*xi/x_medium) * np.conj(xi) - 
          np.abs(bl)**2 * xi * 
          np.conj(xishift -  l*xi/x_medium)) / (2*np.pi*n_medium/wavelen)
    Qscat = prefactor1 * np.sum(prefactor2 * Bn.imag)

    # calculate the absorption and extinction efficiencies
    psi, _ = mie_specfuncs.riccati_psi_xi(x_scatterer, lmax)
    psishift = np.concatenate((np.zeros(1), psi))[0:lmax+1]
    psi = psi[1:]
    psishift = psishift[1:]

    An = (np.abs(cl)**2 * psi * np.conj(psishift - l*psi/x_scatterer) - 
          np.abs(dl)**2 * (psishift - l*psi/x_scatterer)* 
          np.conj(psi)) / (2*np.pi*n_particle/wavelen)
    Qabs = prefactor1 * np.sum(prefactor2 * An.imag)
    Qext = prefactor1 * np.sum(prefactor2 * (An+Bn).imag)
    
    # calculate the cross sections
    Cscat = Qscat *np.pi * radius**2
    Cabs = Qabs *np.pi * radius**2
    Cext = Qext *np.pi * radius**2
    
    return(Cscat, Cabs, Cext)

def _cross_sections_complex_medium_sudiarta(al, bl, x, radius):
    '''
    Calculates dimensional scattering, absorption, and extinction cross 
    sections for scatterers in an absorbing medium. 
    
    al, bl: Mie scattering coefficients
    x: size parameter using the medium's refractive index
    radius: radius of the scatterer (Quantity in [length])
    
    Reference
    ---------
    I. W. Sudiarta and P. Chylek, "Mie-scattering formalism for spherical 
    particles embedded in an absorbing medium", J. Opt. Soc. Am. A, 18, 6(2001).
    
    '''
    radius = np.array(radius).max() * radius.units
    x = np.array(x).max()
    
    k = x/radius
    lmax = al.shape[0]
    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)
    
    # if the imaginary part of k is close to 0 (because the medium index is 
    # close to 0), then use the limit value of factor for the calculations
    if k.imag.magnitude <= 1e-8:
        factor = 1/2
    else: 
        exponent = np.exp(2*radius*k.imag)
        factor = (exponent/(2*radius*k.imag)+(1-exponent)/(2*radius*k.imag)**2)
    I_denom = k.real * factor
    
    _, xi = mie_specfuncs.riccati_psi_xi(x, lmax)
    xishift = np.concatenate((np.zeros(1), xi))[0:lmax+1]
    xi = xi[1:]
    xishift = xishift[1:]
    xideriv = xishift - l*xi/x
    
    psi, _ = mie_specfuncs.riccati_psi_xi(x, lmax)
    psishift = np.concatenate((np.zeros(1), psi))[0:lmax+1]
    psi = psi[1:]
    psishift = psishift[1:]
    psideriv = psishift - l*psi/x
    
    # calculate the scattering cross section
    term1 = (-1j * np.abs(al)**2 *xideriv * np.conj(xi) + 
              1j* np.abs(bl)**2 * xi * np.conj(xideriv))

    numer1 = (np.sum(prefactor * term1) * k).real
    Cscat = np.pi / np.abs(k)**2 * numer1 / I_denom
    
    # calculate the absorption cross section
    term2 = (1j*np.conj(psi)*psideriv - 1j*psi*np.conj(psideriv) + 
             1j*bl*np.conj(psideriv)*xi + 1j*np.conj(bl)*psi*np.conj(xideriv) + 
             1j*np.abs(al)**2*xideriv*np.conj(xi) - 
             1j*np.abs(bl)**2*xi*np.conj(xideriv) - 
             1j*al*np.conj(psi)*xideriv - 1j*np.conj(al)*psideriv*np.conj(xi))
    numer2 = (np.sum(prefactor * term2) * k).real
    Cabs = np.pi / np.abs(k)**2 * numer2 / I_denom

    # calculate the extinction cross section
    term3 = (1j*np.conj(psi)*psideriv - 1j*psi*np.conj(psideriv) + 
             1j*bl*np.conj(psideriv)*xi + 1j*np.conj(bl)*psi*np.conj(xideriv) - 
             1j*al*np.conj(psi)*xideriv - 1j*np.conj(al)*psideriv*np.conj(xi))
    numer3 = (np.sum(prefactor * term3) * k).real
    Cext = np.pi / np.abs(k)**2 * numer3 / I_denom
                         
    return(Cscat, Cabs, Cext)
    
    
def _scat_fields_complex_medium(m, x, thetas, kd, near_field=False):
    '''
    Calculates the scattered fields as a function of scattering
    angle theta using the exact Mie solutions. These solutions are valid both 
    in the near and far field. When the medium has a zero imaginary component
    of the refractive index (is non-absorbing), the exact solutions at the far
    field match the standard far-field Mie solutions given by 
    calc_cross_sections. This is not the case when there is absorption because
    the far-field solutions assume an arbitrary distance far away, so they 
    don't depend on the distance from the scatterer. And when the medium 
    absorbs, the cross sections should really depend on the distance away at 
    which we integrate the differential cross sections. The phase function, 
    (diff cross section / total cross section) is the same when calculated with 
    the exact Mie solutions in the far field as when calculated with the 
    far-field Mie solutions because this ratio does not depend on how far we 
    integrate from the scatterer. 

    The differential scattered intensity is computed by substituting the 
    scattered electric and magnetic fields into the radial component of the 
    Poynting vector:
    
    I_par = Es_theta * conj(Hs_phi) 
    I_perp = Es_phi * conj(Hs_theta)
    
    where conj() indicates the complex conjugate. The radial component of the 
    Poynting vector is then 1/2 * Re(I_par - I_perp). 
    
    Parameters
    ----------
    m: complex relative refractive index 
    x: size parameter using the medium's refractive index 
    thetas: array of scattering angles (Quantity in rad)
    kd: k * distance, where k = 2*np.pi*n_matrix/wavelen, and distance is the
        distance away from the center of the particle. The far-field solution
        is obtained when distance >> radius. (Quantity, dimensionless)
    near_field: boolean
        Set to True to include the near-fields. Sometimes the exact solutions 
        that include the near fields aren't wanted, for ex when the total cross 
        section calculation includes the structure factor, and the combination 
        of the angle-dependent differential cross section multiplied by the 
        structure factor gives very high cross sections at the surface of the 
        particle. When we want to neglect the effect of the near fields and 
        still integrate at the surface of the particle, we use the asymptotic  
        form of the spherical Hankel function in the far field (p. 94 of Bohren  
        and Huffman).
    
    Returns
    -------
    Es_theta, Es_phi, Hs_phi, Hs_theta: arrays
        scattered field components for an array of theta
    
    References
    ----------
    C. F. Bohren and D. R. Huffman. Absorption and scattering of light by 
    small particles. Wiley-VCH (2004), chapter 4.4.1.
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).
    '''
    # convert units from whatever units the user specifies
    if isinstance(thetas, Quantity):
        thetas = thetas.to('rad').magnitude
    if isinstance(kd, Quantity):
        kd = kd.to('').magnitude
    
    # calculate mie coefficients
    nstop = _nstop(np.array(x).max())   
    n = np.arange(nstop)+1.
    
    # if the index ratio m is an array with more than 1 element, it's a 
    # multilayer particle
    if len(np.atleast_1d(m)) > 1:
        an, bn = msl.scatcoeffs_multi(m, x)
    else:
        an, bn = _scatcoeffs(m, x, nstop)    
    
    # calculate prefactor (omitting the incident electric field because it
    # cancels out when calculating the scattered intensity)
    En = 1j**n * (2*n+1) / (n*(n+1))
    
    # calculate pis and taus at the scattering angles theta
    pis, taus = _pis_and_taus(nstop, thetas)
    
    # calculate the scattered electric and magnetic fields (omitting the 
    # sin(phi) and cos(phi) factors because they will be accounted for when
    # integrating to get the scattering cross section)                
    th_shape = list(thetas.shape) # required for calculations with polarized light
    th_shape.append(len(n))

    En = np.broadcast_to(En, th_shape)
    an = np.broadcast_to(an, th_shape)
    bn = np.broadcast_to(bn, th_shape)
    zn = np.broadcast_to(zn, th_shape)
    bessel_deriv = np.broadcast_to(bessel_deriv, th_shape)  

    # if exact Mie solutions are wanted (including the near field effects given
    # by the spherical Hankel terms). The near fields don't change the total 
    # cross section much, but the angle-dependence of the differential cross
    # section will be very different from the ones obtained with the far-field 
    # approximations. 
    # Note that these solutions are not currently used anywhere in mie.py. 
    # They have been removed from the intensity calculations because it is 
    # not necessary to include the full fields in intensity calculations for 
    # far_field, and the exponential term (which should cancel out in intensity
    # calculation) can introduce rounding errors. We leave the expressions here
    # in case users ever have a need to know the actual fields, rather than 
    # the intensities.
    if near_field == True:     
        Es_theta = np.sum(En* (1j * an * taus * bessel_deriv/kd - bn * pis * zn), 
                          axis=-1) 
        Es_phi = np.sum(En* (-1j * an * pis * bessel_deriv/kd + bn * taus * zn), 
                        axis=-1)   
        Hs_phi = np.sum(En* (1j * bn * pis * bessel_deriv/kd - an * taus * zn), 
                        axis=-1)
        Hs_theta = np.sum(En* (1j * bn * taus * bessel_deriv/kd - an * pis * zn), 
                          axis=-1) 
    # if the near field effects aren't desired, use the asymptotic form of the 
    # spherical Hankel function in the far field (p. 94 of Bohren and Huffman).
    else:
        Es_theta = np.sum((2*n+1) / (n*(n+1)) * (an * taus + bn * pis), 
                          axis=-1)* np.exp(1j*kd)/(-1j*kd)
        Es_phi = -np.sum((2*n+1) / (n*(n+1)) * (an * pis + bn * taus), 
                         axis=-1)* np.exp(1j*kd)/(-1j*kd)  
        Hs_phi = np.sum((2*n+1) / (n*(n+1))*(bn * pis + an * taus), 
                        axis=-1)* np.exp(1j*kd)/(-1j*kd)
        Hs_theta = np.sum((2*n+1) / (n*(n+1))* (bn *  taus + an * pis), 
                          axis=-1)* np.exp(1j*kd)/(-1j*kd) 
                      
    return Es_theta, Es_phi, Hs_theta, Hs_phi
    
def diff_scat_intensity_complex_medium(m, x, thetas, kd, 
        coordinate_system = 'scattering plane', phis = None, near_field=False,
        incident_vector=None):
    '''
    Calculates the differential scattered intensity. These solutions are valid 
    both in the near and far field. When the medium has a zero imaginary component
    of the refractive index (is non-absorbing), the exact solutions at the far 
    field match the standard far-field Mie solutions given by calc_cross_sections.
    
    When coordinate_system == 'scattering plane':. 
       The solutions are given as a function of scattering angle theta. 
       
       The differential scattered intensity is computed by substituting the
       scattered electric and magnetic fields into the radial component of the 
       Poynting vector:
           
            I_par = Es_theta * conj(Hs_phi) 
            I_perp = Es_phi * conj(Hs_theta)
    
        where conj() indicates the complex conjugate. The radial component of the 
        Poynting vector is then 1/2 * Re(I_par - I_perp). 
    
    When coordinate_system == 'cartesian':
        The solutions are given as a function of scattering angle theta and 
        azimuthal angle phi
      
        The differential scattered intensity is computing by substituting the
        scattered electric and magnetic fields into the z-component of the Poynting
        vector:
            
            I_x = Es_x * conj(Hs_x) 
            I_y = -Es_y * conj(Hs_y)
    
        where conj() indicates the complex conjugate. The radial component of the 
        Poynting vector is then 1/2 * Re(I_x - I_y). 

    Parameters
    ----------
    m: complex relative refractive index 
    x: size parameter using the medium's refractive index 
    thetas: array of scattering angles (Quantity in rad)
    kd: k * distance, where k = 2*np.pi*n_matrix/wavelen, and distance is the
        distance away from the center of the particle. The far-field solution
        is obtained when distance >> radius. (Quantity, dimensionless)
    coordinate_system: string
        default value 'scattering plane' means scattering calculations will be 
        carried out in the basis defined by basis vectors parallel and 
        perpendicular to scattering plane. Variable also accepts value 
        'cartesian' which scattering calculations will be carried out in the 
        basis defined by basis vectors x and y in the lab frame, with z 
        as the direction of propagation.
    phis: None or ndarray
        azimuthal angles for which to calculate the diff scat intensity. In the
        'scattering plane' coordinate system, the scattering matrix does not
        depend on phi, so phi should be set to None. In the 'cartesian' 
        coordinate system, the scattering matrix does depend on phi, so an 
        array of values should be provided.
    near_field: boolean
        Set to True to include the near-fields. Cannot be set to True
        while using coordinate_system='cartesian' because near field solutions 
        are not implemented for cartesian coordinate system. Also cannot be set
        to True if using an incident_vector that is not None (unpolarized for
        'scattering plane' coordinate system). Often, the exact solutions 
        that include the near fields aren't wanted, for ex when the total cross 
        section calculation includes the structure factor, and the combination 
        of the angle-dependent differential cross section multiplied by the 
        structure factor gives very high cross sections at the surface of the 
        particle. When we want to neglect the effect of the near fields and 
        still integrate at the surface of the particle, we use the asymptotic  
        form of the spherical Hankel function in the far field (p. 94 of Bohren  
        and Huffman).
    incident_vector: None or tuple
        vector describing the incident electric field. It is multiplied by the 
        amplitude scattering matrix to find the vector scattering amplitude. If
        coordinate_system is 'scattering plane', then this vector should be in 
        the 'scattering plane' basis, where the first element is the parallel 
        component and the second element is the perpendicular component. If 
        coordinate_system is 'cartesian', then this vector should be in the 
        'cartesian' basis, where the first element is the x-component and the 
        second element is the y-component. Note that the vector for unpolarized
        light is the same in either basis, since either way it should be an 
        equal mix between the two othogonal polarizations: (1,1). Note that if 
        indicent_vector is None, the function assigns a value based on the 
        coordinate system. For 'scattering plane', the assigned value is (1,1) 
        because most scattering plane calculations we're interested in involve 
        unpolarized light. For 'cartesian', the assigned value is (1,0) because
        if we are going to the trouble to use the cartesian coordinate system, 
        it is usually because we want to do calculations using polarization, 
        and these calculations are much easier to convert to measured 
        quantities when in the cartesian coordinate system.
    
    Returns
    -------
    I components: tuple
        tuple of the two orthogonal components of scattered intensity. If in
        cartesian coordinate system, each component is a function of theta and
        phi values. If in scattering plane coordinate system, each component 
        is an array of theta values (dimensionless).These intensities are 
        technically 'unitless.' The intensities would get their units from 
        the E_n term in the fields, which gets its units from an E_0 term,
        which is taken to be 1 here. To get an intensity with real units
        you would need to multiply these by |E_0|**2 where E_0 is the amplitude
        of the incident wave at the origin. 
        
    References
    ----------
    C. F. Bohren and D. R. Huffman. Absorption and scattering of light by 
    small particles. Wiley-VCH (2004), chapter 4.4.1.
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).
    
    '''
    if near_field==True:
        if coordinate_system == 'scattering plane':
            # calculate scattered fields in scattering plane coordinate system
            Es_theta, Es_phi, Hs_theta, Hs_phi = _scat_fields_complex_medium(m,
                                        x,thetas, kd, near_field=near_field)
            I_1 = Es_theta * np.conj(Hs_phi) # I_par
            I_2 = -Es_phi * np.conj(Hs_theta) # I_perp
        else:
            raise ValueError('Near fields have not been implemented for the \
                specified coordinate system. set near_field to False to\
                calculate scattered intensity')
    else:
    # If near field is not true, we don't actually need to calculate the fields
    # to find the intensities. The field equations contain an imaginary 
    # exponential term that cancels out when we multiply by the complex conjugate
    # in Es_theta * np.conj(Hs_phi) to find the intensity. Testing has shown 
    # that this exponential term can introduce rounding errors, particularly
    # when the diff scat intensity is integrated to find a total cross section. 
    # These rounding errors can be avoided by simply calculating the intensities
    # using the vector scattering amplitude. These two methods are mathematically
    # exactly the same. The difference is that the vector_scattering_amplitude
    # does not include the exponential terms, so there is no rounding error. 
        
        # calculate vector scattering amplitude
        vec_scat_amp_1, vec_scat_amp_2 = vector_scattering_amplitude(m, x, 
                                            thetas,
                                            coordinate_system=coordinate_system,
                                            phis=phis, 
                                            incident_vector = incident_vector)
        
        # calculate the intensities
        # note that we divide by a factor of kd because the intensities
        # calculated using the fields include this factor. 
        I_1 = np.abs(vec_scat_amp_1/kd)**2 # par or x
        I_2 = np.abs(vec_scat_amp_2/kd)**2 # perp or y

    return I_1.real, I_2.real # real or abs?

def integrate_intensity_complex_medium(I_1, I_2, distance, thetas, k,
                                       phi_min=Quantity(0, 'rad'), 
                                       phi_max=Quantity(2*np.pi, 'rad'),
                                       coordinate_system = 'scattering plane',
                                       phis = None):
    '''
    Calculates the scattering cross section by integrating the differential 
    scattered intensity obtained with the exact Mie solutions. The integration 
    is done over scattering angles theta and azimuthal angles phi using the
    trapezoid rule.  
    
    Parameters
    ----------
    I_1, I_2: nd arrays 
        differential scattered intensities, can be functions of theta or of
        theta and phi. If a function of theta and phi, the theta dimension MUST
        come first
    distance: float (Quantity in [length])
        distance away from the scatterer 
    thetas: nd array (Quantity in rad)
        scattering angles
    k: wavevector given by 2 * pi * n_medium / wavelength 
       (Quantity in [1/length])
    phi_min: float (Quantity in rad).
        minimum azimuthal angle, default set to 0 
        optional, only necessary if coordinate_system is 'scattering plane'
    phi_max: float (Quantity in rad).
        maximum azimuthal angle, default set to 2pi 
        optional, only necessary if coordinate_system is 'scattering plane'
    phis: None or ndarray
        azimuthal angles
    
    Returns
    -------
    sigma: float (in units of length**2)
        integrated cross section 
    sigma_1: float (in units of length**2)
        integrated cross section for first component of basis
    sigma_2: float (in units of length**2)
        integrated cross section for second component of basis
    dsigma_1: ndarray (in units of length**2)
        differential cross section for first component of basis
    dsigma_2: ndarray (in units of length**2)
        differential cross section for second component of basis

    '''
    # convert to radians from whatever units the user specifies
    if isinstance(thetas, Quantity):
        thetas = thetas.to('rad').magnitude
    
    # this line converts the unitless intensities to cross section
    dsigma_1 = I_1 * distance**2  
    dsigma_2 = I_2 * distance**2 
    
    if coordinate_system == 'scattering plane':
        if phis != None:
            warnings.warn('''azimuthal angles specified for scattering plane
                          calculations. Scattering plane calculations do not
                          depend on azimuthal angle, so specified values will
                          be ignored''')

        # convert to radians
        phi_min = phi_min.to('rad').magnitude
        phi_max = phi_max.to('rad').magnitude
                      
        # Integrate over theta  
        integrand_par = np.trapz(dsigma_1 * np.abs(np.sin(thetas)), 
                                 x=thetas) * dsigma_1.units
                                 
        integrand_perp = np.trapz(dsigma_2 * np.abs(np.sin(thetas)), 
                                  x=thetas) * dsigma_2.units
     
        # integrate over phi: multiply by factor to integrate over phi
        # (this factor is the integral of cos(phi)**2 and sin(phi)**2 in parallel 
        # and perpendicular polarizations, respectively) 
        sigma_1 = (integrand_par * (phi_max/2 + np.sin(2*phi_max)/4 - 
                         phi_min/2 - np.sin(2*phi_min)/4))    
        sigma_2 = (integrand_perp * (phi_max/2 - np.sin(2*phi_max)/4 - 
                          phi_min/2 + np.sin(2*phi_min)/4))
        
    elif coordinate_system == 'cartesian':
        if phis is None:
            raise ValueError('phis set to None, but azimuthal angle must be \
                        specified for scattering calculations in \
                        cartesian coordinate system')
            
        # convert to radians
        if isinstance(phis, Quantity):
            phis = phis.to('rad').magnitude
            
        # Integrate over theta and phi
        thetas_bc = thetas.reshape((len(thetas),1)) # reshape for broadcasting
        
        sigma_1 = np.trapz(np.trapz(dsigma_1 * np.abs(np.sin(thetas_bc)), x=thetas, axis=0),
                               x=phis) * dsigma_1.units
        sigma_2 = np.trapz(np.trapz(dsigma_2 * np.abs(np.sin(thetas_bc)), x=thetas, axis=0),
                               x=phis) * dsigma_2.units
        
    else:
        raise ValueError('The coordinate system specified has not yet been \
                implemented. Change to \'cartesian\' or \'scattering plane\'')
                          
    # multiply by factor that accounts for attenuation in the incident light
    # (see Sudiarta and Chylek (2001), eq 10). 
    # if the imaginary part of k is close to 0 (because the medium index is 
    # close to 0), then use the limit value of factor for the calculations
    if k.imag.magnitude <= 1e-8:
        factor = 2
       
    else:                  
        exponent = np.exp(2*distance*k.imag)
        factor = 1 / (exponent / (2*distance*k.imag)+
                     (1 - exponent) / (2*distance*k.imag)**2)
    
    # calculate the averaged sigma
    sigma = (sigma_1 + sigma_2)/2 * factor

    return(sigma, sigma_1*factor, sigma_2*factor, dsigma_1*factor/2, 
           dsigma_2*factor/2)

def diff_abs_intensity_complex_medium(m, x, thetas, ktd):
    '''   
    Calculates the differential absorbed intensity as a function of scattering
    angle theta when the medium has a non-zero imaginary component of the 
    refractive index. This differential absorbed intensity is computed by 
    substituting the internal electric and magnetic fields (from Fu and Sun)
    into the radial component of the Poynting vector:
    
    I_par = -Et_theta * conj(Ht_phi) 
    I_perp = Et_phi * conj(Ht_theta)
    
    where conj() indicates the complex conjugate. The radial component of the 
    Poynting vector is then 1/2 * Re(I_par + I_perp). 
    
    Parameters
    ----------
    m: complex relative refractive index 
    x: size parameter using the medium's refractive index 
    thetas: array of scattering angles (Quantity in rad)
    ktd: kt * distance, where kt = 2*np.pi*n_particle/wavelen, and distance is 
        the distance away from the center of the particle. The far-field 
        solution is obtained when distance >> radius. (Quantity, dimensionless)
    
    Returns
    -------
    I_par, I_perp: differential absorption intensities for an array of theta
                   (dimensionless).
    
    Reference
    ---------
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).
    
    '''
    # convert units from whatever units the user specifies
    thetas = thetas.to('rad').magnitude
    ktd = ktd.to('').magnitude
    
    # calculate mie coefficients
    nstop = _nstop(np.array(x).max())  
    n = np.arange(nstop)+1.
    cn, dn = _internal_coeffs(m, x, nstop)
    
    # calculate prefactor (omitting the incident electric field because it
    # cancels out when calculating the scattered intensity)
    En = 1j**n * (2*n+1) / (n*(n+1))

    # calculate spherical Bessel function and derivative
    nstop_array = np.arange(0,nstop+1)
    zn = spherical_jn(nstop_array, ktd)
    zn = zn[1:]
    
    psi, _ = mie_specfuncs.riccati_psi_xi(ktd, nstop)
    psishift = np.concatenate((np.zeros(1), psi))[0:nstop+1]
    psi = psi[1:]
    psishift = psishift[1:]
    bessel_deriv = psishift - n*psi/ktd
    
    # calculate pis and taus at the scattering angles theta
    pis, taus = _pis_and_taus(nstop, thetas)
    
    # calculate the scattered electric and magnetic fields (omitting the 
    # sin(phi) and cos(phi) factors because they will be accounted for when
    # integrating to get the scattering cross section)
    En = np.broadcast_to(En, [len(thetas), len(En)])
    cn = np.broadcast_to(cn, [len(thetas), len(cn)])
    dn = np.broadcast_to(dn, [len(thetas), len(dn)])
    zn = np.broadcast_to(zn, [len(thetas), len(zn)]) 
    bessel_deriv = np.broadcast_to(bessel_deriv,[len(thetas),len(bessel_deriv)])                  
    
    Et_theta = np.sum(En* (cn * pis * zn - 1j * dn * taus * bessel_deriv/ktd), 
                      axis=1)  # * cos(phi)
    Et_phi = np.sum(En* (-cn * taus * zn + 1j * dn * pis * bessel_deriv/ktd), 
                    axis=1) # * sin(phi)      
    Ht_theta = np.sum(En* (dn * pis * zn - 1j * cn * taus * bessel_deriv/ktd), 
                      axis=1) # * sin(phi)
    Ht_phi = np.sum(En* (dn * taus * zn - 1j * cn * pis * bessel_deriv/ktd), 
                    axis=1) # * cos(phi)
    
    # calculate the scattered intensities 
    I_par = -m* Et_theta * np.conj(Ht_phi)
    I_perp = m* Et_phi * np.conj(Ht_theta)
    
    return I_par.real, I_perp.real

def amplitude_scattering_matrix(m, x, thetas, coordinate_system = 'scattering plane', 
                        phis = None):
    """
    Calculates the amplitude scattering matrix for an n-dim array of thetas 
    (and phis if in cartesian coordinate system)
    
    Elements of the amplitude scattering matrix are arranged as:
    [S2  S3]
    [S4  S1]    
    
    Change of basis from scattering plane to lab frame cartesian is calculated 
    by multiplying M*S*(M^-1) where S is the amplitude scattering matrix and 
    M is the change of basis matrix. The change of basis matrix M is:
        
    [cosphi  sinphi]
    [sinphi -cosphi]
    
    (from Bohren and Huffman, 3.2, page 61)
    
    This matrix is equal to it's inverse, so we can get the scattering matrix 
    in the cartesian coordinate system by multiplying:
    
    [cosphi  sinphi] * [S2  S3] * [cosphi  sinphi]
    [sinphi -cosphi]   [S4  S1]   [sinphi -cosphi]
    
    in Mie theory, we have S3 = S4 = 0, so this simplifies to:
        
    =   [cosphi  sinphi] * [S2   0] * [cosphi  sinphi]
        [sinphi -cosphi]   [0   S1]   [sinphi -cosphi]
    
    =   [cosphi  sinphi] * [S2*cosphi  S2*sinphi]
        [sinphi -cosphi]   [S1*sinphi -S1*cosphi]
        
    =   [S2*cosphi**2 + S1*sinphi**2       S2*sinphi*cosphi - S1*sinphi*cosphi]
        [S2*cosphi*sinphi-S1*cosphi*sinphi         S2*sinphi**2 + S1*cosphi**2]
     
    see pages 22-23,51-53 in Annie Stephenson lab notebook #3 for orignal notes 
    
    Parameters:
    ----------
    m: float, or array 
        index ratio between the particle and sample, array if multilayer particle  
    x: float, or array
        size parameter, array if multilayer particle  
    thetas: nd array
        theta angles 
    coordinate_system: string
        default value 'scattering plane' means scattering calculations will be 
        carried out in the basis defined by basis vectors parallel and 
        perpendicular to scattering plane. Variable also accepts value 
        'cartesian' which scattering calculations will be carried out in the 
        basis defined by basis vectors x and y in the lab frame, with z 
        as the direction of propagation.
    phis: None or ndarray
        azimuthal angles for which to calculate the scattering matrix. In the
        'scattering plane' coordinate system, the scattering matrix does not
        depend on phi, so phi should be set to None. In the 'cartesian' 
        coordinate system, the scattering matrix does depend on phi, so an 
        array of values should be provided.

    Returns:
    --------
    S1, S2, S3, S4: tuple of nd arrays 
       amplitude scattering matrix elements for all theta. S2 and S1 have the 
       same shape as theta. 
    """
    # calculate n-array 
    nstop = _nstop(np.array(x).max())
    n = np.arange(nstop)+1.
    prefactor  = (2*n+1)/(n*(n+1))
    
    # calculate mie coefficients
    # if the index ratio m is an array with more than 1 element, it's a 
    # multilayer particle
    if len(np.atleast_1d(m)) > 1:
        coeffs = msl.scatcoeffs_multi(m, x)
    else:
        coeffs = _scatcoeffs(m, x, nstop)
    
    # calculate amplitude scattering matrix in 'scattering plane' coordinate system 
    S2_sp, S1_sp = _amplitude_scattering_matrix(nstop, prefactor, coeffs, thetas)
    S3_sp = 0
    S4_sp = 0
    
    if coordinate_system == 'cartesian':
        # raise error if no phis are specified
        if phis is None:
            raise ValueError('phis set to None, but azimuthal angle must be \
                             specified for scattering calculations in \
                             cartesian coordinate system')
        
        # calculate sines and cosines
        cosphi = np.cos(phis)
        sinphi = np.sin(phis)
        
        # calculate elements of scattering matrix
        S1_xy = S2_sp*(sinphi)**2 + S1_sp*(cosphi)**2
        S2_xy = S2_sp*(cosphi)**2 + S1_sp*(sinphi)**2 
        S3_xy = S2_sp*sinphi*cosphi - S1_sp*sinphi*cosphi
        S4_xy = S2_sp*cosphi*sinphi - S1_sp*cosphi*sinphi

        return S1_xy, S2_xy, S3_xy, S4_xy
    elif coordinate_system == 'scattering plane':
        if phis != None:
            warnings.warn('azimuthal angles specified for scattering plane \
                          calculations. Scattering plane calculations do not \
                          depend on azimuthal angle, so specified values will \
                          be ignored')
        return S1_sp, S2_sp, S3_sp, S4_sp
    else:
        raise ValueError('The coordinate system specified has not yet been \
                implemented. Change to \'cartesian\' or \'scattering plane\'')

    
def vector_scattering_amplitude(m, x, thetas, incident_vector = None, coordinate_system = 'scattering plane', 
                    phis = None):
    '''
    Calculates the vector scattering amplitude  for an nd array of thetas and 
    phis. For more info on the vector scattering amplitude and how to 
    calculate it, see Bohren and Huffman, pg 70-73 of section 3.4 Extinction, 
    Scattering, and Absorption.
    
    When coordinate_system == 'scattering plane', the default incident 
    electric field vector assumes that the incident light is unpolarized, 
    so it is equally split between the parallel and perpendicular components 
    of the electric field, so the vector scattering amplitude can be 
    calculated by:
        
            [S2   0] * [1]  = [S2]
            [0   S1]   [1]    [S1]
            
    where the vector is normalized after the multiplication
    
    When coordinate_system == 'cartesian', if the incident electric field 
    vector indicates the incident light is unpolarized, it is equally 
    split between the x and y components of the electric field, so the vector
    scattering amplitude can be calculated by:
        
        [S2 S3] * [1] = [S2 + S3]
        [S4 S1]   [1]   [S4 + S1]
        
    however, in most cases, if we are in the 'cartesian' coordinate system,
    we are more interested in calculating the the scattered light given 
    and initial polarization of the initial light, since this calculation cannot
    be done in the 'scattering plane' coordinate system. If we assume the initial
    polarization is in the +x direction, the vector scattering amplitude will
    be:
    
        [S2 S3] * [1] = [S2]
        [S4 S1]   [0] = [S4] 
        
    where the vector is normalized after the multiplication. The default value
    is then +x when set to incident_vector is set to None. 
    
    Parameters:
    ----------
    m: float
        index ratio between the particle and sample  
    x: float
        size parameter  
    thetas: nd array
        scattering angles
    incident_vector: None or tuple
        vector describing the incident electric field. It is multiplied by the 
        amplitude scattering matrix to find the vector scattering amplitude. If
        coordinate_system is 'scattering plane', then this vector should be in 
        the 'scattering plane' basis, where the first element is the parallel 
        component and the second element is the perpendicular component. If 
        coordinate_system is 'cartesian', then this vector should be in the 
        'cartesian' basis, where the first element is the x-component and the 
        second element is the y-component. Note that the vector for unpolarized
        light is the same in either basis, since either way it should be an 
        equal mix between the two othogonal polarizations: (1,1). Note that if 
        incident_vector is None, the function assigns a value based on the 
        coordinate system. For 'scattering plane', the assigned value is (1,1) 
        because most scattering plane calculations we're interested in involve 
        unpolarized light. For 'cartesian', the assigned value is (1,0) because
        if we are going to the trouble to use the cartesian coordinate system, 
        it is usually because we want to do calculations using polarization, 
        and these calculations are much easier to convert to measured 
        quantities when in the cartesian coordinate system.
    coordinate_system: string
        describes the coordinate system. Can be either 'scattering plane' or
        'cartesian'
    phis: nd array or None
        azimuthal angles
        
        
    Returns:
    --------
    vector scattering amplitude: tuple
        tuple describing the vector scattering amplitude in the specified 
        coordinate system. not normalized. 
    '''
    # calculate the amplitude scattering matrix
    S1, S2, S3, S4 = amplitude_scattering_matrix(m, x, thetas, 
                                                 coordinate_system = coordinate_system, 
                                                 phis = phis)
    
    if coordinate_system == 'scattering plane':
        if incident_vector is None:
            incident_vector = (1,1) # assume unpolarized
        vec_scat_amp_par = S2*incident_vector[0]
        vec_scat_amp_perp = S1*incident_vector[1]
        
        return vec_scat_amp_par, vec_scat_amp_perp
    
    else:
        if incident_vector is None:
            incident_vector = (1,0) # assume x-polarized
        vec_scat_amp_x = S2*incident_vector[0] + S3*incident_vector[1]
        vec_scat_amp_y = S4*incident_vector[0] + S1*incident_vector[1]
        
        return vec_scat_amp_x, vec_scat_amp_y
        

def _amplitude_scattering_matrix(n_stop, prefactor, coeffs, thetas):
    # amplitude scattering matrix from Mie coefficients
    pis, taus = _pis_and_taus(n_stop, thetas)
    S1 = np.sum(prefactor*(coeffs[0]*pis + coeffs[1]*taus), axis=-1)
    S2 = np.sum(prefactor*(coeffs[0]*taus + coeffs[1]*pis), axis=-1)
    return S2, S1

def _amplitude_scattering_matrix_RG(prefactor, x, thetas):
    # amplitude scattering matrix from Rayleigh-Gans approximation
    u = 2 * x * np.sin(thetas/2.)
    S1 = prefactor * (3./u**3) * (np.sin(u) - u*np.cos(u))
    S2 = prefactor * (3./u**3) * (np.sin(u) - u*np.cos(u)) * np.cos(thetas)
    return S2, S1



# TODO: copy multilayer code from multilayer_sphere_lib.py in holopy and
# integrate with functions for calculating scattering cross sections and form
# factor.
