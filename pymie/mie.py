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
Based on miescatlib.py in HoloPy, written by Jerome Fung. Also includes some
functions from Jerome's old miescat_1d.py library and Jerome's multilayer
scattering code, copied from HoloPy on 12 Sept 2017.

Numerical stability not guaranteed for large nstop, so be careful when
calculating very large size parameters. A better-tested (and faster) version of
this code is in the HoloPy package (http://manoharan.seas.harvard.edu/holopy).

Key reference for multilayer algorithm is [3]_

Some notes on array dimensions and broadcasting:

-   Nearly all calculations are vectorized.  Let "..." denote the dimensions
    corresponding to parameters of the calculations.  For example, we might be
    interested in calculating scattering quantities as a function of wavelength
    and volume fraction (volume fraction can enter into a calculation
    indirectly through an effective refractive index).  Then "..." might
    represent the volume fraction and wavelength dimensions.

-   m and x must be specified as at least 2D arrays.  The shapes are
        m: (..., num_layers)
        x: (..., num_layers)
    where num_layers is the number of layers of a multilayer particle, and
    "..." means at least one other dimension (wavelength). If specified with
    shape (1, 1), the calculation will be done at a single, scalar value of m
    and of x.  This situation would correspond to a single wavelength and a
    single, non-layered sphere.

-   Angles (thetas and phis) must be specified with shape
        thetas: ([angle_leading_dims], num_thetas)
        phis:   ([angle_leading_dims], num_phis)
    where [angle_leading_dims] is a subset of the leading dimensions of m/x.

-   For example, if m has shape
        m:      (num_volume_fractions, num_wavelengths, num_layers),
    then thetas can have shape
        thetas: (num_thetas) or
        thetas: (num_wavelengths, num_thetas) or
        thetas: (num_volume_fractions, num_wavelengths, num_thetas).
    The same is true of phi, but with num_phis instead of num_thetas.
    Therefore, the angles can be a function of wavelength, for example.  As
    long as angle_leading_dims are specified in the same order as in "...",
    thetas and phis will broadcast correctly, once the dimensions of other
    quantities have been expanded (see below)

-   Intermediate calculations: quantities to be used in calculations with
    angles have their dimensions expanded to include theta (and phis).  For
    example, we might expand x to the following:
        x:      (..., 1)
    to broadcast with the thetas array.  If phis is specified, we expand x to
        x:      (..., 1, 1)
    Other intermediate arrays are expanded to the same dimensions for
    broadcasting -- but not necessarily the same shapes, because we want to
    conserve memory.

-   Intermediate calculations: calculations that involve summing over a series
    will have an "order" dimensions, corresponding to the order of the
    coefficients used in the series. We add this order dimension to the end of
    the array, since we want to remove it after we sum over it.  In this case,
    we might expand x as follows:
        x:      (..., n_max)
    where n_max is the maximum order.  If the calculation also involves the
    angle theta, x would be expanded as
        x:      (..., 1, n_max)
    and if phis are involved as well,
        x:      (..., 1, 1, n_max)

-   Broadcasting: All functions broadcast over the leading dimensions of m and
    x (the "..." above). Thus, outputs will contain the same dimensions as in
    "..." and possibly others that have been added (like polarization). A
    calculation done for a single value of m and of x -- shape (1, 1) for both
    -- will retain the singlet dimension corresponding to the first 1. This
    dimension can later be removed by ".squeeze()" but we do not squeeze by
    default because doing so would lead to inconsistent numbers of dimensions
    between, for example, single-wavelength and multi-wavelength calculations.

-   Output shapes are as follows.  [] indicates optional
        scat. coefficients:    (2, ..., n_max)
        diff. cross-secs:      (num_polarizations, ..., num_thetas, [num_phis])
        integrated cross-secs: (...,)
        scat. matrix elements: (..., num_thetas)

-   Outputs generally do not contain a layers dimension because most
    calculations report scattering quantities for the entire sphere, not
    individual layers within it.

-   The broadcasting approach has its limitations, primarily because we choose
    the maximum order (nstop) to calculate the Mie coefficients based on the
    largest value of x in the array of size parameters. Therefore, in
    calculations over a wide range of size parameters, the coefficients
    corresponding to every size parameter are expanded to the same (large)
    order -- which is not only inefficient, but can also lead to numerical
    instabilities for small x's in the size parameter array. One way around
    this limitation would be to use ragged arrays when expanding x to (...,
    n_max), but doing this efficiently would require installing an extension to
    numpy like Awkward Array.

References
----------
[1] Bohren, C. F. and Huffman, D. R. "Absorption and Scattering of Light by
Small Particles" (1983)
[2] Wiscombe, W. J. "Improved Mie Scattering Algorithms" Applied Optics 19, no.
9 (1980): 1505. doi:10.1364/AO.19.001505
[3] Yang, "Improved recursive algorithm for light scattering by a multilayered
sphere," Applied Optics 42, 1710-1720 (2003).

.. moduleauthor:: Jerome Fung <jerome.fung@gmail.com>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Sofia Magkiriadou <sofia@physics.harvard.edu>
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.special import legendre_p_all

from . import Quantity, index_ratio, mie_specfuncs
from . import size_parameter, ureg
from .mie_specfuncs import DEFAULT_EPS1, DEFAULT_EPS2  # default tolerances

# User-facing functions for the most often calculated quantities (form factor,
# efficiencies, asymmetry parameter)

def calc_ang_scat(m, x, thetas, kd=None, phis=None, incident_vector=None,
                  check=False):
    """
    Calculates the angular scattering of light intensity for parallel and
    perpendicular polarization for a sphere.

    Parameters
    ----------
    m : complex or float, array-like
        complex particle relative refractive index, n_part/n_med
    x : complex or float, array-like
        size parameter, x = ka = 2*pi*n_med/lambda * a (sphere radius a)
    thetas : array-like
        array of angles. Must be specified in radians
    kd : float
        k * distance, where k = 2*np.pi*n_matrix/wavelen, and distance is the
        distance away from the center of the particle. The standard far-field
        solutions are obtained when distance >> radius in a non-absorbing
        medium.
    phis : None or ndarray
        azimuthal angles for which to calculate the diff scat intensity. If
        set, a cartesian basis is used
    incident_vector : 2-tuple (default None)
        If supplied, gives the polarization direction in the appropriate basis
        (cartesian if phis is set or scattering plane if not)
    check : Boolean (optional)
        if true, outputs scattering efficiencies

    Returns
    -------
    ndarray : shape (2, ..., num_angles)
        element 0 is ipar: |S_2|^2
        element 1 is iperp: |S_1|^2
        These are the differential scattering cross-sections * k^2 for
        polarization parallel and perpendicular to scattering plane. See Bohren
        & Huffman ch. 3 for details.
    """
    if (kd is not None) or (phis is not None):
        return diff_scat_intensity_complex_medium(m, x, thetas, kd=kd,
                                                  phis=phis,
                                                  incident_vector =
                                                  incident_vector)

    # Mie scattering preliminaries
    nstop = _nstop(x.max())

    coeffs = _scatcoeffs(m, x, nstop)
    n = np.arange(nstop)+1.
    prefactor = (2*n+1.)/(n*(n+1.))

    S2, S1 = _amplitude_scattering_matrix(nstop, prefactor, coeffs, thetas)
    ipar = np.absolute(S2)**2
    iperp = np.absolute(S1)**2

    if check:
        opt = _amplitude_scattering_matrix(nstop, prefactor,
                                           coeffs, 0).real
        qscat, qext, qback = calc_efficiencies(m, x)
        print('Number of terms:')
        print(nstop)
        print('Scattering, extinction, and backscattering efficiencies:')
        print(qscat, qext, qback)
        print('Extinction efficiency from optical theorem:')
        print((4./x**2)*opt)
        print('Asymmetry parameter')
        print(calc_g(m, x))

    if incident_vector is not None:
        ipar = ipar * incident_vector[0]
        iperp = iperp * incident_vector[1]

    return np.array([ipar, iperp])


def calc_ang_scat_RG(m, x, angles):
    """
    Uses the Rayleigh-Gans approximation to calculates the angular scattering
    of light intensity for parallel and perpendicular polarization for a
    sphere.

    Parameters
    ----------
    m : complex or float, array-like
        complex particle relative refractive index, n_part/n_med
    x : complex or float, array-like
        size parameter, x = ka = 2*pi*n_med/lambda * a (sphere radius a)
    angles : array-like
        array of angles. Must be specified in radians

    Returns
    -------
    ndarray : shape (2, ..., num_angles)
        Differential scattering cross-sections * k^2 for
        polarization parallel and perpendicular to scattering plane, under the
        Rayleigh-Gans approximation.
    """
    prefactor = -1j * (2./3.) * x**3 * np.absolute(m - 1)
    S2, S1 = _amplitude_scattering_matrix_RG(prefactor, x, angles)
    ipar = np.absolute(S2)**2
    iperp = np.absolute(S1)**2

    return np.array([ipar, iperp])


def calc_cross_sections(m, x, eps1 = DEFAULT_EPS1,
                        eps2 = DEFAULT_EPS2):
    """
    Calculate dimensionaless scattering, absorption, and extinction cross
    sections, and asymmetry parameter for spherically symmetric scatterers.

    Parameters
    ----------
    m : array-like
        complex relative refractive index
    x : array-like
        size parameter

    Returns
    -------
    cross_sections : tuple (5)
        Dimensionless scattering, absorption, extinction, and backscattering
        cross sections, and <cos theta> (asymmetry parameter g)

    Notes
    -----
    To recover the dimensional cross-sections, multiply the returned
    cross-sections by 1/k^2, where k is the wavevector in *media*. The
    asymmetry parameter is dimensionless and needs no scaling.

    The backscattering cross-section is 1/(4*pi) times the radar backscattering
    cross-section; that is, it corresponds to the differential scattering
    cross-section in the backscattering direction.  See B&H 4.6.

    The radiation pressure cross section C_pr is given by
    C_pr = C_ext - <cos theta> C_sca.

    The radiation pressure force on a sphere is

    F = (n_med I_0 C_pr) / c

    where I_0 is the incident intensity.  See van de Hulst, p. 14.
    """
    lmax = _nstop(x.max())
    albl = _scatcoeffs(m, x, lmax, eps1=eps1, eps2=eps2)

    cscat, cext, cback =  tuple(c*2*np.pi for c in
                                _cross_sections(albl[0], albl[1]))

    cabs = cext - cscat # conservation of energy

    # _asymmetry_parameter returns g*Q_sca*x^2/4.  cscat is k^2 times the
    # dimensional cross-section.  So asymmetry parameter is
    # 4*pi/(k^2 Csca) * sum from _asymmetry_parameter, where Csca is the
    # dimensional cross section
    asym = 4*np.pi / cscat * _asymmetry_parameter(albl[0], albl[1])

    return cscat, cext, cabs, cback, asym


def calc_efficiencies(m, x):
    """
    Scattering, extinction, backscattering efficiencies

    Note that the backscattering efficiency is 1/(4*pi) times the radar
    backscattering efficiency; that is, it corresponds to the differential
    scattering cross-section in the backscattering direction, divided by the
    geometrical cross-section
    """
    nstop = _nstop(x.max())
    coeffs = _scatcoeffs(m, x, nstop)

    cscat, cext, cback = _cross_sections(coeffs[0], coeffs[1])

    # for multilayer spheres, scale by the size parameter corresponding to
    # outermost radius
    x_outer = x.max(axis=-1)
    qscat = cscat * 2./np.abs(x_outer)**2
    qext = cext * 2./np.abs(x_outer)**2
    qback = cback * 1./np.abs(x_outer)**2

    # in order: scattering, extinction and backscattering efficiency
    return qscat, qext, qback


def calc_g(m, x, nstop=None):
    """
    Asymmetry parameter
    """
    if nstop is None:
        nstop = _nstop(x.max())
    coeffs = _scatcoeffs(m, x, nstop)

    # for multilayer particle, need to scale by the x of the outermost layer
    outer_x = x.max(axis=-1)
    cscat = _cross_sections(coeffs[0], coeffs[1])[0] * 2./outer_x**2
    g = ((4./(outer_x**2 * cscat))
         * _asymmetry_parameter(coeffs[0], coeffs[1]))
    return g


def calc_integrated_cross_section(m, x, thetas):
    """
    Calculate (dimensionless) integrated cross section using quadrature

    Parameters
    ----------
    m : array-like
        complex relative refractive index
    x : array-like
        size parameter
    thetas: array-like
        polar angles over which to integrate the scattering.  Must be specifed
        in radians

    Returns
    -------
    cross_section : ndarray
        Dimensionless integrated cross-section.  Multiply by 1/k^2 (where k is
        the wavevector *in media*) to get the dimensional cross-section.

    """
    form_factor = calc_ang_scat(m, x, thetas)

    integrand = form_factor * np.sin(thetas)
    integral = 2 * np.pi * np.trapezoid(integrand, x=thetas)

    # average over two polarizations
    return (integral.sum(axis=0))/2.0


def calc_energy(radius, n_medium, m, x, nstop,
           eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculates the electromagnetic energy inside a dielectric sphere
    according to equation 11 in
    Bott and Zdunkowski, J. Opt. Soc. Am. A, vol 4, no. 8, 1987

    Parameters
    ----------
    radius: float
        radius of the scatterer (Quantity in [length])
    n_medium: float
        refractive index of the medium in which scatterer is embedded
    m: float
        complex relative refractive index
    x: float
        size parameter
    nstop: float
        maximum order

    Returns
    -------
    W: float (Quantity in [energy])
        electromagnetic energy inside the dielectic sphere

    '''
    W0 = _W0(radius, n_medium)
    gamma_n, An = _time_coeffs(m, x, nstop, eps1 = eps1, eps2 = eps2)
    n = np.arange(1,nstop+1)
    y = m*x
    W = 3/4*W0*np.sum((2*n + 1)/y**2 *gamma_n*(1+An**2-n*(n+1)/y**2))

    return W


def calc_dwell_time(radius, n_medium, n_particle, wavelen,
                    min_angle=0.01, num_angles=200,
                    eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculates the dwell time, the time
    according to 3.37 in
    Lagendijk and van Tiggelen, Physics Reports 270 (1996) 143-215

    Parameters
    ----------
    radius: float
        radius of the scatterer (Quantity in [length])
    n_medium: float
        refractive index of the medium in which scatterer is embedded
    n_particle: float
        refractive index of the scatterer
    wavelen: structcol.Quantity [length]
        wavelength of incident light in vacuum
    min_angle: float (in radians)
        minimum angle to integrate over for total cross section
    num_angles: float
        number of angles to integrate over for total cross section
    eps1, eps2: needed for calculating scattcoeffs

    Returns
    -------
    dwell_time: float (Quantity in [to,e])
        time wave spends inside the dielectic sphere
    '''
    m = index_ratio(n_particle, n_medium)
    x = size_parameter(wavelen, n_medium, radius)
    nstop = _nstop(x)
    wavelen_media = wavelen/n_medium

    # calculate the energy contained in sphere
    W = calc_energy(radius, n_medium, m, x, nstop, eps1 = eps1, eps2 = eps2)

    # define speed of light
    # get this from Pint in a somewhat indirect way:
    c = Quantity(1.0, 'speed_of_light').to('m/s')

    # calculate total cross section
    k = 2*np.pi/wavelen_media
    if np.imag(x)>0:
        angles = np.linspace(min_angle, np.pi, num_angles)
        distance = radius.max()
        kd = (k*distance).to("").magnitude
        diff_cscat = diff_scat_intensity_complex_medium(m, x, angles, kd)
        cscat = integrate_intensity_complex_medium(diff_cscat, angles, kd)[0]
    else:
        cscat = calc_cross_sections(m, x, eps1 = eps1, eps2 = eps2)[0]
        cscat = cscat * 1/k**2

    # calculate dwell time
    dwell_time = W/(cscat*c)

    return dwell_time


# TODO: document and test the correctness of this function, or delete (it is
# not currently used anywhere)
@ureg.check('[length]', None, None, '[length]', None, None)
def calc_reflectance(radius, n_medium, n_particle, wavelen,
                     min_angle=np.pi/2, num_angles=50):

    m = index_ratio(n_particle, n_medium)
    x = size_parameter(wavelen, n_medium, radius)
    wavelen_media = wavelen/n_medium
    # geometric cross section is defined by outermost radius
    if np.isscalar(radius):
        rmax = radius
    else:
        rmax = radius.max()
    geometric_cross_sec = np.pi*rmax**2

    thetas = np.linspace(min_angle, np.pi, num_angles)
    # calculate reflectance cross section
    if np.any(np.imag(x) > 0):
        distance = rmax
        k = np.atleast_1d(2*np.pi/wavelen_media)
        kd = (k*distance).to("").magnitude
        diff_cscat = diff_scat_intensity_complex_medium(m, x, thetas,
                                                        kd)
        refl_cscat = integrate_intensity_complex_medium(diff_cscat, thetas,
                                                        kd)[0]
        refl_cscat = refl_cscat/k**2
    else:
        refl_cscat = calc_integrated_cross_section(m, x, thetas)
        refl_cscat = wavelen_media**2/4/np.pi/np.pi * refl_cscat

    reflectance = ((refl_cscat/geometric_cross_sec).to('')
                   / wavelen_media**2)

    return reflectance


# Mie functions used internally

def _pis_and_taus(nstop, thetas):
    '''
    Calculate pi and tau angular functions at an array of theta out to order n

    Parameters
    ----------
    nstop : float
        maximum order
    thetas : ndarray or float
        scattering angles.  Must be specified in radians.

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

    # flatten to make calculations easier
    mu = np.cos(thetas.ravel())

    # returns P_n and derivatives up to degree n for all values in mu array.
    # legendre0 has shape (2, nmax, num_thetas), where legendre0[0,:,:] is P_n
    # and legendre0[1,:,:] is the derivative.
    legendre0 = legendre_p_all(nstop, mu, diff_n=1)

    # Perform calculations on pis to get taus. We swap axes so that the order
    # axis is last; resulting shape is (num_thetas, nstop+1)
    pis = np.swapaxes(legendre0[1, 0:nstop+1, :], 0, 1)
    pishift = np.pad(pis, ((0,), (1,)))[..., 0:nstop+1]
    n = np.arange(nstop+1)
    # add order axis to mu; resulting shape is (num_thetas, 1)
    mu = mu[..., np.newaxis]
    taus = (n * pis * mu) - (n+1) * pishift

    # reshape to match thetas original shape
    ang_shape = thetas.shape + (nstop+1,)
    pis = np.reshape(pis, ang_shape)
    taus = np.reshape(taus, ang_shape)
    return pis[...,1:nstop+1], taus[...,1:nstop+1]


def _scatcoeffs(m, x, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    # index ratio should be specified as a 2D array with shape
    # [num_values, num_layers] to calculate over a set of different values,
    # such as wavelengths. If specified as a 1D array, shape is [num_layers].
    if m.shape[-1] > 1:
        return _scatcoeffs_multi(m, x)

    # Scattering coefficients for single-layer particles.
    # see B/H eqn 4.88
    # implement criterion used by BHMIE plus a couple more orders to be safe
    # nmx = np.array([nstop, np.round(np.absolute(m*x))]).max() + 20
    # Dnmx = mie_specfuncs.log_der_1(m*x, nmx, nstop)
    # above replaced with Lentz algorithm
    z = m * x
    z = z[..., 0]       # remove unneeded layer axis
    Dnmx = mie_specfuncs.dn_1_down(z, nstop + 1, nstop,
                                   mie_specfuncs.lentz_dn1(z, nstop + 1,
                                                           eps1, eps2))

    n = np.arange(nstop+1)
    psi, xi = mie_specfuncs.riccati_psi_xi(x, nstop)

    # insert zeroes at the beginning of second axis (order axis)
    psishift = np.pad(psi, ((0,), (1,)))[:, 0:nstop+1]
    xishift = np.pad(xi, ((0,), (1,)))[:, 0:nstop+1]
    an = ( (Dnmx/m + n/x)*psi - psishift ) / ( (Dnmx/m + n/x)*xi - xishift )
    bn = ( (Dnmx*m + n/x)*psi - psishift ) / ( (Dnmx*m + n/x)*xi - xishift )

    # coefficient array has shape (2, ..., nstop)
    return np.array([an[..., 1:nstop+1], bn[..., 1:nstop+1]])


def _scatcoeffs_multi(marray, xarray, nstop=None, eps1 = 1e-3, eps2 = 1e-16):
    '''Calculate scattered field expansion coefficients (in the Mie formalism)
    for a particle with an arbitrary number of spherically symmetric layers
    with different refractive indices.

    Parameters
    ----------
    marray : array_like, complex128
        array of layer indexes, innermost first.  If specified as a 2D array,
        axis=0 corresponds to the values over which to vectorize (e.g.
        wavelengths), and axis=1 corresponds to the layer indexes
    xarray : array_like, real
        array of layer size parameters (k * outer radius), innermost first.  If
        specified as a 2D array, axis=0 corresponds to the values over which to
        vectorize and axis=1 corresponds to the layer size parameters.
    nstop : int
        maximum order.  If not specified, uses largest value of x to determine
    eps1 : float, optional
        underflow criterion for Lentz continued fraction for Dn1
    eps2 : float, optional
        convergence criterion for Lentz continued fraction for Dn1

    Returns
    -------
    scat_coeffs : ndarray (complex)
        Scattering coefficients

    '''
    # ensure correct data types and shapes
    marray = marray.astype(complex)
    xarray = xarray.astype(complex)

    # sanity check: marray and xarray must be same size
    if marray.size != xarray.size:
        raise ValueError("Arrays of layer indices and size parameters must "
                         "have the same dimensions")

    # need number of layers L
    nlayers = marray.shape[-1]

    # calculate nstop based on largest radius
    if nstop is None:
        nstop = _nstop(xarray.max())

    # initialize H_n^a and H_n^b in the core, see eqns. 12a and 13a
    intl = mie_specfuncs.log_der_13(marray[..., 0] * xarray[..., 0],
                                    nstop, eps1, eps2)[0]
    hans = intl
    hbns = intl

    # m_l x_{l-1}
    z1 = marray[..., 1:] * xarray[..., :-1]
    # # m_l x_l
    z2 = marray[..., 1:] * xarray[..., 1:]

    # pre-calculate logarithmic derivatives for all layers
    derz1s = mie_specfuncs.log_der_13(z1, nstop, eps1, eps2)
    derz2s = mie_specfuncs.log_der_13(z2, nstop, eps1, eps2)

    # pre-calculate ratio Q_n^l for all layers
    Qnl_arr = mie_specfuncs.Qratio(z1, z2, nstop, dns1 = derz1s,
                                   dns2 = derz2s, eps1 = eps1, eps2 = eps2)

    # lay is l-1 (index on layers used by Yang)
    for lay in np.arange(1, nlayers):
        m = marray[..., lay]
        mm1 = marray[..., lay-1]

        # calculate logarithmic derivatives D_n^1 and D_n^3
        dz1s0, dz1s1 = derz1s[0][:, lay-1], derz1s[1][:, lay-1]
        dz2s0, dz2s1 = derz2s[0][:, lay-1], derz2s[1][:, lay-1]

        # calculate G1, G2, Gtilde1, Gtilde2 according to
        # eqns 26-29
        # using H^a_n and H^b_n from previous layer
        G1 = m[:, np.newaxis]*hans - mm1[:, np.newaxis]*dz1s0
        G2 = m[:, np.newaxis]*hans - mm1[:, np.newaxis]*dz1s1
        Gt1 = mm1[:, np.newaxis]*hbns - m[:, np.newaxis]*dz1s0
        Gt2 = mm1[:, np.newaxis]*hbns - m[:, np.newaxis]*dz1s1

        # calculate ratio Q_n^l for this layer
        Qnl = Qnl_arr[:, lay-1]

        # now calculate H^a_n and H^b_n in current layer
        # see eqns 24 and 25
        hans = (G2*dz2s0 - Qnl*G1*dz2s1) / (G2 - Qnl*G1)
        hbns = (Gt2*dz2s0 - Qnl*Gt1*dz2s1) / (Gt2 - Qnl*Gt1)
        # repeat for next layer

    # Relate H^a and H^b in the outer layer to the Mie scat coeffs
    # see Yang eqns 14 and 15
    #
    # n = 0 to nstop
    # (below we vectorize over the first dimension of xarray; we calculate the
    # max x over layers for each value of the first dimension)
    psiandxi = mie_specfuncs.riccati_psi_xi(xarray.max(axis=1)[:, np.newaxis],
                                            nstop)
    n = np.arange(nstop+1)
    psi = psiandxi[0]
    xi = psiandxi[1]
    # this doesn't bother to calculate psi/xi_{-1} correctly,
    # but OK since we're throwing out a_0, b_0 where it appears
    psishift = np.insert(psi, 0,
                         np.zeros(psi.shape[:-1]), axis=-1)[..., 0:nstop+1]
    xishift = np.insert(xi, 0,
                         np.zeros(xi.shape[:-1]), axis=-1)[..., 0:nstop+1]
    mlast = marray[..., nlayers-1][:, np.newaxis]
    xlast = xarray[..., nlayers-1][:, np.newaxis]
    an = (((hans/mlast + n/xlast)*psi - psishift)
          / ((hans/mlast + n/xlast)*xi - xishift))
    bn = (((hbns*mlast + n/xlast)*psi- psishift)
          / ((hbns*mlast + n/xlast)*xi - xishift))

    # output begins at n=1
    return np.array([an[..., 1:nstop+1], bn[..., 1:nstop+1]])


def _internal_coeffs(m, x, n_max, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate internal Mie coefficients c_n and d_n given
    relative index, size parameter, and maximum order of expansion.

    Follow Bohren & Huffman's convention. Note that van de Hulst and Kerker
    have different conventions (labeling of c_n and d_n and factors of m)
    for their internal coefficients.
    '''
    if x.shape[-1] > 1:
        raise ValueError("Internal Mie coefficients cannot yet be calculated "
                         "for layered sphere")

    m = m.astype(complex)
    x = x.astype(complex)
    z = m * x

    ratio = mie_specfuncs.R_psi(x, z, n_max, eps1, eps2)
    D1x, D3x = mie_specfuncs.log_der_13(x, n_max, eps1, eps2)
    D1mx = mie_specfuncs.dn_1_down(z, n_max + 1, n_max,
                                   mie_specfuncs.lentz_dn1(z, n_max + 1,
                                                           eps1, eps2))
    cl = (m[..., np.newaxis] * ratio * (D3x - D1x)
          / (D3x - m[..., np.newaxis] * D1mx))
    dl = (m[..., np.newaxis] * ratio * (D3x - D1x)
          / (m[..., np.newaxis] * D3x - D1mx))
    # start from l = 1
    cldl = np.array([cl[..., 1:], dl[..., 1:]])
    # remove unneeded layer axis
    return cldl[..., 0, :]


def _trans_coeffs(m, x, n_max, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate the transmission Mie coefficients c_n and d_n given
    relative index, size parameter, and maximum order of expansion.

    Note that the implementation here follows van de Hulst [1],
    in accordance with equation 3 from Bott and Zdunkowski [2].
    These coefficients are implemented in this convention for use in
    calculating the electromagnetic energy in the sphere,
    which is needed to calculate dwell times.

    [1] H. C. van de Hulst, Light Scattering by Small Particles
    (Wiley, New York, 1957), pp. 119-130.

    [2] Bott and Zdunkowski [2], J. Opt. Soc. Am. A, vol 4, no. 8, 1987.
    '''
    nstop=n_max
    n = np.arange(nstop+1)
    psi, _ = mie_specfuncs.riccati_psi_xi(m*x, nstop)
    psishift = np.insert(psi, 0,
                         np.zeros(psi.shape[:-1]), axis=-1)[..., 0:nstop+1]
    psi_prime = psishift - n*psi/(m*x)
    psi = psi[..., 1:nstop+1]
    psi_prime = psi_prime[..., 1:nstop+1]

    _, xi = mie_specfuncs.riccati_psi_xi(x, nstop)
    xishift = np.insert(xi, 0,
                        np.zeros(xi.shape[:-1]), axis=-1)[..., 0:nstop+1]
    xi_prime = xishift - n*xi/x
    xi = xi[..., 1:nstop+1]
    xi_prime = xi_prime[..., 1:nstop+1]

    cn = 1j/(xi*psi_prime - m*psi*xi_prime)
    dn = 1j/(m*psi_prime*xi - psi*xi_prime)

    return np.array([cn, dn])


def _time_coeffs(m, x, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    '''
    Calculate what we refer to as the time Mie coefficients gamma_n and An,
    given the relative inted, size parameter, maximum order of expansion.

    We follow the convention of equation 11 in
    Bott and Zdunkowski, J. Opt. Soc. Am. A, vol 4, no. 8, 1987

    using the recurrence relation in Bohren & Huffman's eq 4.88 for psi prime.
    and the expressions for cn and dn from equation 3 in Bott and Zdunkowski.
    '''

    n = np.arange(nstop+1)
    n_max = np.max(n)
    psi, _ = mie_specfuncs.riccati_psi_xi(m*x, nstop)
    psishift = np.insert(psi, 0,
                         np.zeros(psi.shape[:-1]), axis=-1)[..., 1:nstop+1]
    psi = psi[..., 1:nstop+1]
    n = n[..., 1:nstop+1]
    cn, dn = _trans_coeffs(m,x, n_max, eps1=eps1, eps2=eps2)

    # calculate gamma_n and An
    gamma_n = (m**2*(m*cn*psi)*np.conj(m*cn*psi)
               + m**2*(m*dn*psi)*np.conj(m*dn*psi))
    An = (psishift-n*psi/(m*x))/psi

    return gamma_n, An


def _W0(radius, n_medium):
    '''
    Calculates the time-averaged electromagnetic energy of a sphere having the
    electromagnetic properties of the surrounding medium, according to eq. 9
    of Bott and Zdunkowski, J. Opt. Soc. Am. A, vol 4, no. 8, 1987

    W0=2/3*np.pi*radius^3*E0^2*permittivity_medium

    where radius is the radius of the scatterer, permittivity_medium is the
    permittivity of the surrounding medium, and E_0 is the field incident on
    the scatterer

    We use units such that the energy density in vacuum is 1,
    where energy density in vacuum is expressed as:

    energy_density = 1/2*E0^2*permitttivity_medium

    So plugging this expression into the equation for W0, we have:
    W0 = 2/3*pi*radius^3*2*energy_density

    '''
    energy_density = 1
    W0=2/3*np.pi*radius**3*2*energy_density

    return W0


def _nstop(x):
    # Takes size parameter, outputs order to compute.  Previously used
    # criterion from Wiscombe, Applied Optics 19, 1505 (1980):
    #return (np.round(np.absolute(x+4.05*x**(1./3.)+2))).astype('int')
    # now modified to use:
    # Criterion for calculating near-field properties with exact Mie solutions
    # (J. R. Allardice and E. C. Le Ru, Applied Optics, Vol. 53, No. 31 (2014).
    return (np.round(np.absolute(x+11*x**(1./3.)+1))).squeeze().astype('int')


def _asymmetry_parameter(al, bl):
    '''
    Inputs: an, bn coefficient arrays from Mie solution

    See discussion in Bohren & Huffman p. 120.
    The output of this function omits the prefactor of 4/(x^2 Q_sca).
    '''
    # axis -1 (last axis) is order axis
    lmax = al.shape[-1]
    l = np.arange(lmax) + 1
    selfterm = (l[:-1] * (l[:-1] + 2.) / (l[:-1] + 1.) *
                np.real(al[..., :-1] * np.conj(al[..., 1:]) +
                        bl[..., :-1] * np.conj(bl[..., 1:]))).sum(axis=-1)
    crossterm = ((2. * l + 1.)/(l * (l + 1)) *
                 np.real(al * np.conj(bl))).sum(axis=-1)
    return selfterm + crossterm


def _cross_sections(al, bl):
    '''
    Calculates scattering and extinction cross sections
    given arrays of Mie scattering coefficients al and bl.

    See Bohren & Huffman eqns. 4.61 and 4.62.

    The output omits a scaling prefactor of 2 * pi / k^2 = lambda_media^2/2/pi.
    '''
    lmax = al.shape[-1]

    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)

    cscat = (prefactor * (np.abs(al)**2 + np.abs(bl)**2)).sum(axis=-1)
    cext = (prefactor * np.real(al + bl)).sum(axis=-1)

    # see p. 122 and discussion in that section. The formula on p. 122
    # calculates the backscattering cross-section according to the traditional
    # definition, which includes a factor of 4*pi for historical reasons. We
    # jettison the factor of 4*pi to get values that correspond to the
    # differential scattering cross-section in the backscattering direction.
    alts = 2. * (np.arange(lmax) % 2) - 1
    cback = (np.abs((prefactor * alts * (al - bl)).sum(axis=-1))**2)/4.0/np.pi

    return cscat, cext, cback


def _cross_sections_complex_medium_fu(al, bl, cl, dl, radius, n_particle,
                                      n_medium, x_scatterer, x_medium,
                                      wavelen):
    '''
    Calculates dimensional scattering, absorption, and extinction cross
    sections for scatterers in an absorbing medium. This function does not
    handle multilayered particles.

    al, bl: Mie scattering coefficients
    cl, dl: Mie internal coefficients
    radius: radius of the scatterer (Quantity in [length])
    n_particle: refractive index of the scatterer
    n_medium: refractive index of the medium in which scatterer is embedded
    x_scatterer: size parameter using the particle's refractive index
    x_medium: size parameter using the medium's refractive index
    wavelen: wavelength of light in vacuum (Quantity in [length])

    Reference
    ---------
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).

    '''
    # ensure broadcasting will work correctly by adding axis corresponding to
    # order (l)
    num_wavelen = np.atleast_1d(wavelen).shape[0]
    wavelen = np.reshape(wavelen, (num_wavelen, 1))

    # if the imaginary part of the medium index is close to 0, then use the
    # limit value of prefactor1 for the calculations
    if n_medium.imag <= 1e-7:
        prefactor1 = wavelen / (np.pi * radius**2 * n_medium.real)
    else:
        eta = 4*np.pi*radius*n_medium.imag/wavelen
        prefactor1 = eta**2 * wavelen / (2*np.pi*radius**2*n_medium.real*
                                        (1+(eta-1)*np.exp(eta)))

    lmax = al.shape[-1]
    l = np.arange(lmax) + 1
    prefactor2 = (2. * l + 1.)[np.newaxis, ...]

    # calculate the scattering efficiency
    _, xi = mie_specfuncs.riccati_psi_xi(x_medium, lmax)
    xishift = np.insert(xi, 0,
                        np.zeros(xi.shape[:-1]), axis=-1)[..., 0:lmax+1]
    xi = xi[..., 1:]
    xishift = xishift[..., 1:]

    bn = (np.abs(al)**2 * (xishift - l*xi/x_medium) * np.conj(xi) -
          np.abs(bl)**2 * xi *
          np.conj(xishift -  l*xi/x_medium)) / (2*np.pi*n_medium/wavelen)
    qscat = np.sum(prefactor1 * prefactor2 * bn.imag, axis=-1)

    # calculate the absorption and extinction efficiencies
    psi, _ = mie_specfuncs.riccati_psi_xi(x_scatterer, lmax)
    psishift = np.insert(psi, 0,
                        np.zeros(xi.shape[:-1]), axis=-1)[..., 0:lmax+1]
    psi = psi[..., 1:]
    psishift = psishift[..., 1:]

    an = (np.abs(cl)**2 * psi * np.conj(psishift - l*psi/x_scatterer) -
          np.abs(dl)**2 * (psishift - l*psi/x_scatterer)*
          np.conj(psi)) / (2*np.pi*n_particle/wavelen)
    qabs = np.sum(prefactor1 * prefactor2 * an.imag, axis=-1)
    qext = np.sum(prefactor1 * prefactor2 * (an+bn).imag, axis=-1)

    # calculate the cross sections
    cscat = qscat * np.pi * radius**2
    cabs = qabs * np.pi * radius**2
    cext = qext * np.pi * radius**2

    return(cscat, cabs, cext)


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
    particles embedded in an absorbing medium", J. Opt. Soc. Am. A, 18, 6
    (2001).

    '''
    # if multilayer, use outermost radius and size parameter corresponding to
    # outermost radius
    radius = np.array(radius.magnitude).max() * radius.units
    x = x.max(axis=-1)
    k = x/radius

    # add newaxis corresponding to order (l)
    x = x[..., np.newaxis]

    lmax = al.shape[-1]
    l = np.arange(lmax) + 1
    prefactor = (2. * l + 1.)[np.newaxis, ...]

    # if the imaginary part of k is close to 0 (because the medium index is
    # close to 0), then use the limit value of factor for the calculations
    # (see eq 10 of Sudiarta and Chylek for I_denom; the cross-section is
    # calculated from W/I_denom)
    factor_limit = 1/2
    exponent = np.exp(2*radius*k.imag)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(k.imag <= Quantity(1e-8, '1/nm'), factor_limit,
                          (exponent/(2*radius*k.imag) +
                           (1-exponent)/(2*radius*k.imag)**2))
    I_denom = k.real * factor

    psi, xi = mie_specfuncs.riccati_psi_xi(x, lmax)

    xishift = np.insert(xi, 0,
                        np.zeros(xi.shape[:-1]), axis=-1)[..., 0:lmax+1]
    xi = xi[..., 1:]
    xishift = xishift[..., 1:]
    xideriv = xishift - l*xi/x

    psishift = np.insert(psi, 0,
                        np.zeros(xi.shape[:-1]), axis=-1)[..., 0:lmax+1]
    psi = psi[..., 1:]
    psishift = psishift[..., 1:]
    psideriv = psishift - l*psi/x

    # calculate the scattering cross section from eq 5 of Sudiarta and Chylek
    term1 = (-1j * np.abs(al)**2 *xideriv * np.conj(xi) +
              1j* np.abs(bl)**2 * xi * np.conj(xideriv))
    numer1 = (np.sum(prefactor * term1, axis=-1)
              * np.conj(k)).real
    cscat = np.pi / np.abs(k)**2 * numer1 / I_denom

    # calculate the absorption cross section from eq 7 of Sudiarta and Chylek
    term2 = (1j*np.conj(psi)*psideriv - 1j*psi*np.conj(psideriv) +
             1j*bl*np.conj(psideriv)*xi + 1j*np.conj(bl)*psi*np.conj(xideriv) +
             1j*np.abs(al)**2*xideriv*np.conj(xi) -
             1j*np.abs(bl)**2*xi*np.conj(xideriv) -
             1j*al*np.conj(psi)*xideriv - 1j*np.conj(al)*psideriv*np.conj(xi))
    numer2 = (np.sum(prefactor * term2, axis=-1)
              * np.conj(k)).real
    cabs = np.pi / np.abs(k)**2 * numer2 / I_denom

    # calculate the extinction cross section from eq 8 of Sudiarta and Chylek
    term3 = (1j*np.conj(psi)*psideriv - 1j*psi*np.conj(psideriv) +
             1j*bl*np.conj(psideriv)*xi + 1j*np.conj(bl)*psi*np.conj(xideriv) -
             1j*al*np.conj(psi)*xideriv - 1j*np.conj(al)*psideriv*np.conj(xi))
    numer3 = (np.sum(prefactor * term3, axis=-1)
              * np.conj(k)).real
    cext = np.pi / np.abs(k)**2 * numer3 / I_denom

    return(cscat, cabs, cext)


def _scat_fields_complex_medium(m, x, thetas, kd, near_field=False):
    '''
    Calculates the scattered fields as a function of scattering angle theta
    using the full Mie solutions. These solutions are valid both in the near
    and far field. When the medium has a zero imaginary component of the
    refractive index (is non-absorbing), the full solutions at the far field
    match the standard far-field Mie solutions given by calc_cross_sections.
    This is not the case when there is absorption because the standard
    far-field solutions assume an arbitrary distance far away, so they don't
    depend on the distance from the scatterer. And when the medium absorbs, the
    cross sections should really depend on the distance away at which we
    integrate the differential cross sections. The phase function, (diff cross
    section / total cross section) is the same when calculated with the full
    Mie solutions in the far field as when calculated with the far-field Mie
    solutions because this ratio does not depend on how far we integrate from
    the scatterer.

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
    thetas: array of scattering angles
    kd: k * distance, where k = 2*np.pi*n_matrix/wavelen, and distance is the
        distance away from the center of the particle. The standard far-field
        solution is obtained when distance >> radius in a non absorbing medium.
    near_field: boolean
        Set to True to include the near-fields. Sometimes the full solutions
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
    # calculate mie coefficients
    nstop = _nstop(x.max())
    n = np.arange(nstop)+1.

    an, bn = _scatcoeffs(m, x, nstop)

    # calculate prefactor (omitting the incident electric field because it
    # cancels out when calculating the scattered intensity)
    En = 1j**n * (2*n+1) / (n*(n+1))

    # calculate pis and taus at the scattering angles theta
    pis, taus = _pis_and_taus(nstop, thetas)

    # calculate the scattered electric and magnetic fields (omitting the
    # sin(phi) and cos(phi) factors because they will be accounted for when
    # integrating to get the scattering cross section)

    # required for calculations with polarized light
    # reshape to (num_values, num_angles, order)
    th_shape = (kd.shape[0],) + thetas.shape + (len(n),)

    En = np.broadcast_to(En, th_shape)
    an = np.broadcast_to(an, th_shape)
    bn = np.broadcast_to(bn, th_shape)

    # if full Mie solutions are wanted (including the near field effects given
    # by the spherical Hankel terms). The near fields don't change the total
    # cross section much, but the angle-dependence of the differential cross
    # section will be very different from the ones obtained with the far-field
    # approximations. If kd is large (if we're in the far field) in a non
    # absorbing medium, then the full solutions reduce down to the standard
    # far-field solutions given by calc_cross_sections().
    if near_field:
        # calculate spherical Bessel function and derivative
        nstop_array = np.arange(0,nstop+1)
        jn = spherical_jn(nstop_array, kd)
        yn = spherical_yn(nstop_array, kd)
        zn = jn + 1j*yn
        zn = zn[..., 1:]

        _, xi = mie_specfuncs.riccati_psi_xi(kd, nstop)
        # insert zeroes at the beginning of second axis (order axis)
        xishift = np.pad(xi, ((0,), (1,)))[:, 0:nstop+1]
        xi = xi[..., 1:]
        xishift = xishift[..., 1:]
        bessel_deriv = xishift - n*xi/kd
        zn = np.broadcast_to(zn, th_shape)
        bessel_deriv = np.broadcast_to(bessel_deriv, th_shape)

        Es_theta = np.sum(En
                          * (1j * an * taus * bessel_deriv/kd - bn * pis * zn),
                          axis=-1)
        Es_phi = np.sum(En
                        * (-1j * an * pis * bessel_deriv/kd + bn * taus * zn),
                        axis=-1)
        Hs_phi = np.sum(En
                        * (1j * bn * pis * bessel_deriv/kd - an * taus * zn),
                        axis=-1)
        Hs_theta = np.sum(En
                          * (1j * bn * taus * bessel_deriv/kd - an * pis * zn),
                          axis=-1)

    # if the near field effects aren't desired, use the asymptotic form of the
    # spherical Hankel function in the far field (p. 94 of Bohren and Huffman)
    else:
        Es_theta = np.sum((2*n+1) / (n*(n+1)) * (an * taus + bn * pis),
                          axis=-1)* np.exp(1j*kd)/(-1j*kd)
        Es_phi = -np.sum((2*n+1) / (n*(n+1)) * (an * pis + bn * taus),
                         axis=-1)* np.exp(1j*kd)/(-1j*kd)
        Hs_phi = np.sum((2*n+1) / (n*(n+1))*(bn * pis + an * taus),
                        axis=-1)* np.exp(1j*kd)/(-1j*kd)
        Hs_theta = np.sum((2*n+1) / (n*(n+1))* (bn *  taus + an * pis),
                          axis=-1)* np.exp(1j*kd)/(-1j*kd)
        # note that these solutions are not currently used anywhere in mie.py.
        # When the fields are multiplied to calculate the intensity, the
        # exponential terms reduce down to a term that depends on kd (see
        # diff_scat_intensity_complex_medium(). So these equations lead to
        # intensities that are the same as those calculated with the scattering
        # matrix in diff_scat_intensity_complex_medium().
        # We leave the expressions here in case users ever have a need to know
        # the actual fields, rather than the intensities.

    return Es_theta, Es_phi, Hs_theta, Hs_phi


def diff_scat_intensity_complex_medium(m, x, thetas, kd, phis=None,
                                       near_field=False,
                                       incident_vector=None):
    """
    Calculates the differential scattered intensity in an absorbing medium.
    User can choose whether to include near fields.

    When phis is None:
       The solutions are given as a function of scattering angle theta.

       The differential scattered intensity is computed by substituting the
       scattered electric and magnetic fields into the radial component of the
       Poynting vector:

            I_par = Es_theta * conj(Hs_phi)
            I_perp = Es_phi * conj(Hs_theta)

        where conj() indicates the complex conjugate. The radial component of
        the Poynting vector is then 1/2 * Re(I_par - I_perp).

    When phis are provided:
        The solutions are given as a function of scattering angle theta and
        azimuthal angle phi.

        The differential scattered intensity is computed by substituting the
        scattered electric and magnetic fields into the z-component of the
        Poynting vector:

            I_x = Es_x * conj(Hs_x)
            I_y = -Es_y * conj(Hs_y)

        where conj() indicates the complex conjugate. The radial component of
        the Poynting vector is then 1/2 * Re(I_x - I_y).

    Parameters
    ----------
    m : complex, array-like
        complex particle relative refractive index, n_part/n_med
    x : complex, array-like
        size parameter, x = ka = 2*pi*n_med/lambda * a (sphere radius a)
    thetas : array-like
        Scattering angles.  Must be in radians.
    kd : float
        k * distance, where k = 2*np.pi*n_matrix/wavelen, and distance is the
        distance away from the center of the particle. The standard far-field
        solutions are obtained when distance >> radius in a non-absorbing
        medium.
    phis : None or ndarray
        Azimuthal angles for which to calculate the diff scat intensity. If not
        provided, scattering calculations will be carried out in the
        scattering plane coordinate system, defined by basis vectors parallel
        and perpendicular to scattering plane. If provided, scattering
        calculations will be carried out in the basis defined by basis vectors
        x and y in the lab frame, with z as the direction of propagation.
    near_field : boolean
        True to include the near-fields (default is False). Cannot be set to
        True while using cartesian basis (phis provided) because near field
        solutions are not implemented for cartesian coordinate system. Also
        cannot be set to True if using an incident_vector that is not None
        (unpolarized for scattering plane coordinate system). Often, the full
        solutions that include the near fields aren't wanted, for example when
        the total cross section calculation includes the structure factor, and
        the combination of the angle-dependent differential cross section
        multiplied by the structure factor gives very high cross sections at
        the surface of the particle. When we want to neglect the effect of the
        near fields and still integrate at the surface of the particle, we use
        the asymptotic form of the spherical Hankel function in the far field
        (p. 94 of Bohren and Huffman).
    incident_vector : None or tuple
        vector describing the incident electric field. It is multiplied by the
        amplitude scattering matrix to find the vector scattering amplitude. If
        phis are not provided, then this vector should be in the scattering
        plane basis, where the first element is the parallel component and the
        second element is the perpendicular component. If phis are provided,
        then this vector should be in the cartesian basis, where the first
        element is the x-component and the second element is the y-component.
        Note that the vector for unpolarized light is the same in either basis,
        since either way it should be an equal mix between the two othogonal
        polarizations: (1,1). If incident_vector is None, the function assigns
        a value based on the coordinate system. For the scattering plane basis,
        the assigned value is (1,1) because most scattering plane calculations
        we're interested in involve unpolarized light. For the cartesian basis,
        the assigned value is (1,0) because if we are going to the trouble to
        use the cartesian coordinate system, it is usually because we want to
        do calculations using polarization, and these calculations are much
        easier to convert to measured quantities when in the cartesian
        coordinate system.

    Returns
    -------
    I components : array with shape (2, ..., num_thetas, [num_phis])
        The two orthogonal components of scattered intensity as a function of
        angle. If in cartesian coordinate system, each component is a function
        of theta and phi values. If in scattering plane coordinate system, each
        component is an array of theta values (dimensionless). These
        intensities are technically "unitless." The intensities would get their
        units from the E_n term in the fields, which gets its units from an E_0
        term, which is taken to be 1 here. To get an intensity with real units
        you would need to multiply these by |E_0|**2 where E_0 is the amplitude
        of the incident wave at the origin, as well as by 1/|k|^2 (see Notes).

    Notes
    -----
    To get dimensional cross-sections, multiply by 1/|k|^2 (1/np.abs(k)**2),
    where k is the wavevector in media.

    References
    ----------
    C. F. Bohren and D. R. Huffman. Absorption and scattering of light by
    small particles. Wiley-VCH (2004), chapter 4.4.1.
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).

    """
    # ensure that broadcasting will work correctly by adding an axis
    # corresponding to theta
    kd = np.atleast_1d(kd)[..., np.newaxis]
    if phis is not None:
        kd = kd[..., np.newaxis]

    if near_field:
        if phis is None:
            # calculate scattered fields in scattering plane coordinate system
            Es_theta, Es_phi, Hs_theta, Hs_phi = _scat_fields_complex_medium(m,
                                        x,thetas, kd, near_field=near_field)
            I_1 = Es_theta * np.conj(Hs_phi) # I_par
            I_2 = -Es_phi * np.conj(Hs_theta) # I_perp
        else:
            raise ValueError("Near fields have not been implemented for the "
                             "Cartesian coordinate system. Set near_field "
                             "to False to calculate scattered intensity")


    else:
        # calculate vector scattering amplitude
        vec_scat_amp_1, vec_scat_amp_2 = vector_scattering_amplitude(m, x,
                                           thetas,
                                           phis=phis,
                                           incident_vector=incident_vector)

        # calculate the intensities. We multiply by a factor that accounts for
        # the dependence of the intensity on the distance away d from the
        # scatterer, which is necessary when the medium is absorbing. The
        # factor is derived from the multiplication of the exponential term
        # (the asymptotic form at large d of the spherical Hankel equations,
        # which account for near fields, see _scat_fields_complex_medium())
        # with its conjugate, assuming that k can be complex. The form reduces
        # down to 1/(kd)^2 when k is real, which is the factor usually used to
        # get the final intensity in a non-absorbing medium (p. 113 of Bohren
        # and Huffman).
        factor = np.exp(-2*kd.imag) / ((kd.real)**2 + (kd.imag)**2)
        I_1 = (np.abs(vec_scat_amp_1)**2)*factor # par or x
        I_2 = (np.abs(vec_scat_amp_2)**2)*factor # perp or y

    # the intensities should be real. We multiply by |kd|^2 so that the
    # resulting nondimensional cross-sections can be dimensionalized by
    # multiplying by 1/|k|^2 (just as with cross-sections returned by other
    # functions)
    return np.array([I_1.real, I_2.real])*np.abs(kd)**2


def integrate_intensity_complex_medium(dscat, thetas, kd,
                                       phi_min=0.0,
                                       phi_max=2*np.pi,
                                       phis=None):
    """
    Calculates the scattering cross section by integrating the differential
    scattered intensity at a distance of our choice in an absorbing medium.
    Choosing the right distance is essential in an absorbing medium because the
    differential scattering intensities decrease with increasing distance.
    The integration is done over scattering angles theta and azimuthal angles
    phi using the trapezoid rule.

    Parameters
    ----------
    dscat : array-like with shape (2, ..., num_thetas, [num_phis])
        differential scattered intensities for both polarizations. Can be
        functions of theta or of theta and phi. If a function of theta and phi,
        the theta dimension MUST come first
    thetas : array-like, shape ([angle_leading_dims], num_thetas)
        scattering angles
    kd : array-like, shape (...)
        wavevector in medium times distance (from the center of the particle)
        at which to integrate intensity
    phi_min : float
        minimum azimuthal angle, default set to 0. Used only if phis is None
        (coordinate system is scattering plane)
    phi_max : float
        maximum azimuthal angle, default set to 2*pi. Used only if phis is None
        (coordinate system is scattering plane)
    phis : None or ndarray, shape ([angle_leading_dims], num_phis)
        azimuthal angles

    Returns
    -------
    sigma: array-like with shape (...)
        integrated cross section
    sigma_1: array-like with shape (...)
        integrated cross section for first component of basis
    sigma_2: array-like with shape (...)
        integrated cross section for second component of basis

    Notes
    -----
    Returns dimensionless cross-sections.  Multiply these by 1/|k|^2
    (1/np.abs(k)**2) to recover the dimensional cross-sections.

    """
    # do phi integral first because it's the last dimension in dscat when
    # specified (thus phis will broadcast with dscat)
    if phis is not None:
        integrand = np.trapezoid(dscat, x=phis)
    else:
        integrand = dscat
        # integrate over phi: multiply by factor to integrate over phi
        # (this factor is the integral of cos(phi)**2 and sin(phi)**2 in
        # parallel and perpendicular polarizations, respectively)
        # This factor is needed to account for polarization, which introduces
        # factors of cos(phi) and sin(phi) for the electric fields.
        integrand[0] = (integrand[0] * (phi_max/2 + np.sin(2*phi_max)/4
                                        - phi_min/2 - np.sin(2*phi_min)/4))
        integrand[1] = (integrand[1] * (phi_max/2 - np.sin(2*phi_max)/4
                                        - phi_min/2 + np.sin(2*phi_min)/4))

    # integrate diff. cross-sections over theta using Jacobian
    integrand = integrand * np.abs(np.sin(thetas))
    sigma = np.trapezoid(integrand, x=thetas)

    # multiply by factor that accounts for attenuation in the incident light
    # (see Sudiarta and Chylek (2001), eq 10).
    # if the imaginary part of k is close to 0 (because the medium index is
    # close to 0), then use the limit value of factor for the calculations
    exponent = np.exp(2*kd.imag)
    factor_limit = 2
    # ignore division by zero in case k.imag=0; we'll replace the nans with the
    # limit value of factor anyway
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(kd.imag <= 1e-6, factor_limit,
                          1 / (exponent / (2*kd.imag)
                               + (1 - exponent) / (2*kd.imag)**2))

    # calculate the averaged sigma
    sigma = sigma * factor
    sigma_avg = sigma.sum(axis=0)/2

    return(sigma_avg, sigma[0], sigma[1])


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
    m : array-like
        complex relative refractive index
    x : array-like
        size parameter using the medium's refractive index
    thetas: array-like
        array of scattering angles.  Must be specified in radians.
    ktd: array-like
        kt * distance, where kt = 2*np.pi*n_particle/wavelen, and distance is
        the distance away from the center of the particle. The far-field
        solution is obtained when distance >> radius.

    Returns
    -------
    I_par, I_perp : array-like
        differential absorption intensities for an array of theta

    Reference
    ---------
    Q. Fu and W. Sun, "Mie theory for light scattering by a spherical particle
    in an absorbing medium". Applied Optics, 40, 9 (2001).

    '''
    # calculate mie coefficients
    nstop = _nstop(x.max())
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
    bessel_deriv = np.broadcast_to(bessel_deriv,
                                   [len(thetas),len(bessel_deriv)])

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


def amplitude_scattering_matrix(m, x, thetas, phis=None):
    """
    Calculates the amplitude scattering matrix for an n-dim array of thetas
    (and phis if in cartesian coordinate system)

    Elements of the amplitude scattering matrix are arranged as:
    [S2  S3]
    [S4  S1]

    Change of basis from scattering plane to lab frame cartesian is calculated
    by multiplying (M^-1)*S*M where S is the amplitude scattering matrix and
    M is the change of basis matrix. The change of basis matrix M is:

    [cosphi  sinphi]
    [sinphi -cosphi]

    (from Bohren and Huffman, 3.2, page 61)

    This matrix is equal to its inverse, so we can get the scattering matrix
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
    m : array-like, shape (..., num_layers)
        index ratio between the particle and sample
    x : array-like, shape (..., num_layers)
        size parameter
    thetas : array
        theta angles
    phis : None or array
        azimuthal angles for which to calculate the scattering matrix. If not
        provided (default), the calculation is done in the scattering plane
        basis. If provided, the calculation is done in the cartesian basis.

    Returns:
    --------
    S1, S2, S3, S4: tuple of arrays
       amplitude scattering matrix elements for all values (e.g. wavelengths)
       and theta. Shapes of all arrays are (num_values, num_theta, [num_phi])
    """
    # calculate n-array
    nstop = _nstop(x.max())
    n = np.arange(nstop)+1.
    prefactor  = (2*n+1)/(n*(n+1))

    # calculate mie coefficients
    coeffs = _scatcoeffs(m, x, nstop)

    # calculate amplitude scattering matrix in scattering plane basis
    S2, S1 = _amplitude_scattering_matrix(nstop, prefactor, coeffs, thetas)

    if phis is not None:
        # expand dims to allow broadcasting over phi
        S1 = S1[..., np.newaxis]
        S2 = S2[..., np.newaxis]
        phis = phis[..., np.newaxis, :]

        # calculate elements of scattering matrix
        cosphi = np.cos(phis)
        sinphi = np.sin(phis)
        S1_xy = S2*(sinphi)**2 + S1*(cosphi)**2
        S2_xy = S2*(cosphi)**2 + S1*(sinphi)**2
        S3_xy = S2*sinphi*cosphi - S1*sinphi*cosphi
        S4_xy = S2*cosphi*sinphi - S1*cosphi*sinphi
        return S1_xy, S2_xy, S3_xy, S4_xy
    else:
        S3 = np.zeros_like(S1)
        S4 = np.zeros_like(S1)
        return S1, S2, S3, S4


def vector_scattering_amplitude(m, x, thetas,
                                phis = None,
                                incident_vector = None):
    '''
    Calculates the vector scattering amplitude for an nd array of thetas and
    phis. For more info on the vector scattering amplitude and how to
    calculate it, see Bohren and Huffman, pg 70-73 of section 3.4 Extinction,
    Scattering, and Absorption.

    When phis are not provided, we are in the scattering-plane basis, and the
    default incident electric field vector assumes that the incident light is
    unpolarized (equally split between the parallel and perpendicular
    components of the electric field). The vector scattering amplitude can be
    calculated by:

            [S2   0] * [1]  = [S2]
            [0   S1]   [1]    [S1]

    where the vector is normalized after the multiplication

    When phis are provided, we are in the cartesian basis. If the incident
    electric field vector indicates the incident light is unpolarized, it is
    equally split between the x and y components of the electric field, so the
    vector scattering amplitude can be calculated by:

        [S2 S3] * [1] = [S2 + S3]
        [S4 S1]   [1]   [S4 + S1]

    however, in most cases, if we are in the cartesian coordinate system, we
    are more interested in calculating the the scattered light given and
    initial polarization of the initial light, since this calculation cannot be
    done in the scattering plane coordinate system. If we assume the initial
    polarization is in the +x direction, the vector scattering amplitude will
    be:

        [S2 S3] * [1] = [S2]
        [S4 S1]   [0] = [S4]

    where the vector is normalized after the multiplication. The default value
    is then +x when set to incident_vector is set to None.

    Parameters:
    ----------
    m: float or array-like
        index ratio between the particle and sample
    x: float or array-like
        size parameter
    thetas: array-like
        scattering angles
    incident_vector : None or tuple
        vector describing the incident electric field. It is multiplied by the
        amplitude scattering matrix to find the vector scattering amplitude. If
        phis are not provided, then this vector should be in the scattering
        plane basis, where the first element is the parallel component and the
        second element is the perpendicular component. If phis are provided,
        then this vector should be in the cartesian basis, where the first
        element is the x-component and the second element is the y-component.
        Note that the vector for unpolarized light is the same in either basis,
        since either way it should be an equal mix between the two othogonal
        polarizations: (1,1). If incident_vector is None, the function assigns
        a value based on the coordinate system. For the scattering plane basis,
        the assigned value is (1,1) because most scattering plane calculations
        we're interested in involve unpolarized light. For the cartesian basis,
        the assigned value is (1,0) because if we are going to the trouble to
        use the cartesian coordinate system, it is usually because we want to
        do calculations using polarization, and these calculations are much
        easier to convert to measured quantities when in the cartesian
        coordinate system.
    phis: ndarray or None
        azimuthal angles


    Returns:
    --------
    vector scattering amplitude: tuple of arrays (num_values, num_angles)
        tuple describing the vector scattering amplitude in the specified
        coordinate system. not normalized.
    '''
    # calculate the amplitude scattering matrix
    S1, S2, S3, S4 = amplitude_scattering_matrix(m, x, thetas,
                                                 phis = phis)

    if phis is None:
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
    """Amplitude scattering matrix from Mie coefficients

    """
    pis, taus = _pis_and_taus(n_stop, thetas)

    # For broadcasting over theta, set shape to (..., 1, order)
    coeffs = coeffs[..., np.newaxis, :]

    # result should have shape (..., num_thetas)
    S1 = np.sum(prefactor*(coeffs[0]*pis + coeffs[1]*taus), axis=-1)
    S2 = np.sum(prefactor*(coeffs[0]*taus + coeffs[1]*pis), axis=-1)
    return S2, S1


def _amplitude_scattering_matrix_RG(prefactor, x, thetas):
    """Amplitude scattering matrix from Rayleigh-Gans approximation

    """
    if x.shape[-1] > 1:
        raise ValueError("Rayleigh-Gans approximation cannot be used for "
                         "layered spheres")

    u = 2 * x * np.sin(thetas/2.)

    # for theta=0 the limit is 1*prefactor; the following will avoid a divide
    # by zero error by dividing everywhere that u!=0 and returning 1 where u=0
    p = np.divide(np.sin(u) - u*np.cos(u), u**3, out=np.ones_like(u),
                  where=u!=0)

    # result should have shape (..., num_angles)
    S1 = prefactor * 3 * p
    S2 = S1 * np.cos(thetas)
    return S2, S1
