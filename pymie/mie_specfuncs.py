# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, and Anna Wang
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
Compute special functions needed for the computation of scattering coefficients
in the Lorenz-Mie scattering solution and related problems such as layered
spheres.

Notes
-----
Forked from holopy on 16 Aug 2016; added python version of lentz_dn1 and
changed log_der_1 to dn_1_down, following the code in mieangfuncs.f90. These
changes implement the Lentz continued fraction algorithm.

These functions are not to be used for calculations at each field point.
Rather, they should be used once for the calculation of scattering
coefficients, which can then be used for field calculations.

References
----------
[1] D. W. Mackowski, R. A. Altenkirch, and M. P. Menguc, "Internal absorption
cross sections in a stratified sphere," Applied Optics 29, 1551-1559, (1990).

[2] W. Yang, "Improved recursive algorithm for light scattering by a
multilayered sphere," Applied Optics 42, 1710-1720, (2003).

.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""
import numpy as np


# default tolerances
DEFAULT_EPS1 = 1e-3
DEFAULT_EPS2 = 1e-16


def riccati_psi_xi(x, nstop):
    """Calculates Riccati-Bessel functions psi_n(x) and xi_n(x) for
    n = 0, ..., nstop.  From Bohren & Huffman page 101,

        psi_n(x) = x j_n(x)
        xi_n(x) = x h1_n(x)

    Both functions satisfy the recurrence relation (B&H 4.11)

        f_{n-1}(x) + f_{n+1}(x) = (2n+1)/x f_n(x)

    For xi_n we use upward recurrence, which is stable for all n. For psi_n we
    use upward recurrence for n <= |x| and downward recurrence for n > |x| when
    x is real or nearly so. As Wiscombe and others have shown, upward
    recurrence for psi_n is stable at n <= |x| and unstable above. Upward
    recurrence is more accurate in the region where psi_n is oscillatory (n <=
    |x|) because downward recurrence loses accuracy near the zeros of the
    function there. When x has a substantial imaginary part (|Im(x)| >= 1, a
    threshold chosen empirically by Claude), we use downward recurrence for all
    orders. When |Im(x)| is large, upward recurrence becomes unstable, but
    downward recurrence is accurate because the zeros of psi_n lie on the real
    axis, and x is far enough away from the real axis that |psi_n| does not
    become small.

    Specifically, the downward recurrence is for the ratios r_n, where

        r_n = psi_n/psi_{n-1} = 1 / ((2n+1)/x - r_{n+1}),

    starting from r = 0 at an order well above nstop and |x|.  We recover psi_n
    using the relationship

        psi_{n-1} = -i / (xi_n - r_n xi_{n-1})

    which can be derived from the Wronskian (B&H eq. 4.60), the definition xi_n
    = psi_n - i chi_n, the recurrence relations for the Riccati-Bessel
    derivatives (below B&H eq. 4.88), and the definition of r_n. Other
    approaches (see SBESJY.F and scipy's riccati_jn) recover psi_n using
    psi_0(x) = sin(x) but these approaches are not accurate near zeros of
    sin(x).

    Parameters
    ----------
    x : float, complex, or array-like
        Argument. By convention, x is treated as at least 1D.  The order n is
        added as the last axis or, if the trailing axis is length 1 as for a
        layer axis of a homogeneous sphere, replaces it.
    nstop : int
        Highest order to calculate

    Returns
    -------
    ndarray
        Array of shape (2, ..., nstop+1) containing psi_n(x) and xi_n(x)

    Notes
    -----
    This function used to use scipy's spherical_jn/spherical_yn, which are
    vectorized over the parameter x. However, they calculate each order
    separately, even though all orders are calculated en route. They ended up
    being very inefficient for Mie calculations. scipy's riccati_jn/riccati_yn
    calculate all orders in one pass, but they are not vectorized over x. This
    function is vectorized over the inputs and calculates all orders (though
    note that there are loops over the order that cannot be vectorized because
    they come from three-term recurrence relations).

    References
    ----------
    [1] C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by
    Small Particles (Wiley, 1983).

    [2] Wiscombe WJ "Mie Scattering Calculations: Advances in Technique
    and Fast, Vector-speed Computer Codes." NCAR Technical Note
    NCAR/TN-140+STR (1979)

    """
    x = np.atleast_1d(x)
    abs_x = np.abs(x)
    # dtype for psi_n and r_n which is real for real x and complex for complex
    # x (real arithmetic is faster and uses less memory)
    real_or_complex = np.result_type(x, float)

    # xi_n for n = 0, ..., nstop+1 by upward recurrence, starting from
    #     xi_0 = -i exp(ix)
    #     xi_1 = -exp(ix) (1 + i/x)
    # which follow from xi_n = x j_n + ix y_n and the first two orders of j_n
    # and y_n (top of p. 87)
    xi = np.empty(x.shape + (nstop + 2,), dtype=complex)
    xi[..., 0] = -1j * np.exp(1j*x)
    xi[..., 1] = -np.exp(1j*x) * (1 + 1j/x)
    for n in range(1, nstop + 1):
        xi[..., n+1] = (2*n + 1)/x * xi[..., n] - xi[..., n-1]

    # psi_n for n = 0, ..., n_intermediate, where n_intermediate <= |x|, by
    # upward recurrence starting from
    #     psi_0 = sin(x)
    #     psi_1 = sin(x)/x - cos(x).
    # We choose maximum order from max value of x.  So in a vectorized
    # calculation, some calculations may overflow, but we will not use these
    # values in the end.  Note that for real x, psi is real.
    psi_up = np.zeros(x.shape + (nstop+1,), dtype=real_or_complex)
    psi_up[..., 0] = np.sin(x)
    if nstop >= 1:
        psi_up[..., 1] = np.sin(x)/x - np.cos(x)
    n_up = min(nstop, int(abs_x.max()))
    # ignore overflows when they happen (we will discard these anyway)
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(1, n_up):
            psi_up[..., n+1] = (2*n + 1)/x * psi_up[..., n] - psi_up[..., n-1]

    # r_n = psi_n/psi_{n-1} for n = 1, ..., nstop+1 by downward recurrence.
    # The starting order must be well above both nstop and |x| so that error
    # from starting at r = 0 decays before reaching nstop.  Each element of x
    # starts at its own order, so that results for an element don't depend on
    # the other elements.
    ratio = np.zeros(x.shape, dtype=real_or_complex)
    r = np.empty(x.shape + (nstop + 2,), dtype=real_or_complex)
    # BHMIE downward recurrence uses max(nstop, |x|) + 15 terms. We include
    # more terms using a scaling factor determined empirically by Claude
    n_start = np.floor(np.maximum(nstop, abs_x) + 15 + 4*abs_x**(1/3))
    # warmup phase: from n_start to nstop+2; these values are not saved
    for n in range(int(n_start.max()), nstop + 1, -1):
        # np.where is needed because n_start varies with x
        ratio = np.where(n <= n_start, 1 / ((2*n + 1)/x - ratio), 0)
    # store ratios for n = nstop + 1 to n = 1 (n_start > nstop+1 always)
    for n in range(nstop + 1, 0, -1):
        ratio = 1 / ((2*n + 1)/x - ratio)
        r[..., n] = ratio

    # recover psi_n for n = 0, ..., nstop from the Wronskian-based relation
    psi_down = -1j / (xi[..., 1:] - r[..., 1:] * xi[..., :-1])
    if not np.iscomplexobj(x):
        # discard imaginary part because psi(x) should be real for real x
        psi_down = psi_down.real

    # now combine the values from upward and downward recurrence, using upward
    # only when n <= |x| and x is real or nearly so (|Im(x)| < 1, a threshold
    # chosen empirically by Claude).
    n = np.arange(nstop + 1)
    use_up = ((n <= abs_x[..., np.newaxis])
              & (np.abs(x.imag)[..., np.newaxis] < 1))
    psi = np.where(use_up, psi_up, psi_down)

    result = np.array([psi, xi[..., :nstop+1]])
    # the order replaces a trailing axis of length 1 (such as the layer axis
    # for a homogeneous sphere).  We remove that axis only at the end, so that
    # all of the arithmetic above is done on arrays.  NumPy can round
    # arithmetic on scalars differently from arithmetic on arrays, and we want
    # the result for an element of x not to depend on whether it is part of an
    # array.
    if x.shape[-1] == 1:
        result = result[..., 0, :]

    return result


def lentz_dn1(z, n, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    """Calculate logarithmic derivative D_n(z) of the Riccati-Bessel function
    for a single value of n using the Lentz (1976) continued fraction method
    [1]_.

    This function is used to seed the down-recursion algorithm in
    `dn_1_down()`, which calculates the other logarithmic derivatives needed to
    calculate the Mie coefficients.

    Notes
    -----
    Implements check/workaround for ill-conditioning described under "Algorithm
    Improvement" in Lentz (1976) [1]_; see also Wiscombe/NCAR Mie report [2]_.
    Function is vectorized across z (uses array operations to handle multiple
    values of z in parallel) but still requires a loop to test for convergence.
    Elements that have converged are frozen (multiplied by exactly 1) while the
    others continue to iterate, so that the result for each element does not
    depend on the others.

    Parameters
    ----------
    z : array-like (complex)
        argument of the logarithmic derivative
    n : integer
        order of the logarithmic derivative
    eps1 : float
        value of continued fraction numerator or denominator triggering
        ill-conditioning workaround. Recommend 1e-3.
    eps2 : float
        converge when additional products in continued fraction differ by less
        than eps2 from 1. Recommend 1e-16.

    Returns
    -------
    array-like : shape of z
        value of D_n(z)

    References
    ----------
    [1] W. J. Lentz, "Generating Bessel functions in Mie scattering
    calculations using continued fractions," Applied Optics 15(3), 668-671
    (1976).

    [2] W. Wiscombe, "Mie Scattering Calculations: Advances in Technique and
    Fast, Vector-speed Computer Codes," University Corporation for Atmospheric
    Research. https://doi.org/10.5065/D6ZP4414 (Original work published 1979)

    """
    z = np.atleast_1d(z)

    def a_i(i):
        # elements of continued fraction expansion
        # Note: Wiscombe uses a different value for a_1 than the one given by
        # this formula, but Lentz uses the same formula for all a_i
        return (-1.)**(i + 1) * 2. * (n + i - 0.5) / z

    # calculate numerator and denominator and a_2
    numerator = a_i(2) + 1. / a_i(1)
    denominator = a_i(2)
    product = a_i(1) * numerator / denominator
    ratio = product.copy()
    # elements that have not yet converged
    active = np.abs(product - 1) > eps2
    i = 3

    while active.any():
        ai = a_i(i)
        numerator = ai + 1. / numerator
        denominator = ai + 1. / denominator
        ill_conditioned = (np.minimum(np.abs(numerator), np.abs(denominator))
                           < eps1 * np.abs(ai))
        if (ill_conditioned & active).any():
            # ill conditioning: combine two terms, as in Lentz (1976)
            xi1 = 1. + a_i(i + 1) * numerator
            xi2 = 1. + a_i(i + 1) * denominator
            ratio = np.where(active, ratio * xi1 / xi2, ratio)
            numerator = a_i(i + 2) + numerator / xi1
            denominator = a_i(i + 2) + denominator / xi2
            i += 2
        product = np.where(active, numerator / denominator, 1)
        ratio = ratio * product
        active = active & (np.abs(product - 1) > eps2)
        i += 1

    return ratio - n / z


def dn_1_down(z, nmx, nstop, start_val):
    """Calculates logarithmic derivative D_n(z) of the Riccati-Bessel function
    psi_n(z) for n = 0, ..., nstop by downward recurrence, as in BHMIE [1]_.

    The recurrence is

                      n           1
        D_{n-1}(z) = ---  -  -----------
                      z       D_n + n/z

    (B&H eq. 4.89)

    Parameters
    ----------
    z : array-like (real or complex)
        argument of the logarithmic derivative
    nmx : integer
        order from which downward recursion begins (nmx >= nstop)
    nstop : integer
        maximum order returned
    start_val : array-like, shape of z
        value of D_nmx(z), from which recursion begins

    Returns
    -------
    ndarray : shape of z + (nstop+1,)
        values of D_n(z) for n = 0, ..., nstop

    Notes
    -----
    Downward recurrence for D_n is stable, unlike upward recurrence [1]_.
    Function is vectorized across z but requires a loop over n.  D_n(z) is real
    for real z.

    References
    ----------
    [1] C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by
    Small Particles (Wiley, 1983).

    """
    z = np.array(z)
    start_val = np.atleast_1d(start_val)
    # D_n is real for real z
    real_or_complex = np.result_type(z, start_val, float)
    # since dn will be full at the end, start empty to save some time
    dn = np.empty(start_val.shape + (nmx+1,), dtype=real_or_complex)
    dn[..., nmx] = start_val

    # pre-calculate 1/z to save a little time in the downward recurrence loop
    one_over_z = 1/z
    for n in range(nmx, 0, -1):
        n_over_z = n * one_over_z
        dn[..., n-1] = n_over_z - 1.0/(dn[..., n] + n_over_z)
    return dn[..., 0:nstop+1]


def log_der_13(z, nstop, eps1 = DEFAULT_EPS1, eps2 = DEFAULT_EPS2):
    """Calculate logarithmic derivatives D_n^1(z) and D_n^3(z) of the
    Riccati-Bessel functions psi_n(z) and xi_n(z) for n = 0, ..., nstop,
    following Mackowski, Altenkirch, Menguc [1]_.

    We calculate D_n^1 by downward recurrence (Mackowski eq. 62), seeded by
    `lentz_dn1()`, and D_n^3 by upward recurrence together with the product
    (psi_n xi_n) (Mackowski eqs. 63 and 64).

    Parameters
    ----------
    z : array-like
        argument of the logarithmic derivatives, converted to complex
    nstop : integer
        maximum order
    eps1 : float
        underflow criterion for Lentz continued fraction for D_n^1
    eps2 : float
        convergence criterion for Lentz continued fraction for D_n^1

    Returns
    -------
    dn1, dn3 : ndarray (complex), each with [shape of z] + (nstop+1,)
        values of D_n^1(z) and D_n^3(z) for n = 0, ..., nstop

    Notes
    -----
    Riccati-Bessel conventions follow Bohren & Huffman.  Function is vectorized
    across z but requires loops over order.

    This function is used to calculate the scattering coefficients of
    multilayer spheres (`mie._multilayer_h()`) and the internal-field
    coefficients of homogeneous spheres (`mie._internal_coeffs()`), which are
    needed only for calculations in an absorbing medium. Scattering
    coefficients of homogeneous spheres, in absorbing or nonabsorbing media,
    need only D_n^1, which `mie._scatcoeffs()` calculates from `lentz_dn1()`
    and `dn_1_down()`.

    References
    ----------
    [1] D. W. Mackowski, R. A. Altenkirch, M.P. Menguc, "Internal absorption
    cross sections in a stratified sphere," Applied Optics, 29(10):1551-1559
    (1990).

    """
    z = np.atleast_1d(z).astype(complex)

    # calculate D_n^1 (based on psi) by downward recurrence (Mackowski eq. 62)
    dn1 = dn_1_down(z, nstop + 1, nstop, lentz_dn1(z, nstop + 1, eps1, eps2))

    # calculate D_n^3 (based on xi) by upward recurrence (Mackowski eqs. 63 and
    # 64)
    dn3 = np.empty(z.shape + (nstop+1,), dtype=complex)
    psixi = np.empty(z.shape + (nstop+1,), dtype=complex)
    dn3[..., 0] = 1.j
    psixi[..., 0] = -1j * np.exp(1.j*z) * np.sin(z)
    for n in range(1, nstop+1):
        n_over_z = n/z
        # Mackowski eq. 63
        psixi[..., n] = (psixi[..., n-1] * (n_over_z - dn1[..., n-1])
                         * (n_over_z - dn3[..., n-1]))
        # Mackowski eq. 64
        dn3[..., n] = dn1[..., n] + 1j/psixi[..., n]

    return dn1, dn3


def Qratio(z1, z2, nstop, dns1, dns2):
    """Calculate the ratio Q_n^l of Riccati-Bessel functions (defined in Yang
    eq. 23) by upward recurrence (Yang eq. 33) [1]_.

    Parameters
    ----------
    z1 : array-like (complex) with shape (..., num_layers)
        index of layer l (m_l) times size parameter of layer l-1 (x_{l-1})
    z2 : array-like (complex) with shape (..., num_layers)
        index of layer l (m_l) times size parameter of layer l (x_l)
    nstop : integer
        maximum order
    dns1, dns2 : tuple of ndarray
        logarithmic derivatives (D_n^1, D_n^3) at z1 and z2, as returned by
        `log_der_13()`, each with shape (..., num_layers, nstop+1)

    Returns
    -------
    ndarray (complex), shape (..., num_layers, nstop+1)
        Q_n^l for n = 0, ..., nstop

    Notes
    -----
    Because each Q_n^l is Q_{n-1}^l times a factor that does not depend on Q,
    the recurrence is a running product, which we calculate with np.cumprod
    rather than a loop over order.

    This function is used only in calculating the scattering coefficients of
    multilayer spheres (`mie._multilayer_h()`). In Yang's algorithm, the
    functions H_n^a and H_n^b in each layer (Yang eqs. 24 and 25) depend on
    Q_n^l, the ratio of psi_n/xi_n at the inner boundary of layer l to
    psi_n/xi_n at its outer boundary. As Yang notes, calculating Q_n^l by
    recurrence, rather than from psi_n and xi_n themselves, avoids "the
    numerical difficulties in earlier methods that were due to the subtraction
    of nearly equal terms, leaving a small residual."

    References
    ----------
    [1] W. Yang, "Improved recursive algorithm for light scattering by a
    multilayered sphere," Applied Optics 42, 1710-1720, (2003).

    """
    d1z1, d3z1 = dns1
    d1z2, d3z2 = dns2

    # Q_0^l (Yang eq. 34), where z = a + ib
    a1, b1 = z1.real, z1.imag
    a2, b2 = z2.real, z2.imag
    q0 = (np.exp(-2.*(b2 - b1)) * (np.exp(-2j*a1) - np.exp(-2*b1))
          / (np.exp(-2j*a2) - np.exp(-2*b2)))

    # ratios Q_n^l/Q_{n-1}^l for n = 1, ..., nstop (Yang eq. 33). Need to add
    # order axis to z1, z2 for broadcasting; result has shape
    # (..., num_layers, nstop)
    n_over_z1 = np.arange(1, nstop+1) / z1[..., np.newaxis]
    n_over_z2 = np.arange(1, nstop+1) / z2[..., np.newaxis]
    ratio = ((d3z1[..., 1:] + n_over_z1) * (d1z2[..., 1:] + n_over_z2)
             / ((d3z2[..., 1:] + n_over_z2) * (d1z1[..., 1:] + n_over_z1)))

    # vectorized upward recurrence (Yang eq. 33) using np.cumprod. Add order
    # axis to q0, giving shape (..., num_layers, 1)
    q0 = q0[..., np.newaxis]
    return np.concatenate((q0, q0 * np.cumprod(ratio, axis=-1)), axis=-1)


def R_psi(z1, z2, nstop, dn1_z1, dn1_z2):
    """Calculate ratio of Riccati-Bessel functions psi_n(z1)/psi_n(z2) for
    n = 0, ..., nstop by upward recurrence (Mackowski eqs. 65 and 66) [1]_.

    Parameters
    ----------
    z1, z2 : array-like (complex)
        arguments of the Riccati-Bessel functions
    nstop : integer
        maximum order
    dn1_z1, dn1_z2 : ndarray
        logarithmic derivatives D_n^1 at z1 and z2 (each an ndarray, not a
        tuple), as returned by `dn_1_down()` or the first element returned by
        `log_der_13()`, each with shape [shape of z] + (nstop+1,)

    Returns
    -------
    ndarray (complex), shape [shape of z] + (nstop+1,)
        psi_n(z1)/psi_n(z2) for n = 0, ..., nstop

    Notes
    -----
    Because psi_n/psi_{n-1} = 1/(D_n^1 + n/z), each ratio is the previous one
    times a factor that does not depend on the ratio. The recurrence is thus a
    running product, which we calculate with np.cumprod rather than a loop over
    order.

    This function is used only to calculate the internal-field coefficients
    of homogeneous spheres (`mie._internal_coeffs()`).

    References
    ----------
    [1] D. W. Mackowski, R. A. Altenkirch, and M. P. Menguc, "Internal
    absorption cross sections in a stratified sphere," Applied Optics 29(10),
    1551-1559 (1990).

    """
    # psi_0(z1)/psi_0(z2), with an order axis added
    ratio_0 = (np.sin(z1) / np.sin(z2))[..., np.newaxis]

    # calculate the factors by which the ratio changes from order n-1 to
    # order n, for n = 1, ..., nstop. Need to add order axis to z1, z2 for
    # broadcasting
    n_over_z1 = np.arange(1, nstop+1) / z1[..., np.newaxis]
    n_over_z2 = np.arange(1, nstop+1) / z2[..., np.newaxis]
    prod = (dn1_z2[..., 1:] + n_over_z2) / (dn1_z1[..., 1:] + n_over_z1)
    output_vec = np.concatenate((ratio_0, ratio_0 * np.cumprod(prod, axis=-1)),
                                axis=-1)

    return output_vec
