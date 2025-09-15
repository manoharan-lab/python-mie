# Copyright 2016, Vinothan N. Manoharan, Sofia Makgiriadou
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
The python-mie (pymie) python package is a pure Python library for Mie
scattering calculations

Notes
-----
Based on work by Jerome Fung in the Manoharan Lab at Harvard University

Requires pint:
PyPI: https://pypi.python.org/pypi/Pint/
Github: https://github.com/hgrecco/pint
Docs: https://pint.readthedocs.io/en/latest/

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Sofia Magkiriadou <sofia@physics.harvard.edu>.
"""

import numpy as np
import pint

# Load either the default unit registry or the registry that has been set by
# packages that use pymie
ureg = pint.get_application_registry()
Quantity = ureg.Quantity

@ureg.check('[length]', None)
def q(wavelen, theta):
    """
    Calculates the magnitude of the momentum-transfer wavevector

    Parameters
    ----------
    wavelen: structcol.Quantity [length]
        wavelength in vacuum
    theta: structcol.Quantity [dimensionless]
        scattering angle (polar angle with z pointing along the incident
        direction)

    Returns
    -------
    structcol.Quantity [1/length]
        magnitude of wavevector
    """
    return 4*np.pi/wavelen * np.sin(theta/2.0)

def index_ratio(n_particle, n_matrix):
    """
    Calculates the ratio of refractive indices (m in Mie theory)

    Parameters
    ----------
    n_particle : array-like
        refractive index of particle at particular wavelength(s)
        can be complex
    n_matrix : array-like
        refractive index of matrix at a particular wavelength

    Returns
    -------
    ndarray or scalar (complex or float):
        Return type depends on type of n_particle and n_matrix, and return
        shape should be the same as n_particle
    """
    # The following handles these cases:
    # 1. if n_particle and n_matrix are scalars, return shape (1,1).
    # 2. If n_particle is an array and n_matrix is not, it's probably a layered
    # particle, so we return (1, n_particle.shape)
    m = np.atleast_2d(n_particle/n_matrix)

    # function will not change shape for n_particle or n_matrix with two or
    # more dimensions (they will broadcast normally)

    return m

@ureg.check('[length]', None, '[length]')
def size_parameter(wavelen, n_matrix, radius):
    """
    Calculates the size parameter x=k_matrix*a needed for Mie calculations

    Parameters
    ----------
    wavelen: structcol.Quantity [length], array-like
        wavelength in vacuum
    n_matrix: array-like
        refractive index of matrix at wavelength=wavelen.  If specified as 1D
        array, shape is [num_layers].  If 2D, shape is [num_wavelen,
        num_layers]
    radius: structcol.Quantity [length], array-like
        radius of particle

    Notes
    -----
    Nondimensionalizes from input arguments and strips units, returning a pure
    ndarray (not a Quantity object)

    Returns
    -------
    ndarray or scalar (complex or float):
        returns scalar if both wavelen and radius are scalars. If wavelen is an
        array, returns shape [len(wavelen), 1]. If radius is an array, returns
        shape [1, len(radius)].  If both are arrays, returns shape
        [len(wavelen), len(radius)]

    """
    # ensure size parameter calculation broadcasts correctly when both
    # wavelength and radius are arrays
    radius = np.broadcast_to(radius, (np.size(wavelen), np.size(radius)))
    wavelen = np.reshape(wavelen, (np.size(wavelen), 1))
    sp = (2 * np.pi * n_matrix / wavelen * radius)

    # must use to('dimensionless') in case the wavelength and radius are
    # specified in different units; pint doesn't automatically make
    # ratios such as 'nm'/'um' dimensionless
    if isinstance(sp, Quantity):
        sp = sp.to('dimensionless').magnitude

    return sp
