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
from pint import UnitRegistry

# Load the default unit registry from pint and use it everywhere.
# Using the unit registry (and wrapping all functions) ensures that we don't
# make unit mistakes
ureg = UnitRegistry()
Quantity = ureg.Quantity

@ureg.check('[length]', '[]')
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

@ureg.check('[]', '[]')
def index_ratio(n_particle, n_matrix):
    """
    Calculates the ratio of refractive indices (m in Mie theory)

    Parameters
    ----------
    n_particle: structcol.Quantity [dimensionless] or ndarray thereof
        refractive index of particle at particular wavelength(s)
        can be complex
    n_matrix: structcol.Quantity [dimensionless]
        refractive index of matrix at a particular wavelength

    Notes
    -----
    Nondimensionalizes from input arguments and strips units, returning a pure
    ndarray (not a Quantity object)

    Returns
    -------
    ndarray or scalar (complex or float):
        Return type depends on type of n_particle and n_matrix, and return
        shape should be the same as n_particle
    """
    return (n_particle/n_matrix).magnitude

@ureg.check('[length]', '[]', '[length]')
def size_parameter(wavelen, n_matrix, radius):
    """
    Calculates the size parameter x=k_matrix*a needed for Mie calculations

    Parameters
    ----------
    wavelen: structcol.Quantity [length]
        wavelength in vacuum
    n_matrix: structcol.Quantity [dimensionless]
        refractive index of matrix at wavelength=wavelen
    radius: structcol.Quantity [length]
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
    if np.size(sp) == 1:
        sp = sp.item()

    return sp
