# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
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
Copied from holopy on 12 Sept 2017. Exceptions used in scatterpy module.  
These are separated out from the other exceptions in other parts of HoloPy to 
keep things modular.

.. moduleauthor :: Vinothan N. Manoharan <vnm@seas.harvard.edu>
"""

class InvalidScatterer(Exception):
    def __init__(self, scatterer, message):
        self.scatterer = scatterer
        super(InvalidScatterer, self).__init__(
            "Invalid scatterer of type " +
            self.scatterer.__class__.__name__ +
            ".\n" + message)
