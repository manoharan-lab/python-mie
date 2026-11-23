# Copyright 2026, Vinothan N. Manoharan
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
# this package. If not, see <https://www.gnu.org/licenses/>.
"""
Benchmarks for Mie calculations and special function evaluations

"""
import numpy as np

from pymie import mie


class CrossSections:
    def setup(self):
        # arrays need to be at least 1d (axis corresponds to layers)
        self.m = np.array([1.59/1.33])
        self.x = np.array([10.0])

    def time_calc_cross_sections(self):
        mie.calc_cross_sections(self.m, self.x)
