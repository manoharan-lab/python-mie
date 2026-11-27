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
Benchmarks for miepython (https://github.com/scottprahl/miepython) that are
nearly equivalent to those for pymie in benchmarks.py. There is currently no
layered sphere model in miepython, so these benchmarks look only at homogeneous
spheres.

"""
import os
import numpy as np

# allow numba to compile miepython routines for the fastest timings
# (flag must be set before import)
os.environ["MIEPYTHON_USE_JIT"] = "1"
import miepython


class MiepythonCalculations:
    """Analogous to MieCalculations in benchmarks.py. Parameters should be set
    to be the same in both files (though miepython does not handle absorbing
    media), but note that miepython uses the opposite convention to pymie for
    the imaginary index.  In miepython, negative imaginary index corresponds to
    absorption, whereas in pymie, positive imaginary index corresponds to
    absorption.

    """
    # m values correspond to
    #   1) nonabsorbing PS in water
    #   2) absorbing PS in water
    # we don't use the 1-element tuple form because miepython expects scalars
    params = [[1.59/1.33,
               1.59/1.33 - 1e-3j],
              [0.1, 1.0, 10.0, 100.0, 1000.0]]
    param_names = ["m", "x"]

    def setup(self, m, x):
        thetas = np.linspace(0, np.pi, 1000)
        self.mu = np.cos(thetas)
        # run the calculations once to trigger numba compilation, so that
        # later time_ functions don't include the compilation time
        miepython.efficiencies_mx(m, x)
        miepython.S1_S2(m, x, self.mu, norm="bohren")

    def time_efficiencies(self, m, x):
        miepython.efficiencies_mx(m, x)

    def time_S1_S2(self, m, x):
        miepython.S1_S2(m, x, self.mu, norm="bohren")


class VectorizedMiepythonCalculations:
    """Analogous to VectorizedMieCalculations in benchmarks.py, with refractive
    index the complex conjugate of that in pymie (see note in
    MiepythonCalculations).

    """
    # we'll check how time scales with number of sphere diameters
    params = [1, 10, 100, 1000]
    param_names = ["n_sizes"]

    def setup(self, n_sizes):
        # Use a PS sphere with absorption in non-absorbing medium (water)
        self.m = 1.59/1.33 - 1e-3j
        # Choose size parameters around 10, corresponding to about
        # 1-micrometer-diameter particles at 500 nm incident wavelength
        self.x = np.linspace(7.0, 13.0, n_sizes)
        thetas = np.linspace(0, np.pi, 100)
        self.mu = np.cos(thetas)
        # trigger compilation
        miepython.efficiencies_mx(self.m, self.x)
        miepython.S1_S2(self.m, self.x[0], self.mu, norm="bohren")

    def time_efficiencies(self, n_sizes):
        miepython.efficiencies_mx(self.m, self.x)

    def time_S1_S2(self, n_sizes):
        # needs to be run as a loop because S1_S2 doesn't accept array inputs
        for x in self.x:
            miepython.S1_S2(self.m, x, self.mu, norm="bohren")
