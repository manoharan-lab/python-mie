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

from pymie import mie, mie_specfuncs


class SpecialFunctions:
    """Benchmarks for the special functions used in Mie calculations, applied
    to non-vectorized cases.

    """
    # use the same size parameters as in MieCalculations (below)
    params = [0.1, 1.0, 10.0, 100.0, 1000.0,
              0.1 + 0.001j, 10.0 + 0.1j, 1000.0 + 10.0j]
    param_names = ["x"]

    def setup(self, x):
        self.nstop = mie._nstop(x)

    def time_riccati_psi_xi(self, x):
        mie_specfuncs.riccati_psi_xi(x, self.nstop)


class VectorizedSpecialFunctions:
    """As above, but for special functions evaluated over many size parameters.

    """
    params = [1, 10, 100, 1000]
    param_names = ["n_sizes"]

    def setup(self, n_sizes):
        # size parameters have shape (n_sizes, 1), with nstop set by the
        # largest x
        self.x = np.linspace(7.0, 13.0, n_sizes)[:, np.newaxis]
        self.nstop = mie._nstop(self.x.max())

    def time_riccati_psi_xi(self, n_sizes):
        mie_specfuncs.riccati_psi_xi(self.x, self.nstop)


class MieCalculations:
    # m values correspond to
    #   1) nonabsorbing PS in water
    #   2) absorbing PS in water
    #   3) core-shell particle with PS core and absorbing silica shell in water
    params = [[(1.59/1.33,),
               (1.59/1.33 + 1e-3j,),
               (1.59/1.33, 1.45/1.33 + 1e-3j)],
              [0.1, 1.0, 10.0, 100.0, 1000.0,
               # the following values correspond to an absorbing medium with an
               # imaginary index ~0.01
               0.1 + 0.001j, 10.0 + 0.1j, 1000.0 + 10.0j]]
    param_names = ["m", "x"]

    def setup(self, m, x):
        n_layers = len(m)
        self.m = np.array(m)
        # for multilayer spheres, we set the outer radius of each layer k to be
        # k/n of the particle size, where n is the number of layers
        self.x = x * (np.arange(1, n_layers + 1) / n_layers)
        self.thetas = np.linspace(0, np.pi, 1000)
        self.phis = np.linspace(0, 2*np.pi, 300)
        # for absorbing medium, calculate intensity at the particle surface
        if np.iscomplexobj(self.x):
            self.kd = self.x[-1]
        else:
            self.kd = None
        self.nstop = mie._nstop(self.x.max())

    def time_calc_cross_sections(self, m, x):
        mie.calc_cross_sections(self.m, self.x)

    def time_amplitude_scattering_matrix(self, m, x):
        mie.amplitude_scattering_matrix(self.m, self.x, self.thetas)

    def time_calc_ang_scat(self, m, x):
        mie.calc_ang_scat(self.m, self.x, self.thetas, kd=self.kd)

    def time_calc_ang_scat_2d(self, m, x):
        # time calculation of 2D phase function
        mie.calc_ang_scat(self.m, self.x, self.thetas, phis=self.phis,
                          kd=self.kd)

    def peakmem_calc_ang_scat_2d(self, m, x):
        # reports peak memory during execution, including memory used by python
        # and packages in addition to array memory
        mie.calc_ang_scat(self.m, self.x, self.thetas, phis=self.phis,
                          kd=self.kd)

    def time_scatcoeffs(self, m, x):
        mie._scatcoeffs(self.m, self.x, self.nstop)


class VectorizedMieCalculations:
    """As above, but benchmarks results for functions that are vectorized over
    m and x.

    """
    # we'll check how time and memory scale with number of sphere diameters
    params = [1, 10, 100, 1000]
    param_names = ["n_sizes"]

    def setup(self, n_sizes):
        # Use a PS sphere with absorption in non-absorbing medium (water), and
        # remember that m has to have shape (..., 1) for homogeneous spheres
        self.m = np.ones((n_sizes, 1)) * 1.59/1.33 + 1e-3j
        # Choose size parameters around 10, corresponding to about
        # 1-micrometer-diameter particles at 500 nm incident wavelength
        self.x = np.linspace(7.0, 13.0, n_sizes)[:, np.newaxis]
        self.thetas = np.linspace(0, np.pi, 100)
        self.nstop = mie._nstop(self.x.max())

    def time_calc_cross_sections(self, n_sizes):
        mie.calc_cross_sections(self.m, self.x)

    def time_calc_ang_scat(self, n_sizes):
        mie.calc_ang_scat(self.m, self.x, self.thetas)

    def peakmem_calc_ang_scat(self, n_sizes):
        mie.calc_ang_scat(self.m, self.x, self.thetas)

    def time_scatcoeffs(self, n_sizes):
        mie._scatcoeffs(self.m, self.x, self.nstop)
