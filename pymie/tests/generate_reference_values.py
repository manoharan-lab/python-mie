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
Generates reference values for Mie quantities using the arbitrary-precision Mie
implementation in mie_reference.py.  These values are written to
reference_values.json in this directory.  They are used in
test_against_reference.py to test the accuracy of pymie.

To regenerate the reference values (which will take a few minutes), run

    python -m pymie.tests.generate_reference_values

from the repository root. Note that mpmath must be installed.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
   (with assistance from Claude Opus 5.5)
"""
import datetime
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import mpmath
import numpy as np

from pymie import mie
from pymie.tests import mie_reference

# working precision for reference implementation, in decimal digits
WORKING_DIGITS = 30
# number of digits to store in reference file
STORED_DIGITS = 25

OUTPUT_FILE = Path(__file__).with_name("reference_values.json")

# values of m, x to use in calculating scattering coefficients. Per pymie
# convention, both m and x must be at least one-dimensional. Homogeneous
# spheres have a one-dimensional m (and x) with a single value. Multilayer
# spheres have a one-dimensional m (and x) with values for each layer.  For
# now, only homogeneous spheres are supported by the reference implementation.
MX_VALUES = [([1.5], [0.1]),
             ([1.5], [10.0]),
             ([0.8], [10.0]),
             ([0.8], [100.0]),
             ([1.5 + 0.01j], [100.0]),
             ([1.5 + 1j], [100.0]),
             ([1.5 + 1e-3j], [1000.0])]

# scattering angles (in radians) at which to calculate the amplitude scattering
# matrix for each particle.  We use .tolist() to convert to Python floats
# because we're using repr() to output them to the reference value file (and
# repr() for a numpy array would give things like "np.float64(0.8888)")
THETAS = np.linspace(0, np.pi, 7).tolist()

# arguments at which to calculate the Riccati-Bessel functions, along with the
# highest order to calculate. If the order is None, the generation function
# should use pymie's stopping criterion to determine it. Note that z=pi and
# z=10 pi are where psi_0(z) = sin(z) is nearly zero. Tests that check these
# cases will catch methods that normalize by psi_0. Tests that check the cases
# at small z and large n will catch methods that compute psi_n by upward
# recurrence, which is known to be unstable for n > z.
Z_VALUES = [(0.1, None), (1.0, None), (10.0, None), (100.0, None),
            (1000.0, None),
            (np.pi, None), (10*np.pi, None),
            (0.01, 30), (0.1, 30), (1.0, 30)]

# scattering angles (in radians) and maximum order at which to calculate the
# angular functions pi_n and tau_n
ANGULAR_THETAS = THETAS
ANGULAR_NSTOP = 200


def file_hash(path):
    """Generates SHA-256 hash of a text file's contents. Used to identify which
    version of this module and mie_reference module were used to generate the
    reference values.

    """
    return hashlib.sha256(Path(path).read_text().encode()).hexdigest()


def to_string(z, stored_digits=STORED_DIGITS):
    """Convert an mpmath number to a string that can be parsed by
    ast.literal_eval(), with stored_digits significant figures in each part
    (real, imaginary).

    examples of stored strings: "1.5+0.1j", "1.9-0.25j"

    """
    z = mpmath.mpmathify(z)
    real = mpmath.nstr(z.real, stored_digits)
    imag = mpmath.nstr(z.imag, stored_digits)
    if not imag.startswith("-"):
        imag = "+" + imag
    return real + imag + "j"


@contextmanager
def report(step, output=True):
    """Context manager to print the name of a calculation step and output how
    long it took.  If output is False, prints nothing.

    """
    if output:
        print(f"    {step} ...", end=" ", flush=True)
    t_start = time.perf_counter()
    yield
    if output:
        print(f"{time.perf_counter() - t_start:.1f} s", flush=True)


def riccati_values(z, nstop=None, output=True):
    """Calculates reference values of the Riccati-Bessel functions psi_n(z) and
    xi_n(z) for n = 0, ..., nstop. If nstop is None, it is set by pymie's
    stopping criterion.  If output is True, print status messages.

    """
    # use pymie's stopping criterion (Allardice & Le Ru)
    if nstop is None:
        nstop = int(mie._nstop(z))

    with mpmath.workdps(WORKING_DIGITS):
        with report("Riccati-Bessel functions", output):
            psi, xi = mie_reference.riccati_psi_xi(mpmath.mpmathify(z), nstop)
    return {"z": repr(z),
            "nstop": nstop,
            "psi": [to_string(psi_n) for psi_n in psi],
            "xi": [to_string(xi_n) for xi_n in xi]}


def angular_values(theta, nstop, output=True):
    """Calculates reference values of the angular functions pi_n and tau_n for
    n = 1, ..., nstop at scattering angle theta (in radians).  If output is
    True, print status messages.

    """
    with mpmath.workdps(WORKING_DIGITS):
        with report("angular functions", output):
            pi, tau = mie_reference.pis_and_taus(mpmath.mpmathify(theta),
                                                 nstop)
    return {"theta": repr(theta),
            "nstop": nstop,
            "pi": [mpmath.nstr(pi_n, STORED_DIGITS) for pi_n in pi],
            "tau": [mpmath.nstr(tau_n, STORED_DIGITS) for tau_n in tau]}


def particle_values(m, x, output=True):
    """Calculate reference values for scattering from a particle.  Each
    reference value is associated with [m], [x] specified as lists (per pymie
    convention).  If output is True (default) print status message.

    """
    if len(m) > 1:
        raise NotImplementedError("layered particles not yet supported "
                                  "by reference implementation")

    # use the outermost layer to set the stopping criterion
    nstop = int(mie._nstop(x[-1]))
    with mpmath.workdps(WORKING_DIGITS):
        x_outer = mpmath.mpmathify(x[-1])
        with report("scattering coefficients", output):
            a, b = mie_reference.scatcoeffs(mpmath.mpmathify(m[0]),
                                            mpmath.mpmathify(x[0]), nstop)
        with report("internal coefficients", output):
            c, d = mie_reference.internal_coeffs(mpmath.mpmathify(m[0]),
                                                 mpmath.mpmathify(x[0]), nstop)
        with report("efficiencies", output):
            qsca, qext, qback, g = mie_reference.efficiencies(a, b, x_outer)
        with report("amplitude scattering matrix", output):
            S = [mie_reference.amplitude_scattering_matrix(
                    a, b, mpmath.mpmathify(theta)) for theta in THETAS]
    # store the inputs with repr() so that they convert back to the same
    # numbers when loaded again, and store nstop so that tests can load it;
    # then the tests stay valid even if pymie's stopping criterion changes
    return {"m": [repr(m_layer) for m_layer in m],
            "x": [repr(x_layer) for x_layer in x],
            "nstop": nstop,
            "a": [to_string(an) for an in a],
            "b": [to_string(bn) for bn in b],
            "c": [to_string(cn) for cn in c],
            "d": [to_string(dn) for dn in d],
            "qsca": mpmath.nstr(qsca, STORED_DIGITS),
            "qext": mpmath.nstr(qext, STORED_DIGITS),
            "qback": mpmath.nstr(qback, STORED_DIGITS),
            "g": mpmath.nstr(g, STORED_DIGITS),
            "thetas": [repr(theta) for theta in THETAS],
            "S1": [to_string(S1) for S1, _ in S],
            "S2": [to_string(S2) for _, S2 in S]}


def generate_and_save():
    """Generates the reference values and saves them to JSON.

    """
    # mark progress and time
    t_start = time.perf_counter()

    riccati = []
    for z, nstop in Z_VALUES:
        print(f"generating reference values for z = {z}, nstop = {nstop}",
              flush=True)
        riccati.append(riccati_values(z, nstop))

    angular = []
    for theta in ANGULAR_THETAS:
        print(f"generating reference values for theta = {theta}", flush=True)
        angular.append(angular_values(theta, ANGULAR_NSTOP))

    particles = []
    for m, x in MX_VALUES:
        print(f"generating reference values for m = {m}, x = {x}",
              flush=True)
        t_start_case = time.perf_counter()
        particles.append(particle_values(m, x))
        print(f"completed in {time.perf_counter() - t_start_case:.1f} s",
              flush=True)

    values = {
        "metadata": {
            "date": datetime.date.today().isoformat(),
            "mpmath_version": mpmath.__version__,
            "working_digits": WORKING_DIGITS,
            "stored_digits": STORED_DIGITS,
            "mie_reference_sha256": file_hash(mie_reference.__file__),
            "generator_sha256": file_hash(__file__)
        },
        "riccati": riccati,
        "angular": angular,
        "particles": particles,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(values, f, indent=1)

    print(f"Wrote {OUTPUT_FILE} in {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    generate_and_save()
