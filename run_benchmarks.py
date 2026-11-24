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
Convenience script to run asv benchmarks and print a nicely formatted summary
(asv does not have a way to format command-line output, at least as of v0.6)

Usage (from repository root): python run_benchmarks.py

"""
import subprocess
import sys
import ast

import numpy as np
import pandas as pd

import asv.results
from asv.config import Config


CONFIG_FILE = "asv.conf.json"
# relative index of weakly absorbing sphere (should match value in benchmarks)
WEAKLY_ABSORBING_M = 1.59/1.33 + 1e-3j


def run_asv():
    """Run benchmarks on HEAD.  Results are saved to asv's JSON file.

    """
    asv_run_command = ["asv", "run", "HEAD^!", "--show-stderr"]
    # config file sets launch_method to "forkserver" but this won't work on
    # Windows. Fall back to spawning a new process for each benchmark on
    # Windows.
    if sys.platform == "win32":
        asv_run_command += ["--launch-method", "spawn"]
    print("running " + " ".join(asv_run_command))
    # run asv, printing progress and errors to the terminal
    status = subprocess.run(asv_run_command).returncode
    if status != 0:
        # error running asv, or a benchmark failed
        print(f"asv exited with status {status}")
    return status


def load_results():
    """Load asv's results.
    """
    conf = Config.load(CONFIG_FILE)
    all_results = list(asv.results.iter_results(conf.results_dir))
    # search through all benchmarks in all result files to find the one that is
    # most recent
    # TODO: change this to find the commit hash for HEAD and load the file
    # directly, which will be more efficient than searching
    results = max(all_results,
                  key=lambda r: max(r.started_at.values(), default=0))

    return results


def asv_string_to_value(asv_string):
    """Utility function to convert a string from ASV to a value. Converts
    one-element tuples (from pymie) to scalars.

    """
    value = ast.literal_eval(asv_string)
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return value


def get_timings(results, benchmark, param_names):
    """Gets timings from asv Results object for a particular benchmark and
    return a pd.Series of the values, indexed by the parameter names.

    """
    params = results.get_result_params(benchmark)
    # convert to values before constructing index for DataFrame
    levels = [[asv_string_to_value(p) for p in string] for string in params]
    index = pd.MultiIndex.from_product(levels, names=param_names)
    # pd.Series will work here even for 2D data because we've defined a
    # MultiIndex
    return pd.Series(results.get_result_value(benchmark, params), index=index,
                     dtype=float)


def print_results(results):
    """
    Print results for selected benchmarks with consistent units.
    """
    pymie_result = get_timings(
        results,
        "benchmarks.MieCalculations.time_calc_cross_sections",
        ["m", "x"])
    miepython_result = get_timings(
        results,
        "miepython_reference.MiepythonCalculations.time_efficiencies",
        ["m", "x"])
    # we'll look only at the weakly absorbing homogeneous sphere in the
    # nonabsorbing medium.
    pymie_x_result = pymie_result.xs(WEAKLY_ABSORBING_M, level="m")
    miepython_x_result = miepython_result.xs(WEAKLY_ABSORBING_M, level="m")
    # Remove values for absorbing medium from pymie results (these are not in
    # the miepython results)
    pymie_x_result = pymie_x_result[np.imag(pymie_x_result.index) == 0]
    # convert index to real
    pymie_x_result.index = pd.Index(np.real(pymie_x_result.index), name="x")

    print("Cross-section evaluation times (ms) "
          "as a function of size parameter")
    print("pymie")
    print(pymie_x_result*1e3)
    print("miepython")
    print(miepython_x_result*1e3)


if __name__ == "__main__":
    status = run_asv()
    if status != 0:
        # if there's an error, exit and pass asv's exit status to the caller
        sys.exit(status)
    results = load_results()
    print_results(results)

