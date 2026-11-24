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
Convenience script to run asv benchmarks and (eventually) print a nicely
formatted summary (asv does not have a way to format command-line output, at
least as of v0.6)

Usage (from repository root): python run_benchmarks.py

"""
import subprocess
import sys


def run_asv():
    """Run benchmarks on HEAD.  Results are saved to asv's JSON file

    """
    asv_run_command = ["asv", "run", "HEAD^!", "--show-stderr"]
    print("running " + " ".join(asv_run_command))
    # run asv, printing progress and errors to the terminal
    status = subprocess.run(asv_run_command).returncode
    if status != 0:
        # error running asv, or a benchmark failed
        print(f"asv exited with status {status}")
    return status


if __name__ == "__main__":
    # running with sys.exit() will pass asv's exit status to the caller
    sys.exit(run_asv())
