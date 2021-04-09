# Fanoos: Multi-Resolution, Multi-Strength, Interactive Explanations for Learned Systems ; David Bayani and Stefan Mitsch ; paper at https://arxiv.org/abs/2006.12453
# Copyright (C) 2021  David Bayani
# 
# This file is part of Fanoos.
# 
# Fanoos is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License only.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# Contact Information:
# 
# Electronic Mail:
#   dcbayani@alumni.cmu.edu
# 
# Paper Mail:
#   David Bayani
#   Computer Science Department
#   Carnegie Mellon University
#   5000 Forbes Ave.
#   Pittsburgh, PA 15213
#   USA
# 
# 

import z3;
from utils.contracts import *;


def quickResetZ3Solver(z3Solver): 
    requires(isinstance(z3Solver, z3.z3.Solver));  # heavily relies on 
        # invariants upheld elsewhere in code.

    if(len(z3Solver.assertions()) > 1):
        raise Exception("Precondition for using this function violated, "+\
            "almost certainly due to a violation to invariants elsewhere in the code.");

    z3Solver.pop(); # return to previous back-tracking point
    assert(len(z3Solver.assertions()) == 0);
    z3Solver.push(); # create new back-tracking  point, necessary
       # for next time pop is called.
    ensures(len(z3Solver.assertions()) == 0);
    return; 
