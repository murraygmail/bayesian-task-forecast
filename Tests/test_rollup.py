#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022-2026 Murray Cantor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
test_rollup
=============================================================================
Smoke test for `Tasks.rollup` with a logistic velocity likelihood.

Builds two tasks, gives each one progress report, rolls them up (duration =
max of the two) and plots the rollup duration against its target.

Run from the project root:
    PYTHONPATH=. python3 Tests/test_rollup.py

Created on Sun Nov 12 12:18:29 2023

@author: murraycantor
"""


    
from Service_bpl import Distributions as ds
import matplotlib.pyplot as plt
from Service_bpl import Tasks
import time


### Main Program ###





like_pdf = ds.logistic
start_time = time.time()

t1plan = {'low':4, 'det': 6, 'high':9}

task1 = Tasks.task('Task1',t1plan, like_pdf)
t1p1 = {'period':1, 'pct': .10} 
t1p2 = {'period':3, 'pct': .40} 
t1p3 = {'period':4, 'pct': .80} 
t1p4 = {'period':5, 'pct': .95}
obs1 = [t1p1]
for obs in obs1:
    task1.update(obs, plots=False)

t2plan = {'low':10, 'det': 13, 'high':19}

task2 = Tasks.task('Task2',t2plan, like_pdf)
t2p1 = {'period':1, 'pct': .10} 
t2p2 = {'period':3, 'pct': .30} 
t2p3 = {'period':4, 'pct': .50} 
t2p4 = {'period':5, 'pct': .95}
obs2 = [t2p1]
for obs in obs2:
    task2.update(obs, plots=False)

roll = Tasks.rollup('R1', [task1, task2])


fig= roll.plot()
plt.show()

    
