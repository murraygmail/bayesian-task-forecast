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
test_task
=============================================================================
Smoke test for a single `Tasks.task` with a Normal velocity likelihood.

Builds Task1 from the plan {low 4, det 6, high 9}, feeds it four progress
reports (10%, 10%, 10%, 15% complete at periods 1, 3, 4, 5), and shows the
parameter-learning and duration figures after each.  Prints the elapsed time.

Run from the project root (Service_bpl must be importable):
    PYTHONPATH=. python3 Tests/test_task.py
In Spyder, run with the project folder as the working directory.

Note: two of the four observed velocities are zero, so this is a deliberately
pessimistic case; the probability of meeting the target comes out near 0.

Created on Tue Nov  7 10:32:20 2023

@author: murraycantor
"""






### Main Program ###


from Service_bpl import Distributions as ds
#import matplotlib.pyplot as plt
from Service_bpl import Tasks
import time

nsamps = 10000

like_pdf = ds.normal
start_time = time.time()
t1plan = {'low':4, 'det': 6, 'high':9}

task1 = Tasks.task('Task1',t1plan, like_pdf)

"""

plt.plot(task1.dur_pdf.x, task1.dur_pdf.y)
plt.show()
plt.plot(task1.vel_pdf.x,task1.vel_pdf.y)
"""

t1p1 = {'period':1, 'pct': .10} 
t1p2 = {'period':3, 'pct': .10} 
t1p3 = {'period':4, 'pct': .10} 
t1p4 = {'period':5, 'pct': .15}
obs1 = [t1p1,t1p2,t1p3,t1p4]
for obs in obs1:
    task1.update(obs)
    
end_timr = time.time()
"""

t2plan = {'low':4, 'det': 6, 'high':9}

task2 = Tasks.task('Task2',t1plan, like_pdf,depends=[task1])
"""
print(f'{end_timr-start_time:0.1f} seconds')