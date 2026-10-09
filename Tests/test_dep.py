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
test_dep
=============================================================================
Smoke test for task dependencies.

A base task (plan 10 / 15 / 25) depends on two tasks Dep1 and Dep2.  The
script plots the base task's duration including its dependencies, then
updates Dep1 with two progress reports, recomputes the dependencies and
plots again.  The commented block at the end is an older two-task
convolution experiment kept for reference.

Run from the project root:
    PYTHONPATH=. python3 Tests/test_dep.py

Note: the sys.path block below points at a Dropbox location that no longer
exists; it is inert now that imports go through the Service_bpl package.

Created on Mon Sep 18 08:24:19 2023

@author: murraycantor
"""
import sys
from Service_bpl import Distributions as ds
import matplotlib.pyplot as plt
from Service_bpl import Tasks as tks 




def plot_roll(pdf, target, title = ''):
    """Trim the pdf, plot it against the target with Tasks.dur_plot, and show."""
    fig, ax= tks.dur_plot(tks.trimed_pdf(pdf),target)
    ax.set_title(title)
    plt.show()
    
### Main Program ###


"""
Initial task with two dependent tasks
    Initalize all three
    update each
compute independent duration pdf and target
plot 

compute independent duration pdf and target
plot

run variants

"""

original_sys_path = list(sys.path)

if '/Users/murraycantor/Library/CloudStorage/Dropbox/python/Service' not in sys.path:
    sys.path.append( '/Users/murraycantor/Library/CloudStorage/Dropbox/python/Service')

nsamps = 10000

like_pdf = ds.logistic


bplan = {'low':10, 'det': 15, 'high':25} #base plan
d1plan = {'low':4, 'det': 6, 'high':9}  #dependacy plan 1
d2plan = {'low':8, 'det': 12, 'high':15} #dependacy plan 1

#Inital each tasks


d1task = tks.task('Dep1', d1plan, like_pdf)
d2task = tks.task('Dep2',d2plan, like_pdf)
btask = tks.task('Base',bplan,like_pdf,depends=[d1task,d2task])
btask.compute_dependancies()

plot_roll(tks.trimed_pdf(btask.dep_dur_pdf), btask.dep_target,title=f'Task {btask.name} with Dependancies')





    

# Updates

d1p1 = {'period':1, 'pct': .10}
d1p2 = {'period':3, 'pct': .30} 
d1p3 = {'period':4, 'pct': .60} 
d1p4 = {'period':5, 'pct': .80}
d1obs = [d1p1,d1p2]

for obs in d1obs:
    d1task.update(obs)
btask.compute_dependancies()

plot_roll(tks.trimed_pdf(btask.dep_dur_pdf), btask.dep_target,title=f'Task {btask.name} with updated dependancies')

    
"""  

    
t2p1 = {'period':1, 'pct': .01} 
t2p2 = {'period':5, 'pct': .50} 
t2p3 = {'period':6, 'pct': .50} 
t2p4 = {'period':7, 'pct': .80}
obs2 = [t2p1, t2p2]
for obs in obs2:

    vwa2, dur_pdf2, new_pdf2= update(vwa2,obs)



# Record the start time


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"The routine took {elapsed_time} seconds to complete.")

jsamps= dur_pdf1.samples(nsamps)+dur_pdf2.samples(nsamps)
j_pdf = ds.Empirical_PDF(jsamps)
plt.plot(j_pdf.x,j_pdf.y)
plt.show()
"""

