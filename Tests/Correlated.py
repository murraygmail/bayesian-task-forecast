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
Correlated
=============================================================================
Experiment: propagate a percentage increase in an independent cost to
dependent costs through a correlation coefficient.

Builds triangular cost pdfs around a mode using golden-ratio (or 1.2 / 1.8)
spreads, plots them, applies a 20% increase to the independent cost and a
correlation-scaled increase to each dependent, and plots the results.

Run from the project root:
    PYTHONPATH=. python3 Tests/Correlated.py

Created on Sat Dec  2 09:31:43 2023

@author: murraycantor
"""

from Service_bpl import Distributions as ds
import numpy as np
import matplotlib.pyplot as plt

phi = ((1+np.sqrt(5))/2)

def tri_cost(mode:float, opt= 'average', pess = 'average'):
    """
    Triangular cost pdf around `mode`.

    pess : 'low' (high = 1.2 * mode), 'high' (1.8 * mode), else phi * mode
    opt  : 'low' (low = mode / 1.2), 'high' (mode / 1.8), else mode / phi
    """
    
    if pess == 'low': pfact = 1.2
    elif pess == 'high': pfact = 1.8
    else: pfact = phi
    if opt == 'low': ofact = 1/1.2
    elif opt == 'high': ofact = 1/1.8
    else: ofact = 1/phi
    return ds.Triangular_PDF(ofact*mode, mode, mode*pfact)

def plot_pdf(pdf, title = ''):
    """Plot pdf.y against pdf.x with the mean in the x-label, and show."""
    plt.plot(pdf.x, pdf.y)
    plt.title(title)
    plt.xlabel(f'mu = {pdf.mean:.2f}')
    plt.show()


def update_cost(ind, deps, pct_inc, cor):
    """
    

    Parameters
    ----------
    ind :PDF object
        Indepedent variable of linear relationship
    deps : Array of PDF objects
        Dependent variable of linear realtionship
    pct_inc : float
        percent increase of independent variable
    cor : float between -1 and 1
        correlation coefficient 

    Returns
    -------
    new_ind : Empirical_PDF
        `ind` scaled by (1 + pct_inc/100).
    new_deps : list of Empirical_PDF
        Each dependent scaled by (1 + pct_inc/100) * cor.

    """
    ind_samps = (1+pct_inc/100)*ind.samples(nsamps)
    new_ind = ds.Empirical_PDF(ind_samps)
    new_deps = []
    for dep in deps:
        dep_samps = (1+pct_inc/100)*cor*dep.samples(nsamps)
        new_deps.append(ds.Empirical_PDF(dep_samps))
    return new_ind, new_deps
    
    

nsamps = 50000
corr = 1


ind = tri_cost(8)
dep1 = tri_cost(13, pess='high', opt = 'low')
dep2 = tri_cost(13, opt = 'low')
dep3 = ds.Logisitic_PDF(10, 2)

plot_pdf(ind, title='ind')
plot_pdf(dep1, title='dep1')
plot_pdf(dep2, title='dep2')
plot_pdf(dep3, title='dep3')

deps = [dep1, dep2, dep3]

pct_inc = 20

new_ind, new_deps = update_cost(ind, deps, pct_inc, corr)
plot_pdf(new_ind, title='new_ind')
n = 1
for new_dep in new_deps:
    plot_pdf(new_dep, title=f'new_dep{n}')
    n +=1












