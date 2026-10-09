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
weighted
=============================================================================
Weighted Bayesian parameter learning on a (mu, sigma) grid.

Given a two-parameter likelihood f(mu, sigma, x) and priors on mu and sigma,
`likelihood_array` discretises the parameter space into a dim1 x dim2 grid
and fills each cell with

    prod_i f(mu, sigma, x_i) ** w_i  *  prior_mu(mu) * prior_sigma(sigma)

where x_i are the observed velocities and w_i their (normalised) weights, so
that an observation covering a longer interval counts for more.  The row and
column sums give the marginal posteriors of mu and sigma; sampling those and
drawing Normal(mu, sigma) gives the posterior predictive velocity pdf.

Used by `Tasks.task` with the velocities computed from progress reports.
The module also carries a few small helpers (Simpson integration, histogram
normalisation, weight / velocity construction) kept from earlier versions.

Note: the sys.path block below points at a Dropbox location that no longer
exists; imports now go through the Service_bpl package and the block is inert.

Created on Sun Sep 10 12:32:14 2023

@author: murraycantor
"""

import sys

original_sys_path = list(sys.path)

if '/Users/murraycantor/Library/CloudStorage/Dropbox/python/Service' not in sys.path:
    sys.path.append( '/Users/murraycantor/Library/CloudStorage/Dropbox/python/Service')
   
from Service_bpl import Distributions as ds
#import Integrators as intg
import numpy as np
from scipy.interpolate import interp1d


def Simps(f,a, b, n=1000):
    """Composite Simpson's rule for the integral of f over [a, b] with n points."""
    X= np.linspace(a,b,n)
    n_2 = int(n/2)
    sum = 0
    h= (X[-1]-X[0])/n
    for j in range(1,n_2):
        sum+= f(X[2*j-2])+4*f(X[2*j-1])+f(X[2*j])
    return h*sum/3


nsamps = 1000

def weighted_likeli(like_funct,param1,parm2, data, weights):
    """
    Weighted likelihood of `data` at parameters (param1, parm2):
    the product over i of like_funct(param1, parm2, data[i]) ** weights[i].
    """
    prod_array = []
    for i in range(len(data)):
        prod_array.append(like_funct(param1,parm2,data[i])**weights[i])
    return np.array(prod_array).prod()




def normalize(hist):
     """
     Scale a histogram (x, y) so it integrates to one over its x range.
     Returns the pair (x, y / area).
     """
     function = interp1d(hist[0], hist[1])
     area = Simps(function, min(hist[0]), max(hist[0]))
     return (hist[0],hist[1]/area)
 
def find_weights(durs):
    """Normalise interval lengths to weights summing to one."""
    durs = np.array(durs)
    return durs/durs.sum()

def find_vels(comps,durs):
    """Velocities = fraction completed / interval length (element-wise)."""
    return comps/durs

def build_data(vels, durs):
    """
    Cumulative (velocities, weights) pairs: entry n holds the first n+1
    velocities and the weights of the first n+1 intervals.  Used to replay a
    sequence of observations.
    """
    w_data = []
    for n in range(len(vels)):
        datum = [vels[:n+1],find_weights(durs[:n+1])]
        w_data.append(datum)
    return w_data
        


  

    


class likelihood_array():
    """
    Discretised joint posterior of (parm1, parm2) for a two-parameter
    likelihood, with weighted observations.

    Typical use (see Tasks.task.update):
        arr = likelihood_array(like_pdf, mu_prior, sigma_prior)
        arr.update_array(velocities, weights)
        mu_post, sigma_post = arr.margins()
        vel_pdf = arr.pdf()
    """

    def __init__(self,like_pdf, parm1, parm2, dim1=50, dim2=50, nsamps = 10000):
        """
        Parameters
        ----------
        like_pdf : callable f(parm1, parm2, x)
            Likelihood P(x | parm1, parm2).
        parm1 : Distribution
            Prior of the first parameter (mu).  Its .x grid sets the range.
        parm2 : Distribution
            Prior of the second parameter (sigma).  Its .x grid sets the range.
        dim1, dim2 : int, optional
            Grid resolution for parm1 and parm2.  Default 50 x 50.
        nsamps : int, optional
            Samples drawn when building the predictive pdf.  Default 10000.

        Returns
        -------
        None.
        """
        self.like_pdf = like_pdf
        self.parm1 = parm1
        self.parm2 = parm2
        self.array = np.zeros(shape = (dim1,dim2))
        self.dim1 = dim1
        self.dim2 = dim2
        self.nsamps = nsamps
        self.weights = []
        self.data = []
        self.weights = []
        
        
        
    def update_array(self, data, weights):
            """
            Fill the grid with weighted likelihood x priors for every
            (parm1, parm2) cell.

            Parameters
            ----------
            data : sequence of float
                Observations (velocities).
            weights : sequence of float
                One weight per observation; normalised here to sum to one.
                Note the un-normalised `weights` are what is passed to
                weighted_likeli; the normalised copy is kept in self.weights.
            """
            self.data = data
            self.weights = np.array(weights)/sum(weights)
            #print(data)
            #print(self.weights)
            p1x = self.parm1.x
            self.p1_index = np.linspace(min(p1x),max(p1x),self.dim1)
            p2x = self.parm2.x
            self.p2_index = np.linspace(min(p2x),max(p2x),self.dim2)
            i = 0
            for p1 in self.p1_index:
             
                j = 0
                for p2 in self.p2_index:
                        #print(self.like_funct(p1,p2,data),self. parm1.pdf(p1),self.parm2.pdf(p2))
                        cell =  weighted_likeli(self.like_pdf,p1,p2,data,weights)*self.parm1.pdf(p1)*self.parm2.pdf(p2)
                        #print(i,j,cell)
                        self.array[i,j] = cell
                        j += 1
                i +=1
   
    
    def margins(self):
        """
        Marginal posteriors of parm1 and parm2 (row and column sums of the
        grid, normalised), returned as a pair of Distributions.ArrayPDF.
        """
        m1 = normalize((self.p1_index, np.sum(self.array,axis= 1)))
        m2 = normalize((self.p2_index, np.sum(self.array,axis= 0)))

        parm1 = ds.ArrayPDF(*m1)
        parm2 = ds.ArrayPDF(*m2)
        return parm1, parm2
    
    def pdf(self):
        """
        Posterior predictive pdf: draw (parm1, parm2) pairs from the
        marginals, draw Normal(parm1, parm2) for each, keep the positive
        draws, and return their Empirical_PDF.
        """
        parm1, parm2 = self.margins()
        parm1_samps=parm1.samples(self.nsamps)
        parm2_samps=parm2.samples(self.nsamps)
        arr = np.random.normal(parm1_samps,parm2_samps)
        pdf_samps = arr[arr > 0]
        return ds.Empirical_PDF(pdf_samps)
    
