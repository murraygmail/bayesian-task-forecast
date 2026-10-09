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
Tasks
=============================================================================
Bayesian forecasting of task duration from periodic progress reports.

A `task` starts from a three-point plan (low / det / high periods of effort),
which fixes a Triangular prior on duration and hence a prior on *velocity*
(fraction of the task completed per period).  Each progress report
({'period': t, 'pct': fraction complete}) yields an observed velocity.  The
observed velocities, weighted by the length of the interval they cover, update
a two-parameter (mu, sigma) likelihood grid (see `weighted.likelihood_array`);
the posterior predictive velocity then gives a new distribution for the
periods remaining, and a probability of meeting the planned target.

A `rollup` combines several tasks that must all finish (duration = max of the
task durations).  A task whose plan `depends` on other tasks adds the rollup
of those dependencies to its own duration.

Public API
----------
task(name, plan, like_pdf, depends=[], milestone=False)
    .update(obs, plots=True, dur_plots=True)   add one progress report
    .dur_pdf, .vel_pdf                          current duration / velocity pdfs
    .target()                                   periods left to the planned finish
    .compute_dependancies()                     refresh the dependency rollup
rollup(name, tasks)
    .dur_pdf, .target, .plot()
dur_plot(dur_pdf, target, title='')             duration plot with target shading
trimed_pdf(pdf, trimpct=95)                     pdf with the upper tail removed

Data conventions
----------------
plan : {'low': float, 'det': float, 'high': float}   periods of effort
obs  : {'period': float, 'pct': float}               cumulative, pct in [0, 1]
like_pdf : a likelihood function f(mu, sigma, x) from `Distributions`,
           e.g. Distributions.normal or Distributions.logistic

Created on Wed Oct 11 11:25:18 2023

@author: murraycantor
"""


import sys

module_path = '/Users/murraycantor/Dropbox/python/Service'

if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
#from Service_bpl import Distributions as ds
from Service_bpl import Distributions as ds
from Service_bpl.weighted import likelihood_array
import matplotlib.pyplot as plt
from functools import reduce

nsamps = 10000

class task():
    """
    A single schedule task whose duration is learned from progress reports.

    Attributes of interest after construction / updates
    ---------------------------------------------------
    dur_pdf : Distribution
        Distribution of the periods still needed to finish the task.  Starts
        as Triangular_PDF(low, det, high); after each update it is the
        Empirical_PDF of (fraction remaining) / (velocity samples).
    vel_pdf : Distribution
        Distribution of velocity (fraction complete per period).
    array : weighted.likelihood_array
        The (mu, sigma) grid holding the current priors; replaced after each
        update with the new marginals as priors.
    obs, durs, vels, weights : lists
        Progress reports received, the period of each, the velocity observed
        over the interval ending at each, and the interval-length weights.
    ifigs, efigs : lists of matplotlib figures
        Parameter-learning figures and duration figures, one per update.
    roll, dep_dur_pdf, dep_target
        Set only when `depends` is non-empty: the rollup of the dependencies,
        the combined duration pdf, and the combined target.
    """

    def __init__(self,name, plan,like_pdf,depends=[], milestone = False):
        """
        Parameters
        ----------
        name : str
            Name of the task.
        plan : dict
            Three-point estimate of effort in periods, with keys
            'low', 'det' (deterministic / expected), 'high'.
        like_pdf : callable f(mu, sigma, x)
            Two-parameter likelihood used for Bayesian parameter learning of
            velocity, e.g. Distributions.normal or Distributions.logistic.
        depends : list of task, optional
            Tasks that must finish before this one can finish.  Default [].
        milestone : bool, optional
            Marks a zero-duration milestone.  Default False.  (Stored for
            future use; not yet acted on.)

        Returns
        -------
        None.
        """
        
        self.name = name
        self.det = plan['det'] #Static determinist value of the task level of effort
        self.low = plan['low'] #Static low value of the task level of effort.
        self.high = plan['high'] #Determinist value of the task level of effort
        self.like_pdf = like_pdf #Static model for Bayesian Parameter Learning
        self.pct_comp = 0 #Percent complete of the task reported in update, initially 0
        self.obs = [] # Array of distionaries of the form {durs: ,pct: ,}
        self.durs = [] # Array of periods in obs reported
        self.vels = [] # Array of velocities computed from reported ob
        self.weights = [] #Array of weights for computing paramters computed from reported ob
        self.depends = depends 
        self.ifigs = [] #Array of parameter pdf pyplot figures
        self.efigs = [] #Array of predicted duration pdf pyplot figures
        self.dur_pdf = ds.Triangular_PDF(self.low, self.det, self.high) #Initial level of estmate pdf taken from plan
        self.vel_pdf=mu_prior=ds.Empirical_PDF(1/self.dur_pdf.samples(nsamps)) # Initial prior mu for Bayesian Paramter Learning
        sd_prior = ds.Uniform_PDF(.01, 10*mu_prior.std) #Initial sigma prior 
        self.array = likelihood_array(self.like_pdf,mu_prior, sd_prior) #Initialize BPL array
        if len(depends) > 0:
            self.roll= rollup(self.name, self.depends) #Initialize rollup object for accounting for dependancies
            self.dep_dur_pdf = self.roll.dur_pdf #Depenencies duration PDF
            self.dep_target = self.roll.target+self.det #DEpendencie target PDF
        else:
            self.dep_target = None
            self.roll = None
            self.dep_dur_pdf = None

    def find_weights(self):
        """
        Weight each observation by the period it was reported in, normalised
        to sum to one.  Returns a numpy array aligned with self.vels.
        """
        durs = np.array(self.durs)
        return durs/durs.sum()
    
    def add_obs(self, obs):
        """
        Record a progress report and derive the velocity it implies.

        Parameters
        ----------
        obs : dict
            {'period': t, 'pct': cumulative fraction complete at t}.

        The first report gives velocity pct/period; later reports give the
        incremental velocity (pct - previous pct) / (period - previous period).
        Also refreshes self.durs and self.weights.
        """
        self.obs.append(obs)
        if len(self.obs) == 1:
            vel = obs['pct']/obs['period']
            self.vels.append(vel)
            self.weights.append(1)
            self.durs.append(obs['period'])
            
        else:
            prev = self.obs[-2]
            vel = (obs['pct']-prev['pct'])/(obs['period']-prev['period'])
            self.vels.append(vel)
            self.durs.append(obs['period'])
            self.weights = self.find_weights()
        
    def remaining(self):
        """
        Fraction of the task still to do, 1 - (latest pct complete).
        With no reports yet it returns the planned effort `det` instead.
        """
        if len(self.obs) == 0:
            return self.det
        else:
            return 1-self.obs[-1]['pct']
    
    def target(self):
        """
        Periods left to the planned finish: det minus the latest report period.
        Negative means the plan is already blown.  Requires at least one report.
        """
        return self.det-self.obs[-1]['period']
          

    def update(self, new_obs, plots = True, dur_plots = True):
        """
        Incorporate one progress report and re-forecast the duration.

        Steps: add the observation; update the (mu, sigma) likelihood grid
        with all velocities so far; take the marginals as the new priors;
        draw the posterior predictive velocity; set dur_pdf to the
        distribution of (fraction remaining) / velocity; then, if any
        dependencies, refresh the dependency rollup.

        Parameters
        ----------
        new_obs : dict
            {'period': t, 'pct': cumulative fraction complete at t}.
        plots : bool, optional
            Show the mu / sigma / velocity figure.  Default True.
        dur_plots : bool, optional
            Show the duration figure with the target probability.  Default True.

        Returns
        -------
        None.  Results are left in self.dur_pdf, self.vel_pdf, self.array.
        """
    
        self.add_obs(new_obs)
        
        self.array.update_array(self.vels,self.weights)
        
        priors = self.array.margins()
        
        
        vel_pdf= self.array.pdf()
        self.pct_comp = new_obs['pct']
            
    
        
        self.array= likelihood_array(self.like_pdf, priors[0], priors[1]) #Setting new priors
        
       
        remaining = self.remaining()
        vel_samps = vel_pdf.samples(nsamps)
        self.dur_pdf = ds.Empirical_PDF(remaining/vel_samps)
        self.vel_pdf = vel_pdf
        
        if plots == True: self.gen_plots(priors, vel_pdf)
        if dur_plots:self.dur_plot()
        if len(self.depends) > 0: self.compute_dependancies()
        
    def compute_dependancies(self):
        """
        Combine this task's duration with the rollup of its dependencies.

        Sets self.dep_dur_pdf to the Empirical_PDF of
        (rollup duration samples + own duration samples) and self.dep_target
        to det + rollup target.

        Returns
        -------
        None.
        """
        roll = rollup('', self.depends)
        roll_samps = roll.dur_pdf.samples(nsamps)
        ob_samps = self.dur_pdf.samples(nsamps)
        samps = roll_samps+ob_samps
        self.dep_dur_pdf = ds.Empirical_PDF(samps)
        self.dep_target = self.det + roll.target  # was self.target, which shadowed the target() method
        
        
    
       
        
    
    def gen_plots(self,priors, vel_pdf):
        """
        Three-panel figure: posterior of mu, posterior of sigma, and the
        learned velocity pdf.  Appended to self.ifigs and shown.
        """
        
        ifig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(priors[0].x,priors[0].y)
        ax[0].set_title('mu')
        ax[0].set_xlabel(f'mean = {priors[0].mean:0.2f}, Sd = {priors[0].std:0.2f}')
           
        ax[1].plot(priors[1].x,priors[1].y)
        ax[1].set_title('sigma')
        ax[1].set_xlabel(f'mean = {priors[1].mean:0.2f}, Sd = {priors[1].std:0.2f}')
        
        ax[2].plot(vel_pdf.x ,vel_pdf.y)
        ax[2].set_title(' Learned Velocity')
            #ax[2].set_xlim(left= 0)
        ax[2].set_xlabel(f'mean = {vel_pdf.mean:0.2f}, Sd = {vel_pdf.std:0.2f}')
        ifig.suptitle(f'With {len(self.obs)} samples')
        self.ifigs.append(ifig)
        plt.show()
        
    def dur_plot(self):
        """
        Two-panel figure: a table of the progress reports, and the
        (tail-trimmed) periods-to-complete pdf shaded green up to the target
        and red beyond it, with the probability of meeting the target in the
        x-label.  Appended to self.efigs and shown.
        """
        
        pldur_pdf = trimed_pdf(self.dur_pdf)
        columns = ("Period", "Percent Complete")
        cell_data = [(item['period'],int(item['pct']*100)) for item in self.obs]
                     
        
        # Create a figure and axis
        efig, ax = plt.subplots(1,2,figsize=(10, 4))  
        
        # Remove axis
        ax[0].axis('off')
        
        # Display table
        ax[0].set_title( f'{self.name}\n Best case= {self.low}, Expected = {self.det}, Worst case= {self.high}')
        ax[0].table(cellText=cell_data, colLabels=columns, cellLoc = 'center', loc='center')
        if self.target()> 0:
           try:
            ax[1].set_xlabel(f'Probability of meeting target, {self.target()}, is {self.dur_pdf.cdf(self.target())*100:.0f}%')
           except:
                pass
        else:
            ax[1].set_xlabel('The schedule is blown')
        ax[1].set_title('Periods to Complete')
        x = pldur_pdf.x
        y = pldur_pdf.y
       
        
        ax[1].set_ylim(bottom = 0, top = max(1.1*y))
        ax[1].plot(x,y)
        ax[1].fill_between(x,y, color='red')
        ax[1].fill_between(x, y, where = x <= self.target(), color='green')
        self.efigs.append(efig)
        plt.show()
        
    
       
class rollup:
    """
    A set of tasks that must all finish: duration is the max of the task
    durations (by Monte Carlo over their dur_pdfs) and the target is the max
    of their planned `det` values.

    Attributes
    ----------
    name : str
    tasks : list of task
    target : float
    dur_pdf : Distributions.Empirical_PDF
    """

    def __init__(self,name ,tasks):
        self.name = name
        self.tasks = tasks
        self.target = self.__maxDets()
        self.dur_pdf = self.__durPdf()
        
    def __maxDets(self):
        """
        max  of the tasks dets

        Returns
        -------
        max : float
            
        """
        dets = []
        for task in self.tasks:
            dets.append(task.det)
        return max(dets)
    
    def __durPdf(self):
        """
        Empirical pdf of the element-wise maximum of nsamps draws from each
        task's dur_pdf.
        """
        
        dursamps = []
        for task in self.tasks:
            dursamps.append(task.dur_pdf.samples(nsamps))
        maxsamps = reduce(np.maximum,dursamps)
        durPDF = ds.Empirical_PDF(maxsamps)
        return durPDF
    
    def plot(self):
        """Plot the rollup duration pdf against its target; returns (fig, ax)."""
        names = ''
        for task in self.tasks:
            names = names + f'{task.name} '
        title = 'Rollup of tasks: ' + names
        print(title)
        trimmed = trimed_pdf(self.dur_pdf)
        return(dur_plot(trimmed, self.target, title=title))
        
        
        
    
def dur_plot(dur_pdf,target, title = ''):
    """
    Plot a duration pdf shaded green up to `target` and red beyond, with the
    probability of meeting the target in the x-label.

    Parameters
    ----------
    dur_pdf : Distribution
        Periods-to-complete distribution (plotted as given; the trimmed copy
        computed here is not used).
    target : float
        Periods available.  A non-positive target is labelled as blown.
    title : str, optional

    Returns
    -------
    fig, ax : matplotlib figure and axes.
    """
    pldur_pdf = trimed_pdf(dur_pdf)
    fig, ax = plt.subplots(1,1,figsize=(4, 4)) 
    
    if target> 0:
        ax.set_xlabel(f'Probability of meeting target, {target}, is {dur_pdf.cdf(target)*100:.0f}%')
       
    else:
        ax.set_xlabel('The schedule is blown')
    ax.set_title(title)
    x = dur_pdf.x
    y = dur_pdf.y
   
    
    ax.set_ylim(bottom = 0, top = max(1.1*y))
    ax.plot(x,y)
    ax.fill_between(x,y, color='red')
    ax.fill_between(x, y, where = x <= target, color='green')
    return fig, ax
    
 
def trimed_pdf(pdf,low = -np.inf,high = np.inf, nsamp = 10000, trimpct = 95):
        """
        Resample a pdf with its upper tail removed, for plotting.

        Parameters
        ----------
        pdf : Distribution
        low, high : float, optional
            Accepted for compatibility; `high` is overwritten by the trim.
        nsamp : int, optional
            Number of samples drawn.  Default 10000.
        trimpct : float, optional
            Percent of the distribution kept, from the left.  Default 95.

        Returns
        -------
        Distributions.Empirical_PDF built from the samples below the
        trimpct-th percentile.
        """
        high = pdf.quantile(trimpct / 100)  # quantile() takes p in [0, 1]; trimpct is a percent
        samples = pdf.samples(nsamp)
        mask =(samples < high)
        return ds.Empirical_PDF(samples[mask])
"""   
def time_to_complete(remaining, vel_pdf,target):
        samples = remaining/vel_samps
        samples = trim_samples(samples)
        return ds.Empirical_PDF(samples, bins=30)
"""