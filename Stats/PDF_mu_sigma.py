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
PDF_mu_sigma
=============================================================================
Teaching example: the mean and standard deviation of a continuous pdf by
numerical integration (midpoint Riemann sum).  Standalone; does not use
Service_bpl.

Defines a Normal pdf with mu = 2, sigma = 3 and integrates over [-10, 10] to
recover `mean` and `std` (compare with 2 and 3).

Created on Thu May  2 14:11:29 2024

@author: murraycantor
"""

import numpy as np


def riemann_sum(f, a, b, n=1000):
    """
    Calculates the definite integral of a function using the Riemann sum method.
    
    Parameters:
    - f: The function to integrate (callable).
    - a: The lower limit of integration.
    - b: The upper limit of integration.
    - n: The number of subintervals to use in the Riemann sum approximation.
    
    Returns:
    - The approximate value of the definite integral.
    """
    # Calculate the width of each subinterval
    dx = (b - a) / n
    
    # Initialize the sum
    integral_sum = 0
    
    # Iterate over the subintervals
    for i in range(n):
        # Calculate the midpoint of the subinterval
        x = a + (i + 0.5) * dx
        
        # Evaluate the function at the midpoint and add to the sum
        integral_sum += f(x)
    
    # Multiply the sum by the width of each subinterval to get the approximate integral
    integral = dx * integral_sum
    
    return integral

def normal(mu, sigma, x):
    """
    Calculate the normal probability density function (PDF) for a given x, mean (mu), and standard deviation (sigma).
    
    Parameters:
    - x: The point(s) at which the PDF is evaluated.
    - mu: The mean of the distribution.
    - sigma: The standard deviation of the distribution.
    
    Returns:
    - The value of the PDF at x.
    """
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return prefactor * np.exp(exponent)

def meanf(pdf,a,b, n=1000):
    """Mean of `pdf` over [a, b]: the integral of x * pdf(x)."""
    def m1(x):
        return x*pdf(x)
    return riemann_sum(m1, a, b)

def stdf(pdf,a,b, n=1000):
    """Standard deviation of `pdf` over [a, b]: sqrt of the integral of (mean - x)**2 * pdf(x)."""
    m1 = meanf(pdf,a,b)
    def m2(x):
        return (m1 - x)**2*pdf(x)
    variance= riemann_sum(m2, a, b)
    return np.sqrt(variance) #sqrt = square root
    

mu = 2
sigma = 3

def pdf(x):
    """The example pdf: Normal(mu=2, sigma=3)."""
    return normal(mu, sigma, x)




a = -10
b = 10

mean = meanf(pdf, a, b)
std = stdf(pdf,a,b)


