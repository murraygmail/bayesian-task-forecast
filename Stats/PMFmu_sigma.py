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
PMFmu_sigma
=============================================================================
Teaching example: the mean and standard deviation of a discrete PMF.
Standalone; does not use Service_bpl.

Builds the binomial PMF for the number of heads in 10 tosses of a fair coin
(b = 0.5) and a biased coin (b = 0.3), computes mean and std from the PMF,
and shows a bar chart of each.

Created on Thu May  2 13:31:30 2024

@author: murraycantor
"""

import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    """n! by recursion."""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def comb(n, k):
    """Binomial coefficient n choose k."""
    numerator = factorial(n)
    denominator = factorial(k) * factorial(n - k)
    return numerator // denominator

def binomial(b,k, n):
    """
       the probability of K favorable events with n Bernoulli trails with bias w.

       Parameters
       ----------
       b: bias, array of values between 0 and 1
       k: number of heads
       n: number of flips

       Returns
       -------
       """
    b = np.array(b)
    p = (b ** k) * (1 - b) ** (n - k)
    p = comb(n,k)*p
    return p


def PMF_mean(x,y):
    """
    Parameters
    ----------
    x : array of n floats
        Range of PMF
    y : array of n floats
       y = P(x).

    Returns
    -------
   float:
        the mean

    """
    return np.sum(x * y)


def PMF_std(x,y):
     """
     Parameters
     ----------
     x : array of n floats
         Range of PMF
     y : array of n floats
        y = P(x).

     Returns
     -------
    float:
         the std

     """
     mean = PMF_mean(x, y)
     variance = np.sum((x- mean) ** 2 * y)
     return np.sqrt(variance) #sqrt = square root

"""Testing PMF"""
    
n = 10
b = .5

fair = []
biased = []

values = range(0, n+1)

for k in values:
    fair.append(binomial(.5, k, n))
    biased.append(binomial(.3, k, n))
    


fair_mean = PMF_mean(np.array(values), fair)
fair_std = PMF_std(np.array(values), fair)

biased_mean = PMF_mean(np.array(values), biased)
biased_std = PMF_std(np.array(values), biased)

plt.bar(values, fair, color = 'k', alpha = 0.8)
plt.xlabel(f'Number of Heads, mean = {fair_mean}, std = {fair_std:0.4f}')
plt.xticks(values)
plt.ylabel('Probability')
plt.title('PMF for Number of Heads in 10 Tosses of a Fair Coin')

# Display the plot
plt.show()  

plt.bar(values, biased, color = 'k', alpha = 0.8)
plt.xlabel(f'Number of Heads, mean = {biased_mean:0.4f}, std = {biased_std:0.4f}')
plt.xticks(values)
plt.ylabel('Probability')
plt.title('PMF for Number of Heads in 10 Tosses of a Biased Coin')

# Display the plot
plt.show()  