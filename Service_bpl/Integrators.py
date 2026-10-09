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
Integrators
=============================================================================
Two fixed-grid Newton-Cotes quadrature rules used by earlier versions of
`Distributions`.  The current Distributions.py no longer imports this
module; it is kept as a small standalone utility.

Simps(f, a, b, n)   composite Simpson's rule
Boole(f, a, b, N)   composite Boole's rule

`eva` and `f` are trivial helpers left from testing (f(x) = x**3).
"""

import numpy as np

def eva(f, x):
    """Evaluate f at x (test helper)."""
    return f(x)

def f(x):
    """Sample integrand x**3 (test helper)."""
    return x**3

def Simps(f,a, b, n=1000):
    """Composite Simpson's rule for the integral of f over [a, b] on n grid points."""
    X= np.linspace(a,b,n)
    n_2 = int(n/2)
    sum = 0
    h= (X[-1]-X[0])/n
    for j in range(1,n_2):
        sum+= f(X[2*j-2])+4*f(X[2*j-1])+f(X[2*j])
    return h*sum/3


def Boole(f,a,b, N=300):
    """Composite Boole's rule for the integral of f over [a, b] on N grid points."""
    sum1 = 0
    x = np.linspace(a,b,N)
    h= x[1]-x[0]
    for n in range(1,N,2):
        sum1 += f(x[n])
    sum2= 0
    for n in range(2,N-3,4):
        sum2 += f(x[n])
    sum3 = 0
    for n in range(4, N-4, 4):
        sum3 += f(x[n])
    sum = 7*(f(x[0])+f(x[-1]))+32*sum1+12*sum2+14*sum3
    factor = 2*h/45
    return factor*sum