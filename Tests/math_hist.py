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
math_hist
=============================================================================
Scratch check of Empirical_PDF built from Triangular samples over a
restricted histogram range.

An earlier Empirical_PDF(data, Range=True, hist_range=...) passed the range
to np.histogram, which ignores samples outside it; with the current class
the same effect comes from filtering the samples to the range first.

Created on Sun Nov 19 09:50:57 2023

@author: murraycantor
"""

import matplotlib.pyplot as plt
from Service_bpl import Distributions as ds
import numpy as np

nsamps = 20000

T = ds.Triangular_PDF(3, 5, 16)

data = T.samples(nsamps)

hist_range = (4,12)


in_range = data[(data >= hist_range[0]) & (data <= hist_range[1])]
Test = ds.Empirical_PDF(in_range)

plt.plot(Test.x, Test.y)
plt.show()


