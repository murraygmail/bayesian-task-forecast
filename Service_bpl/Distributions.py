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
Distributions
=============================================================================
A small, dependency-light library of probability distributions with a common
interface. Every distribution exposes:

    .pdf(x)        probability density at x
    .cdf(x)        cumulative probability P(X <= x)
    .inv_cdf(p)    quantile (inverse CDF), p in [0, 1]
    .quantile(p)   same as inv_cdf, returns a float; p in [0, 1]   (canonical)
    .samples(n)    n random draws
    .confidence(x) survival P(X > x) = 1 - cdf(x)
    .event((a,b))  P(a <= X <= b)
    .mean, .std    summary statistics
    .x, .y, .cy    a plotting grid, the pdf on it, and the cdf on it

Parametric families build their CDF/quantile in closed form (fast, exact);
Empirical_PDF builds them from a sample by histogram + interpolation.

CONVENTIONS / GOTCHAS (kept for backward compatibility, documented here):
  * `quantile(p)` is the canonical inverse-CDF and ALWAYS takes p in [0, 1].
    Prefer it in new code.
  * `percentile()` is inconsistent by class for historical reasons:
        - Empirical_PDF, Normal_PDF, Triangular_PDF : argument in [0, 1]
        - the base class (used by Lognormal_PDF, Uniform_PDF, Logisitic_PDF) :
          argument in [0, 100]  (it divides by 100 internally)
    Use `quantile()` to avoid the ambiguity.
  * `mode` is a *method* on the base class but an *attribute* on Triangular_PDF.

This revision removes the old dependency on Services.Integrators and the
removed-in-SciPy-1.14 calls (simps / cumtrapz), vectorizes the CDF/inverse-CDF
construction (was O(n^2)), and documents every public name. The public API and
numerical behavior are preserved.
=============================================================================
"""

from functools import cached_property
from math import comb, gamma
import numpy as np
import scipy.interpolate as intp
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv, owens_t

# cumulative_trapezoid moved around across SciPy versions; support both.
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except ImportError:                                   # SciPy < 1.6
    from scipy.integrate import cumtrapz as _cumtrapz

# np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

SQRT2 = np.sqrt(2.0)


# ----------------------------------------------------------------------------
# Internal helpers (standard-normal CDF / inverse, grid-based CDF builder)
# ----------------------------------------------------------------------------
def _Phi(z):
    """Standard-normal CDF, vectorized."""
    return 0.5 * (1.0 + erf(np.asarray(z, float) / SQRT2))


def _Phi_inv(p):
    """Standard-normal inverse CDF (quantile), vectorized."""
    return SQRT2 * erfinv(2.0 * np.asarray(p, float) - 1.0)


def _cdf_inv_from_grid(x, y):
    """
    Build (cdf, inv_cdf) interpolators from a pdf sampled on a grid.

    Vectorized O(n) replacement for the old per-point integration loop.

    Parameters
    ----------
    x : 1-D array, increasing support points
    y : 1-D array, pdf values at x

    Returns
    -------
    (cdf_func, inv_cdf_func) : callables built with linear interpolation
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = _cumtrapz(y, x, initial=0.0)
    c = c / c[-1]                                     # normalize to [0, 1]
    cdf_func = interp1d(x, c, kind="linear", bounds_error=False,
                        fill_value=(0.0, 1.0))
    cu, idx = np.unique(c, return_index=True)         # strictly increasing for inverse
    inv_cdf_func = interp1d(cu, x[idx], kind="linear", bounds_error=False,
                            fill_value=(x[0], x[-1]))
    return cdf_func, inv_cdf_func


# ============================================================================
# PDF helper functions  (plain formulas; handy on their own)
# ============================================================================
def normal(mu, sigma, x):
    """Normal pdf N(mu, sigma) evaluated at x."""
    prefactor = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return prefactor * np.exp(-0.5 * ((np.asarray(x, float) - mu) / sigma) ** 2)


def skew_normal(xi, omega, alpha, x):
    """
    Skew-normal pdf SN(xi, omega, alpha) at x: location xi, scale omega, shape
    alpha. alpha = 0 reduces to the Normal pdf N(xi, omega).
    """
    z = (np.asarray(x, float) - xi) / omega
    return (2.0 / omega) * normal(0.0, 1.0, z) * _Phi(alpha * z)


def lognormal(mu, sigma, x):
    """Lognormal pdf at x (parameters mu, sigma of the underlying normal)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos] if out.ndim else (x if x > 0 else 1.0)
    if out.ndim:
        out[pos] = (np.exp(-((np.log(xp) - mu) ** 2) / (2 * sigma ** 2))
                    / (xp * sigma * np.sqrt(2 * np.pi)))
        return out
    return (np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
            / (x * sigma * np.sqrt(2 * np.pi))) if x > 0 else 0.0


def triangular(low, expected, high, x):
    """
    Triangular pdf with support [low, high] and mode `expected`, at x.
    Degenerate cases (mode equal to an endpoint) are nudged to stay finite.
    """
    x = np.asarray(x, float)
    a, c, b = low, expected, high
    if c == b:
        b = 1.001 * b
    if c == a:
        a = 0.999 * a                                  # (was a no-op '==' in the original)
    pdf = np.zeros(x.shape)
    left = (a <= x) & (x < c)
    pdf[left] = 2 * (x[left] - a) / ((b - a) * (c - a))
    pdf[x == c] = 2 / (b - a)
    right = (c < x) & (x <= b)
    pdf[right] = 2 * (b - x[right]) / ((b - a) * (b - c))
    return pdf


def logistic(mu, s, x):
    """Logistic pdf with location mu and scale s, at x (overflow-safe)."""
    z = -(np.asarray(x, float) - mu) / s
    ez = np.exp(z)
    return ez / (s * np.square(1 + ez))


def weibull(l, k, x):
    """Weibull pdf with scale l and shape k, at x >= 0."""
    x = np.asarray(x, float)
    return (k / l) * (x / l) ** (k - 1) * np.exp(-((x / l) ** k))


def geometric(p, k):
    """Geometric pmf: probability of k failures before the first success."""
    return p * (1 - p) ** k


def binomial(b, k, n):
    """Binomial pmf: P(k successes in n trials) for bias b (scalar or array)."""
    b = np.asarray(b, float)
    return comb(n, k) * (b ** k) * (1 - b) ** (n - k)


def Bernoulli(b, k, n):
    """Likelihood (b^k)(1-b)^(n-k) for bias b (scalar or array); no binom factor."""
    b = np.asarray(b, float)
    return (b ** k) * (1 - b) ** (n - k)


def bernoulli(p, k, n):
    """Unnormalized Bernoulli likelihood over a discretized grid p (sandbox helper)."""
    p = np.asarray(p, float)
    return (p ** k) * (1 - p) ** (n - k) * len(p)


def lognormal_mean_std_dev(mu, sigma):
    """Closed-form mean and std of a lognormal from (mu, sigma)."""
    mean = np.exp(mu + 0.5 * sigma ** 2)
    std = mean * np.sqrt(np.exp(sigma ** 2) - 1)
    return mean, std


# ============================================================================
# Utility functions on an arbitrary pdf callable  (legacy; kept for callers)
# ============================================================================
def cdf(P, a, x, n=1000):
    """Numerically integrate pdf P from a to x (trapezoid, n points)."""
    t = np.linspace(a, x, n)
    return _trapz(P(t), t)


def inverse_cdf(P, a, b, p, tol=1e-6):
    """Bisection inverse-CDF of pdf P on [a, b] for probability p."""
    left, right = a, b
    while right - left > tol:
        mid = (left + right) / 2
        if cdf(P, a, mid) - p < 0:
            left = mid
        else:
            right = mid
    return right


def find_minmax(P, a, b, edge=0.001):
    """Return the (edge, 1-edge) quantiles of pdf P searched on [a, b]."""
    return inverse_cdf(P, a, b, edge), inverse_cdf(P, a, b, 1 - edge)


# ============================================================================
# Base class
# ============================================================================
class Distribution:
    """
    Base class for the parametric distributions below.

    Subclasses set, in __init__: self.x, self.y (pdf on x), self.cy (cdf on x),
    self.mean, self.std, and provide pdf/cdf/inv_cdf. This base supplies the
    shared sampling, quantile, confidence and event helpers.
    """

    # ---- random sampling -------------------------------------------------
    def samples(self, number=1, random_seed=None):
        """
        Draw `number` samples by inverse-transform sampling.

        Vectorized (one uniform draw of size `number` through inv_cdf). Using a
        seed reproduces the legacy stream exactly. Pass random_seed=None (default)
        to leave the global RNG untouched.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        u = np.random.uniform(0.0, 1.0, number)
        return np.asarray(self.inv_cdf(u), float)

    # ---- quantiles -------------------------------------------------------
    def quantile(self, p):
        """Canonical inverse CDF; p in [0, 1]; returns a float."""
        assert 0.0 <= p <= 1.0, "p must be between 0 and 1"
        return float(self.inv_cdf(p))

    def percentile(self, val):
        """
        LEGACY base behavior: `val` is a PERCENT in [0, 100] (divides by 100).
        Subclasses Empirical/Normal/Triangular override this to take [0, 1].
        Prefer quantile(p) for new code.
        """
        return float(self.inv_cdf(val / 100))

    def find_confidence(self, pct):
        """Value x such that P(X <= x) = pct/100 (pct a percentage)."""
        p = pct / 100
        assert 0 < p < 1
        return self.inv_cdf(p)

    # ---- probabilities ---------------------------------------------------
    def confidence(self, p):
        """Survival function: P(X > p) = 1 - cdf(p)."""
        return 1 - self.cdf(p)

    def event(self, event):
        """P(a <= X <= b) for event = (a, b); returns 0 if event is None."""
        if event is None:
            return 0
        a, b = event
        assert a <= b
        return self.cdf(b) - self.cdf(a)

    # ---- shape -----------------------------------------------------------
    def mode(self):
        """Grid mode: the x where the sampled pdf y is largest."""
        return self.x[int(np.argmax(self.y))]

    @property
    def min_domain(self):
        return self.low

    @property
    def max_domain(self):
        return self.high

    @cached_property
    def mean(self):
        # Triangular-style default; every concrete subclass sets self.mean,
        # which shadows this. Kept only for backward compatibility.
        return (self.low + self.expected + self.high) / 3

    def cdf_funcs(self):
        """
        Build (cdf, inv_cdf) interpolators from self.x and self.pdf.
        Vectorized O(n) (replaces the old per-point integration loop).
        """
        return _cdf_inv_from_grid(self.x, self.pdf(self.x))


# ============================================================================
# Array-based pdf (from x, y pairs)
# ============================================================================
class ArrayPDF(Distribution):
    """
    A pdf defined by (x, y) sample points: interpolates, normalizes, and
    provides cdf / inverse-cdf / sampling / mean / std.

    Inherits the shared Distribution interface (quantile, confidence, event,
    mode, find_confidence); percentile is overridden to take p in [0, 1] so it
    matches Empirical_PDF, the other data-driven class.
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.f = intp.interp1d(self.x, self.y, fill_value="extrapolate",
                               bounds_error=False)
        self.norm_fact = _trapz(self.f(self.x), self.x)
        self.norm_y = self.f(self.x) / self.norm_fact
        self.cdf, self.inv_cdf = _cdf_inv_from_grid(self.x, self.norm_y)
        self.mean = self._compute_mean()
        self.std = self._compute_std()
        self.cy = self.cdf(self.x)

    def pdf(self, p):
        """Normalized pdf at p."""
        return self.f(p) / self.norm_fact

    def samples(self, number=1):
        """Inverse-transform samples."""
        return self.inv_cdf(np.random.uniform(0.0, 1.0, number))

    def percentile(self, p):
        """Quantile with p in [0, 1] (matches Empirical_PDF)."""
        assert 0 <= p <= 1, "p must be between 0 and 1"
        return float(self.inv_cdf(p))

    def _compute_mean(self):
        return _trapz(self.x * self.norm_y, self.x)

    def _compute_std(self):
        var = _trapz((self.x - self.mean) ** 2 * self.norm_y, self.x)
        return np.sqrt(var)


# ============================================================================
# Parametric distributions
# ============================================================================
class Normal_PDF(Distribution):
    """Normal (Gaussian) distribution N(mean, std). Closed-form CDF/quantile."""

    def __init__(self, mean, std):
        assert std >= 0
        self.mean = float(mean)
        self.std = float(std)
        self.minx, self.maxx = self.inv_cdf(0.001), self.inv_cdf(0.999)
        self.x = np.linspace(self.minx, self.maxx, 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.title_pdf = 'Normal Distribution for mean = %g, std = %g' % (self.mean, self.std)
        self.title_cdf = 'Normal Cumulative Distribution for mean = %g, std = %g' % (self.mean, self.std)

    def pdf(self, x):
        return normal(self.mean, self.std, x)

    def cdf(self, x):
        return _Phi((np.asarray(x, float) - self.mean) / self.std)

    def inv_cdf(self, p):
        """Closed-form quantile (erfinv); p in [0, 1]."""
        return self.mean + self.std * _Phi_inv(p)

    def percentile(self, p):
        """Quantile with p in [0, 1] (class override of the base convention)."""
        assert 0 <= p <= 1, "p must be between 0 and 1"
        return float(self.inv_cdf(p))


class Skew_normal_PDF(Distribution):
    """
    Skew-normal distribution SN(xi, omega, alpha): location xi, scale omega > 0,
    shape alpha. alpha = 0 recovers N(xi, omega); alpha > 0 skews right and
    alpha < 0 skews left. The pdf and cdf are closed form (the cdf uses Owen's T
    function); the quantile has no closed form, so inv_cdf is interpolated from
    the cdf on a fine grid, the same approach Triangular_PDF uses.
    """

    def __init__(self, xi, omega, alpha):
        assert omega > 0
        self.xi = float(xi)
        self.omega = float(omega)
        self.alpha = float(alpha)
        delta = self.alpha / np.sqrt(1.0 + self.alpha ** 2)
        self.mean = self.xi + self.omega * delta * np.sqrt(2.0 / np.pi)
        self.std = self.omega * np.sqrt(1.0 - 2.0 * delta ** 2 / np.pi)
        gx = np.linspace(self.mean - 6 * self.std, self.mean + 6 * self.std, 4000)
        cg = self.cdf(gx)
        cu, idx = np.unique(cg, return_index=True)        # strictly increasing for inverse
        self.inv_cdf = interp1d(cu, gx[idx], kind="linear", bounds_error=False,
                                fill_value=(gx[0], gx[-1]))
        self.minx, self.maxx = float(self.inv_cdf(0.001)), float(self.inv_cdf(0.999))
        self.x = np.linspace(self.minx, self.maxx, 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.title_pdf = 'Skew-Normal Distribution for xi = %g, omega = %g, alpha = %g' % (self.xi, self.omega, self.alpha)
        self.title_cdf = 'Skew-Normal CDF for xi = %g, omega = %g, alpha = %g' % (self.xi, self.omega, self.alpha)

    def pdf(self, x):
        return skew_normal(self.xi, self.omega, self.alpha, x)

    def cdf(self, x):
        z = (np.asarray(x, float) - self.xi) / self.omega
        return _Phi(z) - 2.0 * owens_t(z, self.alpha)

    def percentile(self, p):
        """Quantile with p in [0, 1] (class override of the base convention)."""
        assert 0 <= p <= 1, "p must be between 0 and 1"
        return float(self.inv_cdf(p))


class Lognormal_PDF(Distribution):
    """Lognormal distribution; (mu, sigma) are the mean/std of log(X)."""

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.mean, self.std = lognormal_mean_std_dev(self.mu, self.sigma)
        self.x = np.linspace(1e-3, self.inv_cdf(0.99), 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.title_pdf = 'Lognormal Distribution for mu = %g, sigma = %g' % (self.mu, self.sigma)
        self.title_cdf = 'Lognormal CDF for mu = %g, sigma = %g' % (self.mu, self.sigma)

    def pdf(self, x):
        return lognormal(self.mu, self.sigma, x)

    def cdf(self, x):
        x = np.asarray(x, float)
        safe = np.where(x > 0, x, 1.0)
        z = np.where(x > 0, (np.log(safe) - self.mu) / self.sigma, -np.inf)
        out = _Phi(z)
        return out if out.ndim else float(out)

    def inv_cdf(self, p):
        """Closed-form quantile; p in [0, 1]."""
        return np.exp(self.mu + self.sigma * _Phi_inv(p))


class Triangular_PDF(Distribution):
    """Triangular distribution on [low, high] with mode `expected`."""

    def __init__(self, low, expected, high):
        assert high >= expected and low <= expected
        self.low = 0.999 * low if low == expected else low
        self.expected = expected
        self.high = 1.001 * high if expected == high else high
        self.std = np.sqrt((high ** 2 + expected ** 2 + low ** 2
                            - high * low - high * expected - low * expected) / 18)
        self.mean = (low + expected + high) / 3
        self.x = np.linspace(self.low, self.high, 300)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.inv_cdf = interp1d(self.cy, self.x, kind="linear", fill_value="extrapolate")
        self.mode = expected                            # attribute (shadows base method)
        self.title_pdf = 'Triangular Distribution for (%g, %g, %g)' % (self.low, self.expected, self.high)
        self.title_cdf = 'Triangular CDF for (%g, %g, %g)' % (self.low, self.expected, self.high)

    def pdf(self, x):
        return triangular(self.low, self.expected, self.high, x)

    def cdf(self, x):
        a, b, c = self.low, self.high, self.expected
        x = np.asarray(x, float)
        out = np.zeros_like(x, dtype=float)
        m1 = (x >= a) & (x < c)
        out[m1] = (x[m1] - a) ** 2 / ((b - a) * (c - a))
        m2 = (x >= c) & (x <= b)
        out[m2] = 1.0 - (b - x[m2]) ** 2 / ((b - a) * (b - c))
        out[x > b] = 1.0
        return out

    def percentile(self, p):
        """Quantile with p in [0, 1] (class override of the base convention)."""
        assert 0 <= p <= 1, "p must be between 0 and 1"
        return float(self.inv_cdf(p))

    def samples(self, number):
        return self.inv_cdf(np.random.uniform(0, 1, number))


class Trunc_norm_PDF(Distribution):
    """
    Truncated normal: N(mu, sigma) restricted to [a, b] and renormalized.
    (Completed and corrected: the original had a broken CDF and no inverse.)
    """

    def __init__(self, mu, sigma, a, b):
        assert sigma > 0 and b > a
        self.mu, self.sigma, self.a, self.b = float(mu), float(sigma), float(a), float(b)
        self._Z = _Phi((b - mu) / sigma) - _Phi((a - mu) / sigma)   # mass in [a, b]
        self.x = np.linspace(a, b, 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.mean = _trapz(self.x * self.y, self.x)
        self.std = np.sqrt(_trapz((self.x - self.mean) ** 2 * self.y, self.x))
        self.title_pdf = 'Truncated Normal on [%g, %g]' % (a, b)
        self.title_cdf = 'Truncated Normal CDF on [%g, %g]' % (a, b)

    def pdf(self, x):
        x = np.asarray(x, float)
        dens = normal(self.mu, self.sigma, x) / self._Z
        return np.where((x >= self.a) & (x <= self.b), dens, 0.0)

    def cdf(self, x):
        x = np.asarray(x, float)
        lo = _Phi((self.a - self.mu) / self.sigma)
        val = (_Phi((np.clip(x, self.a, self.b) - self.mu) / self.sigma) - lo) / self._Z
        return np.clip(val, 0.0, 1.0)

    def inv_cdf(self, p):
        lo = _Phi((self.a - self.mu) / self.sigma)
        return self.mu + self.sigma * _Phi_inv(lo + np.asarray(p, float) * self._Z)


class Uniform_PDF(Distribution):
    """Uniform distribution on [start, end]. Closed-form CDF/quantile."""

    def __init__(self, start, end):
        assert end >= start
        self._start, self._end = start, end
        self.mean = (start + end) / 2
        self.std = np.sqrt((end - start) ** 2 / 12)
        self.x = np.linspace(start, end, 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.title_pdf = 'Uniform Distribution for (%g, %g)' % (start, end)
        self.title_cdf = 'Uniform CDF for (%g, %g)' % (start, end)

    def pdf(self, x):
        x = np.asarray(x, float)
        return np.where((x < self._start) | (x > self._end), 0.0,
                        1.0 / (self._end - self._start))

    def cdf(self, x):
        x = np.asarray(x, float)
        return np.clip((x - self._start) / (self._end - self._start), 0.0, 1.0)

    def inv_cdf(self, p):
        return self._start + np.asarray(p, float) * (self._end - self._start)


class Empirical_PDF(Distribution):
    """
    Build a pdf / cdf / inverse-cdf from a sample array via a (Freedman-Diaconis
    by default) histogram and linear interpolation. Ideal for Monte-Carlo output.
    """

    def __init__(self, array, name='', nbins=0):
        array = np.asarray(array)
        self.bins = 'fd' if nbins == 0 else nbins
        self.name = name
        self.array = array

        self.hist = np.histogram(array, bins=self.bins, density=True)
        bin_edges, bin_heights = self.hist[1], self.hist[0]
        self.centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        self.pdf = self._create_pdf_func(bin_edges, bin_heights)
        self.x = np.linspace(self.centers.min(), self.centers.max(), 100)
        self._cdf = self._create_cdf_func(bin_edges, bin_heights)
        self.inv_cdf = self._create_inv_cdf()

        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.mean = array.mean()
        self.std = array.std()
        self.median = float(self.inv_cdf(0.5))
        self.title = f'Distribution of {self.name}'
        self.title_cdf = f'Empirical Cumulative Distribution for mean = {self.mean:.0f}, std = {self.std:.0f}'

    def _create_pdf_func(self, bin_edges, bin_heights):
        tx = np.concatenate(([bin_edges[0]], self.centers, [bin_edges[-1]]))
        ty = np.concatenate(([0], bin_heights, [0]))
        return intp.interp1d(tx, ty, kind='linear', fill_value='extrapolate')

    def _create_cdf_func(self, bin_edges, bin_heights):
        cumulative = np.cumsum(bin_heights * np.diff(bin_edges))
        cumulative = np.concatenate(([0], cumulative))
        return intp.interp1d(bin_edges, cumulative, kind='linear',
                             bounds_error=False, fill_value=(0, 1))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        out[x <= np.min(self.x)] = 0
        out[x >= np.max(self.x)] = 1
        mask = (x > np.min(self.x)) & (x < np.max(self.x))
        out[mask] = self._cdf(x[mask])
        return out.item() if np.isscalar(x) or x.ndim == 0 else out

    def _create_inv_cdf(self):
        grid_x = np.linspace(self.x.min(), self.x.max(), 2000)
        return intp.interp1d(self.cdf(grid_x), grid_x, bounds_error=False,
                             fill_value=(grid_x[0], grid_x[-1]))

    def confidence(self, x):
        """Survival P(X > x) = 1 - cdf(x)."""
        return 1.0 - self.cdf(x)

    def percentile(self, p):
        """Quantile with p in [0, 1] (class override of the base convention)."""
        assert 0 <= p <= 1, "p must be between 0 and 1"
        return float(self.inv_cdf(p))


class Logisitic_PDF(Distribution):           # (name kept; alias Logistic_PDF below)
    """Logistic distribution with location mu and scale s. Closed-form CDF/quantile."""

    def __init__(self, mu, s):
        assert s >= 0
        self.mean = mu
        self.s = s
        self.std = s * np.pi / np.sqrt(3)              # corrected: std = s*pi/sqrt(3)
        self.minx, self.maxx = self.inv_cdf(0.001), self.inv_cdf(0.999)
        self.x = np.linspace(self.minx, self.maxx, 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.title_pdf = 'Logistic Distribution for mu = %g, s = %g' % (mu, s)
        self.title_cdf = 'Logistic CDF for mu = %g, s = %g' % (mu, s)

    def pdf(self, x):
        return logistic(self.mean, self.s, x)

    def cdf(self, x):
        return 1.0 / (1.0 + np.exp(-(np.asarray(x, float) - self.mean) / self.s))

    def inv_cdf(self, p):
        p = np.asarray(p, float)
        return self.mean + self.s * np.log(p / (1.0 - p))


Logistic_PDF = Logisitic_PDF                 # corrected-spelling alias


class Weibull_PDF(Distribution):
    """Weibull distribution with shape k and scale l. Closed-form everything."""

    def __init__(self, shape, scale):
        self.k = shape
        self.l = scale
        self.mean = self.l * gamma(1 + 1 / self.k)
        var = self.l ** 2 * (gamma(1 + 2 / self.k) - gamma(1 + 1 / self.k) ** 2)
        self.std = np.sqrt(var)
        self.x = np.linspace(0, self.mean + 4 * self.std, 100)
        self.y = self.pdf(self.x)
        self.cy = self.cdf(self.x)
        self.title_pdf = 'Weibull Distribution for k = %g, l = %g' % (self.k, self.l)
        self.title_cdf = 'Weibull CDF for k = %g, l = %g' % (self.k, self.l)

    def pdf(self, x):
        return weibull(self.l, self.k, x)

    def cdf(self, x):
        x = np.asarray(x, float)
        return 1 - np.exp(-((x / self.l) ** self.k))

    def inv_cdf(self, p):
        p = np.asarray(p, float)
        return self.l * (-np.log(1 - p)) ** (1 / self.k)

    # legacy name kept (other code may call .inverse_cdf)
    inverse_cdf = inv_cdf

    def find_stats(self, nsamps=1000):
        """Closed-form (mean, std); nsamps kept for signature compatibility."""
        return self.mean, self.std

    def samples(self, n):
        return self.inv_cdf(np.random.uniform(0, 1, n))