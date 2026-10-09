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
regression_test
=============================================================================
Regression suite for the project.  Runs headless (no figure windows).

    cd project
    PYTHONPATH=. python3 Tests/regression_test.py

Covers
------
* every script in Tests/ and Stats/ runs to completion
* Distributions: quantile / cdf round trips, Empirical_PDF recovers moments
* weighted.likelihood_array: marginals normalise, predictive pdf is positive
* Tasks.task: velocities and weights from progress reports, forecast
  behaviour in a healthy and a stalled case, target() still callable after
  compute_dependancies (the shadowing fix), dep_target arithmetic
* Tasks.rollup: duration is the max, target is the max det
* Tasks.trimed_pdf: tail actually removed
* Correlated.tri_cost: opt / pess branches
* math_hist: histogram support restricted to the requested range
* Stats scripts: mean / std of the examples

Monte Carlo results are checked against bands, with the numpy RNG seeded,
so the suite is deterministic.

Created on Fri Oct  9 2026
"""

import os
import sys
import subprocess
import unittest
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Service_bpl import Distributions as ds
from Service_bpl import Tasks
from Service_bpl.weighted import likelihood_array

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def load_script(relpath):
    """Import a script file as a module (runs its top-level code, headless)."""
    path = os.path.join(ROOT, relpath)
    name = os.path.splitext(os.path.basename(relpath))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plt.close('all')
    return mod


def make_task(name, plan, reports, like_pdf=ds.normal, depends=[]):
    t = Tasks.task(name, plan, like_pdf, depends=depends)
    for r in reports:
        t.update(r, plots=False, dur_plots=False)
    plt.close('all')
    return t


PLAN = {'low': 4, 'det': 6, 'high': 9}
HEALTHY = [{'period': 1, 'pct': .10}, {'period': 3, 'pct': .40},
           {'period': 4, 'pct': .80}, {'period': 5, 'pct': .95}]
STALLED = [{'period': 1, 'pct': .10}, {'period': 3, 'pct': .10},
           {'period': 4, 'pct': .10}, {'period': 5, 'pct': .15}]


class ScriptsRun(unittest.TestCase):
    """Every script in Tests/ and Stats/ runs to completion, headless."""

    SCRIPTS = ['Tests/test_task.py', 'Tests/test_rollup.py', 'Tests/test_dep.py',
               'Tests/Correlated.py', 'Tests/math_hist.py',
               'Stats/PDF_mu_sigma.py', 'Stats/PMFmu_sigma.py']

    def test_scripts(self):
        env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND='Agg')
        for s in self.SCRIPTS:
            with self.subTest(script=s):
                r = subprocess.run([sys.executable, os.path.join(ROOT, s)],
                                   env=env, capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stderr[-800:])


class DistributionsTests(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_triangular(self):
        T = ds.Triangular_PDF(4, 6, 9)
        self.assertAlmostEqual(T.quantile(0), 4, places=6)
        self.assertAlmostEqual(T.quantile(1), 9, places=6)
        self.assertAlmostEqual(T.cdf(T.quantile(0.3)), 0.3, places=4)
        s = T.samples(20000)
        self.assertTrue((s >= 4).all() and (s <= 9).all())
        self.assertAlmostEqual(s.mean(), (4 + 6 + 9) / 3, delta=0.05)

    def test_normal_roundtrip(self):
        N = ds.Normal_PDF(2, 3)
        for p in (0.05, 0.5, 0.95):
            self.assertAlmostEqual(N.cdf(N.quantile(p)), p, places=6)
        self.assertAlmostEqual(N.quantile(0.5), 2, places=6)

    def test_empirical_recovers_moments(self):
        samps = np.random.normal(10, 2, 50000)
        E = ds.Empirical_PDF(samps)
        self.assertAlmostEqual(E.mean, 10, delta=0.05)
        self.assertAlmostEqual(E.std, 2, delta=0.05)
        self.assertAlmostEqual(E.quantile(0.5), 10, delta=0.1)
        self.assertAlmostEqual(E.cdf(12), 0.841, delta=0.02)
        self.assertLess(E.quantile(0.05), E.quantile(0.5))
        self.assertLess(E.quantile(0.5), E.quantile(0.95))

    def test_percentile_fraction_on_empirical(self):
        """Empirical_PDF.percentile takes a fraction (the trimed_pdf fix relies on this)."""
        E = ds.Empirical_PDF(np.random.normal(0, 1, 10000))
        self.assertAlmostEqual(E.percentile(0.5), E.quantile(0.5), places=9)
        with self.assertRaises(AssertionError):
            E.percentile(95)


class LikelihoodArrayTests(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_marginals_and_predictive(self):
        mu_prior = ds.Uniform_PDF(0.01, 0.5)
        sd_prior = ds.Uniform_PDF(0.01, 0.5)
        arr = likelihood_array(ds.normal, mu_prior, sd_prior, dim1=30, dim2=30, nsamps=5000)
        arr.update_array([0.1, 0.15, 0.2], [1, 2, 1])
        self.assertEqual(arr.array.shape, (30, 30))
        self.assertTrue((arr.array >= 0).all())
        self.assertAlmostEqual(arr.weights.sum(), 1.0, places=9)
        m1, m2 = arr.margins()
        # marginals integrate to one over their grids
        self.assertAlmostEqual(_trapz(m1.y, m1.x), 1.0, delta=0.05)
        self.assertAlmostEqual(_trapz(m2.y, m2.x), 1.0, delta=0.05)
        # mu posterior sits around the weighted mean of the data (0.15)
        self.assertAlmostEqual(m1.quantile(0.5), 0.15, delta=0.05)
        pred = arr.pdf()
        self.assertGreater(pred.quantile(0.01), 0)


class TaskTests(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    def test_observations(self):
        t = make_task('T', PLAN, HEALTHY)
        np.testing.assert_allclose(t.vels, [0.10, 0.15, 0.40, 0.15], atol=1e-12)
        self.assertEqual(t.durs, [1, 3, 4, 5])
        self.assertAlmostEqual(sum(t.weights), 1.0, places=9)
        np.testing.assert_allclose(t.weights, np.array([1, 3, 4, 5]) / 13, atol=1e-12)
        self.assertEqual(t.target(), 1)                 # det 6 - period 5
        self.assertAlmostEqual(t.remaining(), 0.05, places=9)
        self.assertEqual(t.pct_comp, 0.95)

    def test_healthy_forecast(self):
        t = make_task('T', PLAN, HEALTHY)
        d = t.dur_pdf
        # 5% left at ~0.15-0.4 per period: a few tenths of a period, well under a period
        self.assertLess(d.quantile(0.5), 1.0)
        self.assertGreater(d.cdf(t.target()), 0.8)
        self.assertLess(d.quantile(0.05), d.quantile(0.5))
        self.assertLess(d.quantile(0.5), d.quantile(0.95))

    def test_stalled_forecast(self):
        t = make_task('T', PLAN, STALLED)
        np.testing.assert_allclose(t.vels, [0.1, 0.0, 0.0, 0.05], atol=1e-12)
        d = t.dur_pdf
        self.assertGreater(d.quantile(0.5), 1.0)        # 85% left, velocity ~0.05
        self.assertLess(d.cdf(t.target()), 0.1)

    def test_velocity_pdf_positive(self):
        t = make_task('T', PLAN, HEALTHY)
        self.assertGreater(t.vel_pdf.quantile(0.001), 0)

    def test_logistic_likelihood(self):
        t = make_task('T', PLAN, HEALTHY, like_pdf=ds.logistic)
        self.assertLess(t.dur_pdf.quantile(0.5), 1.0)

    def test_update_with_plots(self):
        """Plotting paths (gen_plots, dur_plot, trimed_pdf) run headless."""
        t = Tasks.task('T', PLAN, ds.normal)
        for r in HEALTHY[:2]:
            t.update(r)                                  # plots=True, dur_plots=True
        plt.close('all')
        self.assertEqual(len(t.ifigs), 2)
        self.assertEqual(len(t.efigs), 2)

    def test_dependencies_and_target_not_shadowed(self):
        d1 = make_task('D1', {'low': 4, 'det': 6, 'high': 9}, HEALTHY[:1])
        d2 = make_task('D2', {'low': 8, 'det': 12, 'high': 15}, HEALTHY[:1])
        b = Tasks.task('B', {'low': 10, 'det': 15, 'high': 25}, ds.normal, depends=[d1, d2])
        self.assertEqual(b.dep_target, 15 + 12)
        b.compute_dependancies()
        self.assertEqual(b.dep_target, 15 + 12)
        self.assertTrue(callable(b.target))              # the shadowing fix
        # dependency duration is own duration plus the rollup: strictly longer
        self.assertGreater(b.dep_dur_pdf.quantile(0.5), b.dur_pdf.quantile(0.5))
        # an update with duration plots on a dependent task must not fail
        b.update(HEALTHY[0])
        plt.close('all')
        self.assertEqual(b.target(), 14)
        self.assertEqual(b.dep_target, 15 + 12)


class RollupTests(unittest.TestCase):

    def setUp(self):
        np.random.seed(3)

    def test_rollup(self):
        t1 = make_task('T1', {'low': 4, 'det': 6, 'high': 9}, HEALTHY[:1], like_pdf=ds.logistic)
        t2 = make_task('T2', {'low': 10, 'det': 13, 'high': 19}, HEALTHY[:1], like_pdf=ds.logistic)
        roll = Tasks.rollup('R', [t1, t2])
        self.assertEqual(roll.target, 13)
        for q in (0.1, 0.5, 0.9):
            self.assertGreaterEqual(roll.dur_pdf.quantile(q) + 1e-6,
                                    max(t1.dur_pdf.quantile(q), t2.dur_pdf.quantile(q)))
        fig, ax = roll.plot()
        plt.close('all')
        self.assertIsNotNone(fig)

    def test_trimed_pdf(self):
        d = ds.Empirical_PDF(np.random.lognormal(1, 0.8, 20000))
        cut = d.quantile(0.95)
        tr = Tasks.trimed_pdf(d)
        self.assertLessEqual(tr.array.max(), cut)
        self.assertLess(tr.quantile(0.99), d.quantile(0.99))


class CorrelatedTests(unittest.TestCase):

    def test_tri_cost_branches(self):
        np.random.seed(4)
        C = load_script('Tests/Correlated.py')
        phi = (1 + np.sqrt(5)) / 2
        self.assertAlmostEqual(C.tri_cost(10).quantile(0), 10 / phi, places=6)
        self.assertAlmostEqual(C.tri_cost(10).quantile(1), 10 * phi, places=6)
        self.assertAlmostEqual(C.tri_cost(10, opt='low').quantile(0), 10 / 1.2, places=6)
        self.assertAlmostEqual(C.tri_cost(10, opt='high').quantile(0), 10 / 1.8, places=6)   # the opt/pess fix
        self.assertAlmostEqual(C.tri_cost(10, pess='low').quantile(1), 10 * 1.2, places=6)
        self.assertAlmostEqual(C.tri_cost(10, pess='high').quantile(1), 10 * 1.8, places=6)
        self.assertAlmostEqual(C.tri_cost(10, opt='average', pess='high').quantile(0), 10 / phi, places=6)


class MathHistTests(unittest.TestCase):

    def test_support_restricted(self):
        np.random.seed(5)
        M = load_script('Tests/math_hist.py')
        self.assertGreaterEqual(M.Test.x.min(), 4)
        self.assertLessEqual(M.Test.x.max(), 12)
        self.assertEqual(len(M.in_range), int(((M.data >= 4) & (M.data <= 12)).sum()))


class StatsTests(unittest.TestCase):

    def test_pdf_mu_sigma(self):
        S = load_script('Stats/PDF_mu_sigma.py')
        # The script integrates N(2, 3) over [-10, 10], which is not symmetric
        # about the mean (4 sigma below, 2.7 sigma above), so the truncated
        # moments differ slightly from 2 and 3.  Compare with the exact
        # truncated moments by fine midpoint quadrature.
        x = -10 + (np.arange(200000) + 0.5) * (20 / 200000)
        w = S.normal(2, 3, x) * (20 / 200000)
        m_exact = (x * w).sum()
        s_exact = np.sqrt(((m_exact - x) ** 2 * w).sum())
        self.assertAlmostEqual(S.mean, m_exact, delta=1e-3)
        self.assertAlmostEqual(S.std, s_exact, delta=1e-3)
        self.assertAlmostEqual(S.mean, 2, delta=0.1)
        self.assertAlmostEqual(S.std, 3, delta=0.2)

    def test_pmf_mu_sigma(self):
        S = load_script('Stats/PMFmu_sigma.py')
        self.assertAlmostEqual(S.fair_mean, 5.0, places=9)
        self.assertAlmostEqual(S.fair_std, np.sqrt(10 * .5 * .5), places=9)
        self.assertAlmostEqual(S.biased_mean, 3.0, places=9)
        self.assertAlmostEqual(S.biased_std, np.sqrt(10 * .3 * .7), places=9)
        self.assertAlmostEqual(sum(S.fair), 1.0, places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
