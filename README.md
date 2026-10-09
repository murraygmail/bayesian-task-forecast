# project: Bayesian task-duration forecast

Python library and test scripts for forecasting how long a task will take
from its three-point plan and its periodic progress reports. The model learns
the task's *velocity* (fraction completed per period) by Bayesian parameter
learning on a (mu, sigma) grid, then forecasts the periods remaining and the
probability of meeting the planned target. Tasks can be rolled up, and a task
can depend on others.

Requires numpy, scipy, matplotlib (`pip install -r requirements.txt`).
Licensed under the Apache License 2.0; see the License section.

## Layout

```
project/
  Service_bpl/        the library ("bpl" = Bayesian parameter learning)
    Distributions.py       probability distributions with a common interface
    weighted.py            weighted likelihood grid: likelihood_array
    Tasks.py               task, rollup, duration plots
    Integrators.py         Simpson / Boole quadrature (standalone utility)
  Tests/              scripts that exercise the library (run, look at the figures)
    test_task.py           one task, four progress reports, Normal likelihood
    test_rollup.py         two tasks rolled up, logistic likelihood
    test_dep.py            a task with two dependencies
    Correlated.py          correlated cost increase experiment
    math_hist.py           Empirical_PDF over a restricted range
    regression_test.py     regression suite (unittest, headless)
  Stats/              standalone teaching examples, independent of Service_bpl
    PDF_mu_sigma.py        mean and std of a continuous pdf by Riemann sum
    PMFmu_sigma.py         mean and std of a binomial PMF
  LICENSE, NOTICE     Apache License 2.0 and copyright notice
  requirements.txt    numpy, scipy, matplotlib
  _not_released/      superseded Distributions_old.py and pre-fix backups; not part
                      of the release (git-ignored)
```

## Running

All imports are package imports (`from Service_bpl import Tasks`), so the
project root must be on `sys.path`. From a terminal:

```
cd project
PYTHONPATH=. python3 Tests/test_task.py
```

In Spyder, run the scripts with the project folder as the working directory
(or add it to the PYTHONPATH manager). The scripts open matplotlib windows;
set `MPLBACKEND=Agg` to run them headless.

Every script in `Tests/` and `Stats/` runs as of 9 Oct 2026.

### Regression test

```
PYTHONPATH=. python3 Tests/regression_test.py
```

19 tests, about 8 seconds, no figure windows. It runs every script, checks
the `Distributions` round trips and moments, the likelihood grid, task
velocities / weights / forecasts in a healthy and a stalled case,
dependencies, rollups, tail trimming, the `Correlated` and `math_hist`
scripts and the `Stats` examples. Monte Carlo checks use seeded draws and
tolerance bands. Run it after any change to `Service_bpl`.

## How the model works (Tasks.py and weighted.py)

1. A `task` is built from a plan `{'low', 'det', 'high'}` in periods of
   effort and a likelihood function `f(mu, sigma, x)` from `Distributions`
   (`normal` or `logistic`). The plan gives a `Triangular_PDF` prior on
   duration; its reciprocal samples give the prior on velocity (mu), and a
   wide `Uniform_PDF` is the prior on sigma.
2. Each progress report `{'period': t, 'pct': cumulative fraction done}`
   yields an observed velocity over the interval it covers. Observations are
   weighted by interval length.
3. `weighted.likelihood_array` fills a 50 x 50 (mu, sigma) grid with the
   weighted likelihood of all observations times the priors. The row and
   column sums are the marginal posteriors; they become the priors for the
   next update.
4. Sampling (mu, sigma) from the marginals and drawing Normal(mu, sigma)
   gives the posterior predictive velocity; the duration forecast is
   (fraction remaining) / velocity, as an `Empirical_PDF`.
5. `task.target()` is `det` minus the latest report period; the duration
   figure shades the forecast green up to the target and red beyond, and
   labels the probability of meeting it.
6. `rollup(name, tasks)`: duration is the Monte Carlo max of the tasks'
   durations, target the max of their `det`s. A task constructed with
   `depends=[...]` adds the rollup of its dependencies to its own duration
   (`compute_dependancies`).

## Distributions.py conventions

Every distribution has `.pdf(x)`, `.cdf(x)`, `.inv_cdf(p)`, `.quantile(p)`,
`.samples(n)`, `.mean`, `.std`, and a plotting grid `.x`, `.y`. Use
`quantile(p)` with p in [0, 1] as the inverse CDF. `percentile()` is
inconsistent by class for historical reasons (fraction on `Empirical_PDF`,
`Normal_PDF`, `Triangular_PDF`, `ArrayPDF`; percent on the base class) and
is best avoided in new code. The module docstring lists everything.

## Changes on 9 Oct 2026

- `weighted.py`: `import Distributions as ds` (a top-level import that relied
  on a hard-coded path to an old Dropbox folder) changed to
  `from Service_bpl import Distributions as ds`.
- `Tasks.py`, `trimed_pdf`: `pdf.percentile(trimpct)` changed to
  `pdf.quantile(trimpct / 100)`, matching the revised `Distributions.py`.
- `Tasks.py`, `compute_dependancies`: now sets `self.dep_target` (it set
  `self.target`, a float that shadowed the `target()` method, so a later
  `update()` with duration plots on a dependent task failed).
- `Tests/math_hist.py`: ported to the current `Empirical_PDF` signature; the
  old `Range=True, hist_range=` is replaced by filtering the samples to the
  range, which is what `np.histogram(range=...)` did.
- `Tests/Correlated.py`, `tri_cost`: the `'high'` branch for `opt` tested
  `pess`; now tests `opt`.
- `Tests/test_rollup.py`: `plots='False'` (a string, truthy) is now `False`.
- Module, class and function docstrings added throughout, and
  `Tests/regression_test.py` added.
- Released under the Apache License 2.0, replacing the 2022 use-only
  header. `Distributions_old.py` (superseded) and the pre-fix `.bak` copies
  moved to `_not_released/`.

## Quirks (left as found)

- `Tasks.task.remaining()` returns `det` (periods) before any report and a
  fraction after; only the latter is used.
- Stale `sys.path.append` blocks in `weighted.py`, `Tasks.py` and
  `Tests/test_dep.py` point at a Dropbox path that no longer exists; they are
  harmless now that imports go through the package.

## License

Copyright 2022-2026 Murray Cantor.

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this software except in compliance with the License. You may obtain a
copy of the License at http://www.apache.org/licenses/LICENSE-2.0 or in the
`LICENSE` file. Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

Every source file carries an SPDX header (`SPDX-License-Identifier:
Apache-2.0`). Contributions are accepted under the same license (Section 5
of the License).
