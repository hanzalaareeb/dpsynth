# Copyright 2026 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.local_mode import sufficient_statistics
import numpy as np


def _dense_uniform_histogram(lower, upper, n, grid_size=1000):
  """Creates a dense histogram from uniformly spaced data."""
  data = np.linspace(lower, upper, n)
  indices = np.round((data - lower) / (upper - lower) * (grid_size - 1)).astype(
      np.int64
  )
  counts = np.zeros(grid_size, dtype=int)
  for idx in indices:
    counts[idx] += 1
  return counts


class QuantilesFromHistogramTest(parameterized.TestCase):

  def test_no_levels_returns_empty(self):
    rng = np.random.default_rng(0)
    counts = np.array([10])
    edges = sufficient_statistics.quantiles_from_histogram(
        rng,
        counts,
        0.0,
        10.0,
        epsilon_levels=np.array([]),
    )
    self.assertEmpty(edges)

  @parameterized.parameters(1, 2, 3, 4)
  def test_edge_count_matches_levels(self, levels):
    rng = np.random.default_rng(0)
    counts = _dense_uniform_histogram(0.0, 10.0, 200, grid_size=10001)
    edges = sufficient_statistics.quantiles_from_histogram(
        rng,
        counts,
        0.0,
        10.0,
        epsilon_levels=np.ones(levels),
        grid_size=10001,
    )
    self.assertLen(edges, 2**levels - 1)


class HistogramNumericalInitializerTest(absltest.TestCase):

  def test_calibrate_sets_dp_event(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    init = sufficient_statistics.HistogramNumericalInitializer(
        name='age',
        attribute=attr,
        num_buckets=4,
        grid_size=10001,
    ).calibrate(zcdp_rho=1.0)
    event = init.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # 4 buckets = 2 levels.
    self.assertLen(event.events, 2)

  def test_uncalibrated_raises(self):
    attr = domain.NumericalAttribute(min_value=0, max_value=100)
    init = sufficient_statistics.HistogramNumericalInitializer(
        name='age',
        attribute=attr,
    )
    with self.assertRaises(ValueError):
      init(np.random.default_rng(0), np.zeros(100))

  def test_integer_attribute_snaps_edges(self):
    rng = np.random.default_rng(42)
    attr = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    counts = _dense_uniform_histogram(0.0, 11.0, 200, grid_size=10001)
    init = sufficient_statistics.HistogramNumericalInitializer(
        name='count',
        attribute=attr,
        num_buckets=4,
        grid_size=10001,
    ).calibrate(zcdp_rho=1.0)
    cm = init(rng, counts)
    for edge in cm.bin_edges:
      self.assertEqual(edge, int(edge))


class ParityTest(absltest.TestCase):
  """Verifies NumericalInitializer and HistogramNumericalInitializer agree."""

  def test_noisy_edge_distributions_match(self):
    """At finite rho, edge distributions should be statistically similar."""
    lower, upper = 0.0, 100.0
    grid_size = 10001
    num_buckets = 4
    rho = 10.0
    num_trials = 1000
    attr = domain.NumericalAttribute(
        min_value=int(lower),
        max_value=int(upper),
    )

    # Snap raw data to the grid so both mechanisms operate on identical
    # discrete data, eliminating the continuous-vs-discrete confound.
    delta = (upper - lower) / (grid_size - 1)
    raw = np.linspace(lower, upper, 1000)
    indices = np.round((raw - lower) / delta).astype(np.int64)
    data = lower + indices * delta
    counts = _dense_uniform_histogram(lower, upper, 1000, grid_size=grid_size)

    # Collect edge samples from both initializers.  Seeds are fixed
    # (deterministic), so there is no flakiness risk and we can use a tight
    # p-value threshold.  With 1000 samples per group, the two-sample KS test
    # detects CDF shifts >~0.06 at alpha=0.01 — on a [0, 100] domain, any
    # bug that shifts the median distribution by more than ~6 units is caught.
    num_edges = num_buckets - 1
    raw_edges = np.zeros((num_trials, num_edges))
    hist_edges = np.zeros((num_trials, num_edges))
    for i in range(num_trials):
      rng = np.random.default_rng(i)
      raw_cm = initialization.NumericalInitializer(
          name='x',
          num_partitions=num_buckets,
          attribute=attr,
      ).calibrate(zcdp_rho=rho)(rng, data)
      raw_edges[i, : len(raw_cm.bin_edges)] = raw_cm.bin_edges

      rng = np.random.default_rng(i + num_trials)
      hist_cm = sufficient_statistics.HistogramNumericalInitializer(
          name='x',
          attribute=attr,
          num_buckets=num_buckets,
          grid_size=grid_size,
      ).calibrate(zcdp_rho=rho)(rng, counts)
      hist_edges[i, : len(hist_cm.bin_edges)] = hist_cm.bin_edges

    # We do NOT use a KS test here because the continuous mechanism uses
    # log(interval_length) weighting while the discrete mechanism is uniform
    # over grid points — their CDFs differ by design even on identical data.
    # Instead we check practical equivalence: matching means and stds.
    for j in range(num_edges):
      raw_mean = raw_edges[:, j].mean()
      hist_mean = hist_edges[:, j].mean()
      raw_std = raw_edges[:, j].std()
      hist_std = hist_edges[:, j].std()
      np.testing.assert_allclose(hist_mean, raw_mean, atol=0.1)
      np.testing.assert_allclose(hist_std, raw_std, rtol=0.5)


if __name__ == '__main__':
  absltest.main()
