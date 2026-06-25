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

"""Histogram-based numerical initialization from sufficient statistics.

This module enables numerical attribute initialization from pre-aggregated
dense histograms, removing the need for raw-data access.  The primary use
case is a two-pass pipeline: a first pass (e.g., in Apache Beam) computes a
dense histogram over a fine-grained grid, then this module computes DP
quantiles from that histogram to discretize the numerical domain — exactly
as ``NumericalInitializer`` does from raw data, but without ever touching
individual records after aggregation.

Public API:
  - ``quantiles_from_histogram``: DP quantiles via recursive median splits.
  - ``HistogramNumericalInitializer``: ``DPMechanism`` that produces a
    ``ColumnMeasurement`` from a dense histogram.
"""

from __future__ import annotations

import dataclasses

import dp_accounting
from dpsynth import domain
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
import numpy as np
import scipy.special


def _median_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    epsilon: float,
) -> int:
  """Returns the index of a DP median within a dense histogram.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram counts.
    epsilon: Exponential mechanism privacy parameter for this level.

  Returns:
    The index of the selected median grid point within ``counts``.
  """
  total_points = len(counts)
  n = counts.sum()
  target = n / 2.0
  cumsum = np.cumsum(counts)

  # Infinite budget = exact median, useful for testing.
  if epsilon == np.inf:
    return int(np.searchsorted(cumsum, target))

  # Score u(v) = -dist(target, [L_v, R_v]), sensitivity 1/2.
  left_ranks = np.r_[0, cumsum[:-1]]
  scores = -np.maximum(0, np.maximum(left_ranks - target, target - cumsum))

  probs = scipy.special.softmax(epsilon * scores)
  return int(rng.choice(total_points, p=probs))


def quantiles_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    lower: float,
    upper: float,
    epsilon_levels: np.ndarray,
    grid_size: int = 10_000_000,
) -> list[float]:
  """Computes DP quantiles from a dense histogram via recursive median splits.

  Uses the discrete exponential mechanism to recursively find medians,
  splitting the histogram at each level to produce ``num_buckets - 1``
  quantile edges.  The number of buckets is ``2 ** len(epsilon_levels)``.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram of shape ``(grid_size,)``.
    lower: Lower bound of the data domain.
    upper: Upper bound of the data domain (exclusive).
    epsilon_levels: Per-level exponential mechanism epsilons, ordered from the
      deepest (finest) level to the shallowest (coarsest).
    grid_size: Number of uniformly spaced grid points spanning ``[lower,
      upper]``.

  Returns:
    A sorted list of ``2 ** len(epsilon_levels) - 1`` quantile edge values.
  """
  levels = len(epsilon_levels)
  if levels == 0:
    return []

  # Uniform grid: counts[i] corresponds to value lower + i * delta.
  delta = (upper - lower) / (grid_size - 1)

  def _rec(lo_idx, hi_idx, depth):
    if depth == 0:
      return []
    sub_counts = counts[lo_idx:hi_idx]
    median_local = _median_from_histogram(
        rng, sub_counts, epsilon_levels[depth - 1]
    )
    median_global_idx = lo_idx + median_local
    median_value = lower + median_global_idx * delta
    left = _rec(lo_idx, median_global_idx, depth - 1)
    right = _rec(median_global_idx, hi_idx, depth - 1)
    return left + [median_value] + right

  return _rec(0, len(counts), levels)


@dataclasses.dataclass
class HistogramNumericalInitializer(primitives.DPMechanism):
  """Initializes a numerical attribute from a pre-aggregated dense histogram.

  This mechanism mirrors ``NumericalInitializer`` but operates on a dense
  histogram rather than raw data.  It is a composition of exponential
  mechanisms (one per recursion level), producing quantile edges that
  discretize the numerical domain.

  Usage follows the standard three-phase ``DPMechanism`` pattern::

      initializer = HistogramNumericalInitializer(
          name='age', attribute=attr, num_buckets=4, grid_size=10001,
      ).calibrate(zcdp_rho=1.0)
      result = initializer(rng, counts)

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The ``NumericalAttribute`` defining the data domain.
    num_buckets: Number of quantile buckets (must be a power of 2).
    grid_size: Number of uniformly spaced grid points spanning the attribute's
      ``[min_value, exclusive_max_value]`` range.
  """

  name: str
  attribute: domain.NumericalAttribute
  num_buckets: int = 32
  grid_size: int = 10_000_000
  _epsilon_levels: tuple[float, ...] | None = dataclasses.field(
      default=None, repr=False
  )

  @property
  def _num_levels(self) -> int:
    result = int(np.log2(self.num_buckets))
    if 2**result != self.num_buckets:
      raise ValueError(f'{self.num_buckets=} must be a power of 2.')
    return result

  def calibrate(
      self, *, zcdp_rho: float, epsilon_ratio: float = 2.0
  ) -> HistogramNumericalInitializer:
    """Returns a copy calibrated to the given zCDP budget.

    Args:
      zcdp_rho: The zCDP privacy budget (rho).
      epsilon_ratio: Factor by which epsilon grows at each deeper level.

    Returns:
      A calibrated ``HistogramNumericalInitializer``.
    """
    if zcdp_rho <= 0:
      raise ValueError(f'zcdp_rho must be positive, got {zcdp_rho}.')
    levels = self._num_levels
    if levels == 0:
      return dataclasses.replace(self, _epsilon_levels=())
    rho_ratio = epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    return dataclasses.replace(self, _epsilon_levels=tuple(eps.tolist()))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    if self._epsilon_levels is None:
      raise ValueError('Must call calibrate() before accessing dp_event.')
    return dp_accounting.ComposedDpEvent([
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self._epsilon_levels
    ])

  def __call__(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
      *,
      estimated_total: float | None = None,
      out_of_domain_count: int | None = None,
  ) -> initialization.ColumnMeasurement:
    """Computes DP quantiles from a dense histogram and returns a ColumnMeasurement.

    Args:
      rng: A numpy random number generator.
      counts: Dense 1D histogram of shape ``(grid_size,)``.
      estimated_total: If provided, a heuristic one-way measurement is included
        assuming a uniform distribution over the bins.
      out_of_domain_count: Count of records outside the domain range.  May only
        be provided when ``attribute.clip_to_range`` is False.  Currently
        unused; reserved for future OOD-aware measurement construction.

    Returns:
      A ``ColumnMeasurement`` with bin edges and optionally a measurement.
    """
    if self._epsilon_levels is None:
      raise ValueError('Must call calibrate() before calling.')
    del out_of_domain_count  # Reserved for future use.

    raw_edges = quantiles_from_histogram(
        rng,
        counts,
        self.attribute.min_value,
        self.attribute.exclusive_max_value,
        epsilon_levels=np.asarray(self._epsilon_levels),
        grid_size=self.grid_size,
    )
    rho = sum(e**2 / 8.0 for e in self._epsilon_levels)
    return initialization.edges_to_column_measurement(
        raw_edges=raw_edges,
        attribute=self.attribute,
        name=self.name,
        zcdp_rho=rho,
        estimated_total=estimated_total,
    )
