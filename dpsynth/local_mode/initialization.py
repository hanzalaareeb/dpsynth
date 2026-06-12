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

"""Utilities for measuring and integer-encoding single columns."""

import dataclasses

import dp_accounting
from dpsynth import domain
from dpsynth import transformations
from dpsynth.discrete_mechanisms import accounting
from dpsynth.local_mode import primitives
import mbi
import numpy as np


@dataclasses.dataclass
class ColumnMeasurement:
  categorical_attribute: domain.CategoricalAttribute
  transform_fn: transformations.DataTransformation
  measurement: mbi.LinearMeasurement | None


@dataclasses.dataclass
class NumericalInitializer:
  """Mechanism that creates the data encoding transform for numerical data."""

  name: str
  num_partitions: int
  attribute: domain.NumericalAttribute
  rng: np.random.Generator

  def dp_event(self, zcdp_rho: float) -> dp_accounting.DpEvent:
    levels = int(np.log2(self.num_partitions))
    budget_weights = 4 ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    epsilons = [accounting.zcdp_exponential_eps(rho) for rho in rho_levels]

    return dp_accounting.ComposedDpEvent(
        [dp_accounting.ExponentialMechanismDpEvent(epsilon=e) for e in epsilons]
    )

  def __call__(self, zcdp_rho: float, data: np.ndarray) -> ColumnMeasurement:
    """Returns a differentially private measurement of the given data."""
    bucket_edges = primitives.quantiles(
        self.rng,
        data,
        self.attribute.min_value,
        self.attribute.max_value,
        self.num_partitions,
        zcdp_rho,
    )
    attr, discretize_fn = transformations.create_discretize_transformation(
        self.attribute, bucket_edges
    )
    transform_fn = transformations.discrete_encoder(attr) @ discretize_fn
    return ColumnMeasurement(attr, transform_fn, None)


@dataclasses.dataclass
class CategoricalInitializer:
  """Mechanism that measures a noisy histogram for categorical data.

  Computes a closed-domain histogram over the pre-specified categories using
  the Gaussian mechanism. Values not in the domain are mapped to the
  attribute's designated out-of-domain value before histogramming.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The CategoricalAttribute defining the closed domain.
    rng: A numpy random number generator.
  """

  name: str
  attribute: domain.CategoricalAttribute
  rng: np.random.Generator

  def dp_event(self, zcdp_rho: float) -> dp_accounting.DpEvent:
    """Returns the DpEvent for the Gaussian mechanism.

    Args:
      zcdp_rho: Total zCDP privacy budget.

    Returns:
      A GaussianDpEvent describing the privacy cost.
    """
    # Gaussian mechanism with L2 sensitivity 1: rho = 1 / (2 * sigma^2).
    sigma = 1.0 / np.sqrt(2.0 * zcdp_rho)
    return dp_accounting.GaussianDpEvent(noise_multiplier=sigma)

  def __call__(self, zcdp_rho: float, data: np.ndarray) -> ColumnMeasurement:
    """Returns a differentially private measurement of the given data.

    Args:
      zcdp_rho: Total zCDP privacy budget for the histogram measurement.
      data: 1D array of raw categorical values.

    Returns:
      A ColumnMeasurement containing the categorical attribute, the encoding
      transform, and a LinearMeasurement with the noisy histogram.
    """
    sigma = 1.0 / np.sqrt(2.0 * zcdp_rho)
    transform_fn = transformations.discrete_encoder(self.attribute)
    encoded = np.array([transform_fn(v) for v in data])
    noisy_counts = primitives.gaussian_histogram(
        self.rng, encoded, self.attribute.size, sigma
    )
    measurement = mbi.LinearMeasurement(
        noisy_counts, (self.name,), stddev=sigma
    )
    return ColumnMeasurement(self.attribute, transform_fn, measurement)
