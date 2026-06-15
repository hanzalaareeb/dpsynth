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

from __future__ import annotations

import dataclasses
from typing import TypeVar

import dp_accounting
from dpsynth import domain
from dpsynth import transformations
from dpsynth.local_mode import primitives
import mbi
import numpy as np

_M = TypeVar('_M')


@dataclasses.dataclass
class ColumnMeasurement:
  categorical_attribute: domain.CategoricalAttribute
  transform_fn: transformations.DataTransformation
  measurement: mbi.LinearMeasurement | None


def _validate_mechanism(mechanism: _M | None) -> _M:
  """Validates that the mechanism has been calibrated and returns it."""
  if mechanism is None:
    raise ValueError('Must call calibrate() before using the mechanism.')
  return mechanism


@dataclasses.dataclass
class NumericalInitializer(primitives.DPMechanism):
  """Mechanism that creates the data encoding transform for numerical data.

  Internally delegates to a ``DPQuantiles`` mechanism for privacy accounting
  and quantile computation.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    num_partitions: Number of quantile partitions (must be a power of 2).
    attribute: The NumericalAttribute defining the data domain.
  """

  name: str
  num_partitions: int
  attribute: domain.NumericalAttribute
  mechanism: primitives.DPQuantiles | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> NumericalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPQuantiles(
        lower=self.attribute.min_value,
        upper=self.attribute.max_value,
        num_partitions=self.num_partitions,
    ).calibrate(zcdp_rho=zcdp_rho)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    return _validate_mechanism(self.mechanism).dp_event

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the discretization transform."""
    bucket_edges = _validate_mechanism(self.mechanism)(rng, data)
    attr, discretize_fn = transformations.create_discretize_transformation(
        self.attribute, bucket_edges
    )
    transform_fn = transformations.discrete_encoder(attr) @ discretize_fn
    return ColumnMeasurement(attr, transform_fn, None)


@dataclasses.dataclass
class CategoricalInitializer(primitives.DPMechanism):
  """Mechanism that measures a noisy histogram for categorical data.

  Internally delegates to a ``DPGaussianHistogram`` mechanism for privacy
  accounting and noise addition.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The CategoricalAttribute defining the closed domain.
  """

  name: str
  attribute: domain.CategoricalAttribute
  mechanism: primitives.DPGaussianHistogram | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> CategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPGaussianHistogram(
        domain_size=self.attribute.size,
    ).calibrate(zcdp_rho=zcdp_rho)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return _validate_mechanism(self.mechanism).dp_event

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the noisy histogram."""
    mechanism = _validate_mechanism(self.mechanism)
    transform_fn = transformations.discrete_encoder(self.attribute)
    encoded = np.array([transform_fn(v) for v in data])
    noisy_counts = mechanism(rng, encoded)
    measurement = mbi.LinearMeasurement(
        noisy_counts, (self.name,), stddev=mechanism.sigma
    )
    return ColumnMeasurement(self.attribute, transform_fn, measurement)


@dataclasses.dataclass
class OpenSetCategoricalInitializer(primitives.DPMechanism):
  """Mechanism that discovers and measures an open-set categorical domain.

  Uses Gaussian Thresholding (Algorithm 2 from the DP-SIPS paper) to privately
  select significant partitions from the data and simultaneously obtain noisy
  counts for each discovered partition. The discovered partitions, together
  with the attribute's default_value (used as a catch-all for undiscovered
  values), form a CategoricalAttribute used for downstream synthesis.

  Privacy note: Gaussian Thresholding is an approximate (delta > 0) mechanism,
  but ``dp_accounting`` does not currently support approximate DpEvents. As a
  workaround, ``dp_event`` returns a pure GaussianDpEvent (GDP), and ``delta``
  is stored in the dataclass so that callers can subtract it from the overall
  delta budget separately.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The OpenSetCategoricalAttribute specifying the default value.
    delta: Failure probability for the partition selection threshold. Must be
      subtracted from the overall delta budget by the caller, since it is not
      captured in the DpEvent returned by ``dp_event``.
  """

  name: str
  attribute: domain.OpenSetCategoricalAttribute
  delta: float
  mechanism: primitives.DPGaussianHistogram | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> OpenSetCategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPGaussianHistogram(
        domain_size=0,
    ).calibrate(zcdp_rho=zcdp_rho)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for the thresholding mechanism."""
    return _validate_mechanism(self.mechanism).dp_event

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a differentially private measurement of the given data."""
    sigma = _validate_mechanism(self.mechanism).sigma
    gdp_budget = np.inf if sigma == 0.0 else 1.0 / (sigma**2)
    # Map raw values to integer partition IDs for thresholding.
    unique_values, inverse = np.unique(data, return_inverse=True)
    selected_ids, counts, stddev = (
        primitives.select_partitions_gaussian_thresholding(
            rng, inverse, gdp_budget, self.delta
        )
    )
    selected_values = list(unique_values[selected_ids])

    # Build the discovered domain: default first, then selected values.
    possible_values = [self.attribute.default_value] + selected_values
    cat_attr = domain.CategoricalAttribute(
        possible_values=possible_values,
        out_of_domain_index=0,
    )
    transform_fn = transformations.discrete_encoder(cat_attr)

    # The measurement covers only the discovered partitions (indices 1:),
    # not the unmeasured default at index 0.
    measurement = mbi.LinearMeasurement(
        counts,
        (self.name,),
        stddev=stddev,
        query=lambda x: x[1:],
    )
    return ColumnMeasurement(cat_attr, transform_fn, measurement)
