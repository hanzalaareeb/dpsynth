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

"""Bulk LLM inference for synthetic text generation."""

from collections.abc import Sequence
import dataclasses
import enum
import re
from typing import Protocol

from absl import logging
from google import genai
from google.genai import types
import pandas as pd
import pydantic


class ModelName(enum.StrEnum):
  """Model names supported by the google.genai API."""

  GEMINI_2_5_FLASH_LITE = 'gemini-2.5-flash-lite'
  GEMINI_3_FLASH = 'gemini-3-flash-preview'
  GEMINI_3_5_FLASH = 'gemini-3.5-flash'
  GEMMA_4_27B = 'gemma-4-26b-a4b-it'
  GEMMA_4_31B = 'gemma-4-31b-it'


class TextGenerationBackend(Protocol):
  """Interface for bulk LLM inference operations.

  Implementations provide the two LLM inference capabilities needed by the
  synthetic text generation pipeline:

  1. **Annotation**: extracting structured categorical features from text,
     typically via constrained decoding with a pydantic response schema.
  2. **Generation**: producing free-form synthetic text conditioned on features.

  **Index-alignment guarantee**: both methods return output that is
  positionally aligned with the input sequence — ``len(output) == len(input)``
  always holds.  Failed items are represented as null rows (annotation) or
  empty strings (generation).
  """

  def annotate(
      self,
      texts: Sequence[str],
      schema: type[pydantic.BaseModel],
      system_prompt: str,
  ) -> pd.DataFrame:
    """Extract structured features from texts using an LLM.

    Args:
      texts: Input texts to annotate.
      schema: Pydantic model class defining the features to extract. The model's
        field names, ``Literal`` type annotations, and field descriptions guide
        the LLM.  This same class is used as the ``response_schema`` for
        constrained decoding in supported backends.
      system_prompt: System-level instructions for the LLM describing how to
        annotate the texts.

    Returns:
      A DataFrame with exactly ``len(texts)`` rows and one column per field
      in ``schema``.  Rows where annotation failed contain ``None`` values.
    """
    ...

  def generate(self, prompts: Sequence[str]) -> list[str]:
    """Generate free-form text from prompts.

    Args:
      prompts: Fully constructed prompts, each describing the desired output
        including the target features and any formatting requirements.

    Returns:
      A list of exactly ``len(prompts)`` strings.  Failed generations are
      represented as empty strings.
    """
    ...


@dataclasses.dataclass
class GenAIBackend:
  """TextGenerationBackend using the google.genai API.

  Uses ``client.models.generate_content()`` for both annotation (with
  structured output via ``response_schema``) and free-form generation.

  Attributes:
    model: Model name string (e.g., ``'gemini-2.5-flash-lite'``).  Accepts any
      ``ModelName`` enum value or arbitrary string for unlisted models.
    api_key: API key for authentication.  If None, uses Application Default
      Credentials (ADC).
  """

  model: str = ModelName.GEMINI_2_5_FLASH_LITE
  api_key: str | None = None

  def _make_client(self) -> genai.Client:
    """Creates a genai client."""
    kwargs: dict[str, object] = {
        'http_options': types.HttpOptions(api_version='v1alpha'),
    }
    if self.api_key is not None:
      kwargs['api_key'] = self.api_key
    return genai.Client(**kwargs)

  def annotate(
      self,
      texts: Sequence[str],
      schema: type[pydantic.BaseModel],
      system_prompt: str,
  ) -> pd.DataFrame:
    """Extract structured features via constrained decoding.

    Args:
      texts: Input texts to annotate.
      schema: Pydantic model used as the ``response_schema`` for constrained
        decoding.
      system_prompt: System-level instructions for the LLM.

    Returns:
      DataFrame with exactly ``len(texts)`` rows.  Failed rows have ``None``.
    """
    client = self._make_client()
    field_names = list(schema.model_fields.keys())
    null_row = {f: None for f in field_names}
    rows: list[dict[str, str | None]] = []
    for i, text in enumerate(texts):
      try:
        response = client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type='application/json',
                response_schema=schema,
            ),
        )
        if response.text:
          cleaned = _strip_markdown_fences(response.text)
          parsed = schema.model_validate_json(cleaned)
          rows.append(parsed.model_dump())
        else:
          logging.warning('Empty annotation response for text %d.', i)
          rows.append(null_row)
      except Exception:  # pylint: disable=broad-except
        logging.warning('Annotation failed for text %d.', i)
        rows.append(null_row)
    return pd.DataFrame(rows)

  def generate(self, prompts: Sequence[str]) -> list[str]:
    """Generate free-form text via google.genai.

    Args:
      prompts: Fully constructed prompts.

    Returns:
      List of exactly ``len(prompts)`` strings.  Empty string on failure.
    """
    client = self._make_client()
    results: list[str] = []
    for i, prompt in enumerate(prompts):
      try:
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        results.append(response.text or '')
      except Exception:  # pylint: disable=broad-except
        logging.warning('Generation failed for prompt %d.', i)
        results.append('')
    return results


def _strip_markdown_fences(text):
  """Strips markdown code fences from LLM output if present."""
  regex = r'^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$'
  m = re.compile(regex, re.DOTALL).match(text)
  return m.group(1).strip() if m else text.strip()
