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

# Forked from //research/privacy/synthetic_tabular_data/google/prompts.py
# with unused prompts removed.

"""Prompt builders for LLM-based annotation and conditional text generation."""

import textwrap
import typing


def _pydantic_to_text_description(pydantic_dataclass) -> str:
  """Converts a pydantic model into a numbered feature description."""
  definitions = []
  fields = pydantic_dataclass.__pydantic_fields__
  for i, name in enumerate(fields):
    description = fields[name].description
    possible_values = typing.get_args(fields[name].annotation)
    feature_str = (
        f'{i+1}.  **{name}**: \n'
        f'    * Description: {description}\n'
        f'    * Possible Values: {list(possible_values)}'
    )
    definitions.append(feature_str)
  return '\n\n'.join(definitions)


def annotate_features_prompt(
    dataset_description: str,
    dataclass,
    text: str,
) -> str:
  """Returns a prompt for LLM-based feature extraction from text.

  Constructs a prompt that instructs an LLM to annotate a given text
  based on a predefined set of features derived from a Pydantic dataclass.
  The LLM is instructed to output a JSON object containing the annotated
  features.

  Args:
    dataset_description: A 1-2 sentence description of the dataset.
    dataclass: A Pydantic dataclass defining the features to be extracted. Each
      field should use a ``Literal`` type annotation to specify the possible
      values.
    text: The text to be annotated.  Use ``'{{text}}'`` as a placeholder when
      constructing a prompt template for batch inference (e.g., TuneLab).

  Returns:
    A detailed prompt suitable for use with frontier language models.
  """
  return textwrap.dedent("""\
      You are an expert annotation system. Your task is to analyze the text \
in the 'Example to Annotate' section and classify it according to the \
features provided.

      Your output MUST be a single, valid JSON object. Do not include any \
explanatory text before or after the JSON.

      ### Rules for Annotation
      - The JSON object must contain a key for every feature listed.
      - The value for each key must be **exactly** one of the choices from \
its 'Possible Values' list.
      - Do not add new values or modify the existing ones.
      - Pay close attention to any conditional logic described in the \
feature definitions.

      ---

      ### Dataset Description
      {dataset_description}

      ---

      ### Features
      {feature_list}

      ---

      ### Example to Annotate
      {text}
  """).format(
      dataset_description=dataset_description,
      feature_list=_pydantic_to_text_description(dataclass),
      text=text,
  )


def conditional_generation_prompt(
    dataset_description: str,
    target_features: str,
    formatting_requirements: str,
    exemplar: tuple[dict[str, str], str] | None = None,
    target: str = '',
) -> str:
  """Constructs a prompt for conditional text generation.

  Generates a prompt that instructs an LLM to produce text adhering to
  specific formatting rules and matching a set of target features.

  Args:
    dataset_description: A 1-2 sentence description of the dataset.
    target_features: A string listing the desired feature values for the text to
      be generated (e.g., formatted as a markdown list).
    formatting_requirements: A string describing the formatting rules that the
      generated text must follow.
    exemplar: An optional tuple of (attributes_dict, example_text) providing a
      demonstration of the task.
    target: An optional string representing the ground truth text (for
      supervised fine-tuning).  Leave empty for inference mode.

  Returns:
    A detailed prompt suitable for use with frontier language models.
  """

  def _format_attributes(attributes):
    """Formats a dictionary of attributes into a markdown list."""
    if not attributes:
      return 'N/A'
    return '\n'.join(f'- {name}: {value}' for name, value in attributes.items())

  main_instructions = textwrap.dedent("""\
    Your primary task is to generate a text sample that follows a strict
    set of rules.

    # Rules Hierarchy
    You MUST adhere to the following rules in order of priority:
    1.  **Strictly Conform**: The output MUST follow all rules listed in the
        "Formatting Requirements" section. This is not optional.
    2.  **Best Effort**: The output should try to conform to all \
"Target Attributes"
        as closely as possible, but not at the expense of the formatting rules.

    # Context
    ## Dataset Description
    {dataset_description}

    ## Formatting Requirements
    {formatting_requirements}""").format(
      dataset_description=dataset_description,
      formatting_requirements=formatting_requirements,
  )
  prompt_parts = [main_instructions]

  if exemplar:
    example_attributes, example_text = exemplar
    example_block = textwrap.dedent("""\
      # Completed Example
      This is a demonstration of the task. Observe how the attributes map to \
the
      text while respecting the formatting requirements.

      ## Example Attributes:
      {attributes}

      ## Example Text:
      {text}""").format(
        attributes=_format_attributes(example_attributes),
        text=example_text,
    )
    prompt_parts.append(example_block)

  task_block = textwrap.dedent("""\
    # Your Task
    Now, generate a NEW text sample using the following attributes.

    ## Target Attributes:
    {target_features}

    ## Generated Text:
    {target}""").format(
      target_features=target_features,
      target=target,
  )
  prompt_parts.append(task_block)

  return '\n\n'.join(prompt_parts)
