# DPSynth vs. DPSDA Benchmark Audit

**Audit date:** 2026-09-05
**Scope:** the Adult benchmark configuration, DPSynth/AIM and DPSDA/Tab-PE adapters, Stage-5 evaluator, generated CSV files, and saved evaluation reports in this repository.

## Executive conclusion

The implementation is a useful **single-run prototype**, but it is not yet a defensible comparative benchmark.

The strongest parts are the shared train/test files, pinned DPSDA source revision, common privacy parameters, common output schema and row count, domain validation, deterministic DPSynth seed plumbing, and use of an untouched real test set for evaluation. The DPSDA adapter also closely reproduces Microsoft's upstream Adult example.

However, four issues prevent publication-quality conclusions:

1. A clean end-to-end run is broken: `run_stage5_eval.py` passes `--seed` to `run_dpsda.py`, but that runner does not define a `--seed` argument.
2. Only one dataset, one privacy budget, and one seed are evaluated. The saved `manual_check_full` and current reports disagree substantially for DPSynth while referring to the same mutable synthetic-file path, and the manifests do not contain enough provenance to explain the difference.
3. The evaluator treats every integer-valued numerical feature as categorical. Consequently, fields such as `fnlwgt` produce TV distances near 1, and the reported aggregate 1-way/2-way TV and Cramer's-V distance are not suitable mixed-type metrics.
4. The benchmark omits downstream train-synthetic-test-real classification, higher-order fidelity, and persisted runtime/memory metrics. It therefore cannot test the main Tab-PE claims or establish which tool is better overall.

For the **current, reproducible Stage-5 artifacts only**, DPSynth/AIM has better measured low-order statistical fidelity than DPSDA/Tab-PE. This is a narrow result, not a general ranking.

## 1. How DPSynth is intended to be benchmarked

DPSynth does not prescribe a single external benchmark suite. Its repository explicitly invites comparison with other DP synthetic-data implementations and supplies an `eval/` engine for real-versus-synthetic tabular distribution checks. It also has separate in-memory and scalable pipeline code paths whose utility and budget splits can differ, so the selected path must be recorded. This benchmark correctly selects the in-memory `TabularConfig` path and AIM mechanism. See the [DPSynth repository documentation](https://github.com/google/dpsynth).

A sound comparison should hold constant:

- the exact private training split and untouched real test split;
- the neighbouring-dataset definition and target `(epsilon, delta)`;
- all public side information, including categorical domains, numeric bounds, and class proportions;
- output schema and synthetic sample count;
- random seeds or data splits;
- evaluation code, preprocessing, and downstream model;
- hardware and runtime measurement boundaries.

It should then evaluate several independent aspects of usefulness. A systematic DP synthetic-data benchmark uses 1-way and 2-way distributions, correlation preservation, and ML classification rather than relying on one metric ([Tao et al., 2022](https://aaai-ppai22.github.io/files/26.pdf)). The Tab-PE study similarly uses train-synthetic-test-real classification, mixed-type Wasserstein marginal distances, and representation-space precision/recall, averages over three seeded splits, and reports standard deviations ([Tab-PE experiment setup](https://arxiv.org/html/2606.08259v1#S5.SS1), [evaluation metrics](https://arxiv.org/html/2606.08259v1#A1.SS2)).

Adult is a valid conventional dataset, but it is explicitly characterized by the Tab-PE paper as being dominated by low-order relationships. A benchmark intended to examine Tab-PE's main advantage should add XOR/SCM stress tests and at least one high-order real dataset. On Adult at epsilon 1, the paper reports AIM ahead of Tab-PE on accuracy, macro-F1, and all 1/2/3-way Wasserstein metrics ([Tab-PE Adult results](https://arxiv.org/html/2606.08259v1#A2.SS4)).

## 2. What this repository currently does

| Dimension | Current setting |
|---|---|
| Private train data | Adult, 33,334 rows |
| Evaluation data | Separate Adult test set, 7,144 rows |
| Privacy | epsilon = 1.0; delta = `1 / (n ln n)` = 2.8805876962760987e-6 |
| Adjacency declared | Add/remove-one |
| DPSynth | In-memory `TabularConfig`, AIM defaults |
| DPSDA | Tab-PE, 30 schedule entries = 29 private histogram rounds, 5 sampling-stage entries, variation degree 3 |
| Synthetic output | 1,000 rows per method |
| Seeds | `[0]` only |
| Main saved metrics | Categorical TV for 1-way and 2-way marginals, chi-square p-values, Cramer's-V matrix RMSE |

The source CSV hashes exactly match `benchmarks/datasets/adult.json`. The DPSDA checkout resolves to the configured commit `9078c67995499e6769113780200bbf1d788d3d60`. The three local changes inside that checkout only narrow eager package imports to the tabular components; they do not change Tab-PE's algorithm.

The DPSDA construction matches the upstream example's mutation schedule, nearest-neighbour histogram, sampling/ranking populations, output count, epsilon, and delta ([upstream Adult example](https://github.com/microsoft/DPSDA/blob/main/example/tabular/adult.py)). The schedule length nuance is handled correctly: DPSDA treats the first schedule entry as initialization, so 30 entries lead to 29 private Gaussian histogram queries.

The class distribution is passed explicitly to DPSDA. This matches a simplifying assumption in the Tab-PE paper, which assumes class proportions are known. The paper also notes a noisy-proportion ablation ([Tab-PE algorithm and assumption](https://arxiv.org/html/2606.08259v1#S4)).

## 3. Implementation audit

### Correct or reasonable choices

- **Shared data:** both methods train on the same 33,334 rows and are evaluated on the same held-out 7,144 rows.
- **Shared privacy target:** both receive epsilon 1 and the same delta rule.
- **Pinned competitor:** the DPSDA commit is recorded, and the adapter follows the pinned upstream API.
- **Shared domains:** both see the same categorical domains and numeric bounds, consistent with Tab-PE's fairness requirement that baselines receive the same known bounds.
- **Output harmonization:** DPSDA floating values are rounded/clipped to the shared integer schema. This is DP-safe post-processing and all saved rows are in-domain with no nulls.
- **DPSynth downsampling:** DPSynth generates at the private-data cardinality and deterministically samples 1,000 rows. This is DP-safe post-processing and produces an equal evaluation sample size, although its cost and sampling design differ from Tab-PE's 1,000-candidate population.
- **Current report consistency:** re-evaluating the current DPSynth CSV reproduced the current Stage-5 aggregate values (to floating-point rounding).
- **Evaluator migration:** the edits in `bin/run_tabular_eval.py` correctly adapt report serialization from protobuf text to the repository's present dataclass/JSON representation. The `.pb` suffix is now misleading, but the stored content is valid JSON.

### Findings requiring correction

#### Critical: end-to-end seed interface mismatch

`run_stage5_eval.py` always invokes each generator with `--seed=<seed>`. `run_dpsda.py` has no such CLI option and always selects `experiment.seeds[0]`. A direct validation with `--seed=0` fails with `unrecognized arguments: --seed=0`. Existing files hide the defect because generation is skipped whenever the expected output already exists.

**Fix:** add `--seed` to the DPSDA runner, thread it through `prepare_dpsda_inputs`, and test generation through the Stage-5 orchestrator with an empty temporary output directory.

#### Critical: stale-cache and provenance ambiguity

Generation is based only on output-file existence. Changing code, configuration, dependency versions, or the source checkout does not invalidate an old CSV. Manifests record absolute file paths but not file hashes, config hash, git revisions, package lock hash, timestamps, runtime, hardware, or generator logs.

This matters in the current tree: for seed 0, the current DPSynth report has mean 1-way/2-way TV of `0.1484 / 0.2911`, while `manual_check_full` has `0.4943 / 0.7489`. Both manifests point to the same DPSynth CSV path. DPSDA's aggregate values agree across those directories. The discrepancy may be an older output/config/code version rather than nondeterminism, but the saved artifacts cannot distinguish these cases.

**Fix:** use immutable run IDs, hash inputs and outputs, capture the effective config and revisions, and support explicit `--force` versus `--reuse-if-manifest-matches` behavior.

#### High: mixed numerical columns are evaluated as categories

The evaluator's current `DataType` enum has no numerical type, and `run_tabular_eval.py` marks integer columns as `INT_CATEGORICAL`. As a result, all 15 Adult columns enter categorical 1-way/2-way marginals and Cramer's V. High-cardinality continuous-like fields dominate the aggregate: `fnlwgt` has 1-way TV `0.99986` for DPSynth and `0.99664` for DPSDA.

Those numbers do not mean both methods entirely failed to model `fnlwgt`; exact integer values almost never coincide between samples. The Tab-PE paper specifically prefers Wasserstein distance because categorical TV requires discretizing continuous fields and becomes distorted by sparse bins.

**Fix:** retain categorical TV/Cramer's V only for the eight configured categorical features plus label. Evaluate the six numeric fields using range-normalized Wasserstein distance (or a documented, shared discretizer fitted without test leakage). Add mixed-type joint metrics separately.

#### High: no downstream task utility

The Stage-5 evaluator has no train-synthetic-test-real classifier. Thus the benchmark cannot measure whether label-feature relationships remain useful, which is the primary high-order metric in the Tab-PE study. It also cannot be compared directly with the paper's Adult accuracy and macro-F1 results.

**Fix:** train the same fixed pipeline on each synthetic set and evaluate on the untouched test set. Report accuracy, macro-F1, and preferably ROC-AUC, alongside train-real-test-real and majority-class references. Use either the paper's TabICL protocol or a clearly labelled reproducible alternative such as fixed XGBoost plus one-hot preprocessing.

#### High: public class distribution is unproven and asymmetric

The configured counts `25,255 / 8,079` are the exact private training counts. Calling them public does not make them public. If they were derived from the private dataset for this run, releasing/using them consumes privacy unless covered by another mechanism. They are also supplied directly only to DPSDA; AIM must spend budget learning its label marginal.

**Fix:** cite an independent public source for the proportions, give equivalent public information to every compatible method, or privately estimate them under a shared accounted budget. Report both known-proportion and private/noisy-proportion variants if reproducing the paper.

#### High: experimental coverage is too narrow

One Adult run at epsilon 1 cannot support a broad claim about either library. There is no variance estimate, privacy–utility curve, alternate synthesizer, or high-order dataset. The paper uses three seeded splits; stronger benchmarking commonly uses at least 3–5 generator seeds per fixed split and several privacy levels.

**Fix:** minimally run epsilon in `{0.1, 1, 10}`, at least five seeds, Adult plus one more conventional dataset, and XOR/SCM or another high-order dataset. Include DPSynth's Independent and MST mechanisms as sanity baselines in addition to AIM.

#### Medium: efficiency is not actually benchmarked

DPSDA prints elapsed generation time but does not persist it. DPSynth does not time generation. Evaluation time is printed by the child evaluator but is not captured. Peak memory, hardware, thread count, and warm/cold compilation state are absent.

**Fix:** record wall time and peak RSS for both generation and evaluation under the same hardware and thread limits; separate one-time compilation/import cost from steady-state generation.

#### Medium: forward compatibility and artifact naming

The Adult domain file omits the newer explicit `type` field and currently emits 15 warnings that this will become an error. JSON reports retain a `.pb` extension after migration away from protobuf. The current local DPSynth environment also warns about JAX float32 and persistent compilation-cache behavior.

**Fix:** re-save/update the domain schema with explicit types, rename reports to `.json`, and record numerical/runtime settings.

## 4. Discussion of the saved results

Lower values are better for every metric in this table.

| Saved metric | DPSynth/AIM | DPSDA/Tab-PE | Narrow winner |
|---|---:|---:|---|
| Mean 1-way TV, all 15 columns | 0.1484 | 0.2782 | DPSynth |
| Mean 2-way TV, all 105 pairs | 0.2911 | 0.4937 | DPSynth |
| Cramer's-V matrix RMSE | 0.2908 | 0.3415 | DPSynth |
| Mean 1-way TV, true categorical columns + label only | 0.0232 | 0.0563 | DPSynth |
| Mean 2-way TV, true categorical pairs + label only | 0.0606 | 0.1128 | DPSynth |
| Mean range-normalized 1-D Wasserstein over six numeric columns* | 0.0119 | 0.0288 | DPSynth |

\*Supplementary audit calculation from the saved CSVs, not part of the checked-in Stage-5 report. Each numeric field was scaled by its configured public range before 1-D Wasserstein distance was calculated.

Both output files have 1,000 rows, 1,000 unique full rows, correct column order, no nulls, no categorical domain violations, and no numerical bound violations.

DPSynth is better on the checked-in 1-way TV for 12 of 15 fields. DPSDA is better on `age`, `fnlwgt`, and `income`. The `fnlwgt` categorical comparison is not meaningful. DPSDA's income TV is almost zero (`0.00030`) because it is explicitly instructed to emit the configured class ratio; DPSynth's is `0.01170`. That should not be presented as evidence that Tab-PE learned the label distribution more accurately.

For the categorical subset, DPSynth's mean 1-way TV is about **59% lower** and mean 2-way TV about **46% lower** than DPSDA's. For the supplementary normalized numeric 1-D Wasserstein measure, DPSynth is about **59% lower**. These observations agree directionally with the Tab-PE paper's Adult finding that AIM is stronger on low-order fidelity.

What the results do **not** establish:

- that DPSynth is generally better than DPSDA;
- that DPSynth preserves higher-order relationships better;
- that either method has better downstream predictive utility;
- that one is faster or more memory-efficient;
- that these differences survive generator randomness, data resplitting, or other privacy budgets.

Chi-square p-values should not be used to rank the methods here. They are highly sensitive to sample size and sparse/high-cardinality cells, and many assumptions of the asymptotic test are poor for these tables. Effect sizes and uncertainty intervals are more useful.

## 5. Recommended benchmark protocol

1. **Freeze inputs:** version the train/validation/test split, schema, domain source, and all public side information with hashes.
2. **Define privacy:** document add/remove adjacency, verify each tool's accountant against that relation, and account for every data-dependent preprocessing and tuning step.
3. **Use immutable runs:** one directory per dataset/method/epsilon/seed/code revision. Save effective YAML, stdout/stderr, versions, hashes, runtime, and peak memory.
4. **Run multiple conditions:** epsilon `{0.1, 1, 10}`, at least five generator seeds, common output counts, and several low- and high-order datasets.
5. **Include sanity references:** non-private train-real-test-real upper bound, public-only/zero-DP baseline where appropriate, and simple DP Independent and MST baselines.
6. **Evaluate categorical fidelity:** 1-way/2-way TV and Cramer's V only on explicitly categorical fields; report per-field and aggregate values.
7. **Evaluate numerical fidelity:** normalized 1-D Wasserstein, quantile error, and pairwise numeric correlation error; document scaling and missing-value handling.
8. **Evaluate joint/high-order fidelity:** 2/3-way mixed-type metrics plus XOR/SCM stress tests or task-specific workloads.
9. **Evaluate downstream utility:** identical preprocessing/classifier, train on each synthetic dataset, test on the same real test set; report accuracy, macro-F1 and uncertainty.
10. **Evaluate efficiency:** same hardware and resource limits; report generation time, evaluation time, peak memory, and failures/timeouts.
11. **Summarize uncertainty:** mean, standard deviation, and preferably bootstrap confidence intervals. Do not select the best seed.
12. **Make reuse explicit:** never silently reuse an artifact unless its manifest matches all input, config, code, and environment hashes.

## 6. Verification performed during this audit

- Confirmed all three Adult file SHA-256 hashes against the checked-in dataset manifest.
- Confirmed the DPSDA checkout commit and inspected its three local import-only modifications.
- Ran the DPSynth configuration validation successfully.
- Ran the DPSDA configuration/input/runner validation successfully in its own environment.
- Confirmed the orchestrated DPSDA `--seed` failure directly.
- Ran 21 existing evaluation/CLI tests successfully.
- Re-evaluated the current DPSynth synthetic CSV; its aggregate Stage-5 values match the current checked-in report.
- Checked both synthetic CSVs for row count, uniqueness, nulls, categorical-domain violations, and numeric-range violations.
- Parsed and compared both current reports and both `manual_check_full` reports.

The attempted full DPSynth regeneration was stopped after several minutes because the runner provides no bounded runtime or progress summary suitable for an audit pass; no checked-in artifact was changed. This reinforces the need to persist timing and provide controlled benchmark run metadata, but it is not evidence of a functional synthesis failure.

## Bottom line

The adapters capture the intended algorithms reasonably well, and the present CSVs are structurally valid. The current numbers support only this statement: **on one Adult split at epsilon 1 and seed 0, the saved DPSynth/AIM output has better low-order categorical and one-dimensional numerical fidelity than the saved DPSDA/Tab-PE output.** Fix seed plumbing and provenance first; then add mixed-type metrics, downstream utility, multiple seeds/privacy budgets, and high-order datasets before drawing a broader conclusion.
