# Neural-versus-ARX Comparison Identity Guard

## Purpose

The frozen comparison analysis validates complete files and exact row pairs, but its
generic runner does not select one particular completed study instance. This separate,
outcome-blind guard binds the intended ownership-corrected v5 transport collection,
the matching deterministic-neural evaluation, the schedule-matched ARX prelock and
training grid, and comparison prelock v2.

It does not alter the immutable comparison source or decision rules.

## Before result access

The guard requires:

- transport prelock
  `50dbd5d24537b61e109ff6634361ddb9ca9bceac2528b57394125a6667d80094`;
- v5 readiness
  `d245795503482417ac1d717782f33c56d05b0fa96d72f5156d5e954d4cdba74b`;
- the completed v5 manifest, attempt, completion, readiness, and public-receipt hashes;
- the exact ARX prelock, public receipt, training grid, source lock, full training-tree
  inventory, transport binding, configuration, and all 15 model-file hashes;
- comparison prelock v2
  `812db4961dfd98424c045db8bc1812662874e109b95d7ca650e77d3588e85e11`
  and its revision-pinned public receipt;
- exact evaluator completion schemas, complete file inventories, core-file hashes, and
  semantic provenance bindings.

Every check above runs before the frozen comparison runner can call `pandas.read_csv`.
The guard itself never parses either evaluator core CSV.

## Outputs

Execution uses an unoccupied staging directory. After the immutable comparison
completes, the guard writes a provenance record and completion inventory, seals every
file and directory read-only, and atomically renames the staging tree. Existing output
paths are rejected before any input result can be read.

The output interface is:

- provenance: `identity_guard_provenance.json`, with
  `identity_wrapper_prelock_registry_sha256`,
  `identity_wrapper_public_freeze_receipt_file_sha256`,
  `neural_evaluation_completion_file_sha256`,
  `arx_evaluation_completion_file_sha256`, and `v5_readiness_sha256`;
- completion: `identity_guard_complete.json`, with `complete: true` and a complete
  inventory of every other output file; and
- frozen comparison output: the `comparison/` subdirectory, whose completion is
  `comparison/analysis_complete.json`.

The guard has its own metadata/source-only local prelock and revision-pinned public
freeze. No trajectory, evaluator result, or comparison result is included in that
public-freeze set.
