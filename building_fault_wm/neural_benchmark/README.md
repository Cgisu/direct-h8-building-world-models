# Neural benchmark

Fault generation, training utilities, the recurrent state-space training and
evaluation entry points, and the evaluation corpus for the study.

## Contents

```
data/                    72-trajectory locked-test transport corpus
  manifests/             corpus manifest and per-scope receipts
  locked_transport_raw/  the trajectories, with worker receipts
  disjointness_certificate.json
study_config.py          study contract: arms, dimensions, training budget
reliability_loss.py      observation, KL, overshooting, and direct-H8 terms
reliability_model.py     recurrent state-space model and its gate modes
fault_data.py            synthetic sensor-fault generation
protocol.py              simulator plans and window contracts
```

`study_config.py` holds the study contract. `ARM_CONFIGS` defines the two
evaluated arms as `legacy = ("bypass", 0.0, 0.0)` and
`ungated_h8 = ("bypass", 0.0, 1.0)`: both use the availability bypass filter
rather than the learned reliability gate, and they differ only in the direct-H8
loss weight. `StudyConfig.__post_init__` checks the horizon, update schedule,
seed distinctness, and the relation
`gru_batch_size == batch_size * (sequence_length - direct_horizon)`.

## The corpus

`data/` holds the locked-test transport corpus; its manifest records
`output_role: locked_test` and `collection_kind: paired_locked_transport`. These
are the response-unseen windows the models are evaluated on. No model is fitted
on any trajectory here.

## Running

The verification in the repository root covers what can run from this package:

```bash
PYTHON=.venv/bin/python bash scripts/verify_all.sh
```

The original collection, development-screening, pre-lock, and confirmation
commands do not run from this repository. They read development trajectories,
checkpoints, and intermediate run directories that are held in the archive
deposit because of their size; see the root `README.md`.
