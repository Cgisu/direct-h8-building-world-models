# Contributing

This repository accompanies a research manuscript. Its code, evaluation corpus,
and result tables correspond to the state the paper reports, and every file is
listed in `FILE_MANIFEST.sha256` so a reader can confirm they hold the same
version the results were computed from.

## Reporting a problem

Please open an issue for any of the following.

- **A verification failure.** Include your operating system, the Python version,
  the output of `bash scripts/verify_all.sh`, and `python -m pip freeze`.
- **A discrepancy between the paper and this repository.** Quote the manuscript
  value, the file and column you compared it against, and what you obtained.
- **Questions about the method, the sign conventions, or the downstream
  protocol.** The sign conventions are in the README and in Section 3 of the
  paper.
- **Requests for material held in the archive deposit** — checkpoints,
  step-level downstream trajectories, or the evidence packages.

## Changes to the results

Files under `results/` and `building_fault_wm/neural_benchmark/data/` are the
evidence the paper reports. Corrections to them are handled through the journal
rather than through a pull request, so please raise an issue describing what you
found instead of proposing a change to those files.

The same applies to re-runs with different seeds, budgets, or hyperparameters:
those are new experiments, and we would rather discuss them in an issue than
merge them as revisions to the reported numbers.

## Portability fixes

Pull requests that help the verification path run on more platforms are welcome
— shell portability in `scripts/verify_all.sh`, or a dependency pin that no
longer resolves on a supported Python version. Such a pull request should:

1. leave the files under `results/` and
   `building_fault_wm/neural_benchmark/data/` unchanged;
2. regenerate `FILE_MANIFEST.sha256` for the files it does change; and
3. still pass `bash scripts/verify_all.sh` and `python -m pytest -q`.

## Notes

The verification suite runs offline on CPU and needs no simulator. The full
comparator re-evaluation additionally requires the neural evidence package from
the archive deposit, placed at `external_evidence/neural_evaluation_package/`.
