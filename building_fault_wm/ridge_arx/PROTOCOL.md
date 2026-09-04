# Schedule-Matched Recursive ARX Transport Addendum

## Status and boundary

This is a secondary engineering-comparator addendum to the direct-H8 parent and
the separately frozen deterministic transport study. It cannot change either
study's gate, category, primary estimand, or interpretation.

The addendum asks one bounded question:

> Under the same scheduled development-sequence exposure and future-information
> contract, how does a parsimonious recursive linear ARX predictor transport across
> the paired two-hour and four-hour action-dwell policies?

It is not a new primary gate, a grey-box claim, a physical-parameter identification
study, a stochastic-versus-deterministic claim, or a closed-loop control experiment.

## Immutable sources

The development source is the immutable publication package
`artifacts/direct_h8_publication_v2`, canonical digest:

```text
b758859c6cb99d34930452c36e3fd59b5abd0e7f56b19710fa2b1998b23760b8
```

For each of the three cases and five model seeds, training reuses byte-for-byte:

- the parent development corpus and fault contract;
- the parent FIT scalers;
- the parent `boptest-reliability-rssm-training-schedule-v1` schedule.

The transport corpus is not a training or tuning source. Only the
ownership-corrected v5 full recollection in the separate `data`, `state_v3`,
and `freeze_v5` namespaces is admissible. The terminal `data_v7` attempt is
rejected. V5 metadata can enter the addendum lock only after a canonical
collection completion exists. No raw transport CSV is opened while training,
selecting, or preparing the addendum lock.

## Model and information

The comparator is multi-output Ridge-ARX with eight causal lags. A one-step feature
contains:

- eight standardized corrupted-observation lags, with unavailable values zeroed;
- eight availability-mask lags;
- eight `log1p(age)` lags;
- eight previous-action lags;
- the candidate current action;
- current and next known context.

There are 115 features and four standardized clean-observation targets. Including
four intercepts, the fitted model has 464 active coefficients. This deliberately
parsimonious engineering baseline is not inflated to the neural parameter budget.
The deterministic recurrent v3 comparator already provides the parameter-matched
architecture comparison.

At rollout, predictions replace unavailable future observations, with availability
one and age zero. Candidate future actions and known future contexts are provided.
Future simulator observations never enter the model.

## Exact scheduled exposure

Every parent schedule contains 400 updates and four 48-step sequence references per
update. For each referenced sequence, the ARX uses the 40 one-step sources whose
complete eight-lag history and next target lie within the scheduled 48 rows. Thus
each case/seed fit contains exactly:

```text
400 updates * 4 sequences * 40 sources = 64,000 scheduled rows
```

Repeated schedule references remain repeated and therefore retain their exact
schedule weight. Rows are additionally weighted so each fault-channel/family stratum
has equal total weight, matching the parent engineering-baseline convention.

## Selection

History is fixed at eight; it is not tuned. Ridge alpha is selected independently
for each case and schedule seed from:

```text
1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
```

Selection uses only complete parent validation trajectories. The metric is H8
standardized affected-channel MAE, averaged equally over fault family and channel.
Ties choose the smaller alpha. No transport value can affect selection or refitting.

## Transport evaluation

The byte-frozen selected models are evaluated at H1, H2, H4, and H8 on the same
paired ownership-corrected v5 corpus, fault contract, affected-channel target,
FIT scalers, 12 windows per case, and five schedule seeds. Both `old_2h` and
`new_4h` are always reported.

The addendum writes its own `schedule_matched_arx` rows and descriptive equal-weight
summaries. Any comparison with the sealed v3 model rows is secondary and is stored
only in the addendum result tree. The original v3 CSV, JSON, gate, category, and
artifact hashes remain untouched.

Permitted conclusions are symmetric:

- if a neural arm is better, complexity helped relative to this parsimonious linear
  comparator under the tested transport;
- if ARX is equivalent or better, neural or latent complexity was not required for
  this benchmark.

No result permits claims about observed buildings, physical parameter recovery,
intrinsic architectural superiority, planning, MPC, energy, cost, or comfort.

## Separate lock and external timestamp

The addendum uses a new namespace, training tree, prelock tree, external record, and
evaluation tree. Its prelock binds:

- every addendum source file;
- all 15 model files, score tables, receipts, and the training-grid receipt;
- the complete parent digest and the exact reused schedule/scaler/fault hashes;
- the completed v5 attempt, completion, readiness, operational external-freeze
  receipt, metadata adapter, corpus-manifest metadata, and their hashes;
- an explicit rejection of the terminal `data_v7` namespace.

The external-freeze inventory is prepared locally. The explicit
`create-public-freeze` command publishes those exact bytes, pins the returned
revision, writes its receipt once, and live-validates every remote byte. A valid
revision-pinned public timestamp must exist before the evaluation CLI can open
any transport trajectory. Data may already have been collected, but analysis
remains outcome-blind and is disclosed as such.
