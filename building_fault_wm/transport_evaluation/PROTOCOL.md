# Direct-H8 evaluation-only recovery

## Scope

This module repairs one metadata-dispatch defect in the v5 collection adapter.
The v5 adapter replaced the frozen external-freeze validator globally. Because
the v5 runner's terminal-v4 audit imports the same module object, that audit was
incorrectly sent to the v5-only validator and failed before trajectory loading.

The recovery hook has one rule:

- preserve the published v5 adapter as the sole router for v5 readiness and
  receipt validation;
- expose the original frozen validator through a temporary runner alias for
  only the exact terminal-v4 receipt tuple;
- reject every mixed or third identity at that alias.

The hook does not replace or edit the v5 adapter, corpus loading, model loading,
inference, metrics, bootstrap, gates, checkpoints, trajectory files, or result writing.
The imported frozen `run_evaluation` and `verify_only` functions are
byte-verified before and after use.

## Ordering

1. Record and verify the metadata-only terminal closeout of the failed v5
   invocation.
2. Prepare the local recovery prelock. This copies only source and metadata.
3. Independently audit the local prelock.
4. Publish and revision-pin the exact prelock files.
5. Run `adapter run-verify` in a persistent service using a fresh v6 output
   namespace.

The public-freeze step is intentionally separate. Preparing or verifying this
recovery prelock must not parse any trajectory or result CSV into structured or
numerical values. The inherited terminal-v4 integrity audit necessarily opens
legacy `data_v7/locked_transport_raw` CSV files only to recompute their fixed
SHA-256 evidence. It does not parse those CSVs, does not open v5 `data`
locked trajectory CSVs, and does not open any evaluation-result CSV.

An unpublished local v1 prelock was rejected before evaluation because its
wording denied every CSV open even though that inherited raw-byte hash audit was
present. An unpublished local v2 prelock was then rejected because its live
rejection validator treated the shared future v6 output namespace as
permanently absent, which would have invalidated legitimate post-run
verification. Both byte trees remain preserved. Publication or execution under
either identity is prohibited, and their rejection chain is bound into this v3
identity. Historical output absence is recorded at each rejection; persistent
live checks cover only version-specific public receipts and attempt state.

The persistent command obtains an exclusive lock and writes a one-shot attempt
marker only after live public-freeze verification. The marker records the
trusted GitHub verification time, not the unsynchronized host clock. Any
exception writes a terminal failure marker; success requires the frozen
verifier to pass before a completion marker is written. An interrupted or
failed attempt cannot be retried under the same recovery digest.
