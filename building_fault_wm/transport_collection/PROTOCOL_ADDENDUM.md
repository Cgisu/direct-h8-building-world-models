# Direct-H8 Transport Collection Operational Addendum v5

## Scope

This is an operational replacement for the failed v4 collection attempt. It
does not change the scientific protocol, plan grid, disjointness certificate,
models, metrics, gate, or numerical evaluation. Those remain bound by the
original scientific prelock:

`50dbd5d24537b61e109ff6634361ddb9ca9bceac2528b57394125a6667d80094`

The replacement exists only because the v4 Docker workers created output owned
by a different host identity. Collection completed inside the workers, but host
publication failed with `PermissionError`. The v4 readiness digest is terminal
and must never be retried:

`2bb8caf76d635189d1b0c738eca6a332a42639b63be7e5a8b3a9fee540e2df38`

## Operational delta

1. The worker command is byte-for-byte the frozen command except that a
   successful worker invocation ends with `chown -R 1005:1006 /out`.
2. Readiness requires host UID 1005 and GID 1006 and a disposable Docker bind
   mount probe proving that recursive ownership transfer works.
3. Plans and the disjointness certificate are copied only from the frozen
   prelock bundle into the new `data` namespace.
4. Raw trajectories are recollected in full. No file, receipt, manifest, or
   trajectory from `data_v7` may be reused.
5. Attempt state is isolated in `state_v3`. A failed readiness digest is
   terminal and cannot be retried.
6. Readiness and its public GitHub Gist revision live in `freeze_v5`. The
   public revision binds this addendum, the readiness report, the runner,
   freeze validator, evaluation adapter, original prelock, certificate, and
   immutable v4 terminal-failure evidence.
7. Post-collection evaluation invokes the frozen `run_evaluation.py` numerical
   path unchanged. The adapter may replace only readiness-metadata loading and
   external-freeze validation.

## Order of operations

1. Stage plan and certificate copies from the frozen prelock.
2. Generate and write readiness after all identity, source, runtime, and
   ownership checks pass.
3. Create a public, revision-pinned GitHub Gist and verify it live.
4. Start collection once for the resulting readiness digest.
5. On success, run and independently verify the frozen evaluation through the
   adapter.

Steps 2 through 5 are outcome-blind until the immutable attempt marker has been
written. Any failed collection requires a new operational diagnosis, new code
hashes, a new readiness digest, and a new external revision before a fresh full
recollection.
