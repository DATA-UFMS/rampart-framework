# Running a probe on Kaggle

The Camber student plan ran out of CPU hours before the matched-context arm could
be submitted. Kaggle gives thirty hours of T4 a week, and this arm needs half of
one, so the code goes there. Nothing about the science changes; only where the
forward passes happen.

## Building the bundle

The panels live outside this repository and Kaggle has no way to reach them, so
code and data travel together:

    scripts/kaggle/bundle.sh          # writes kaggle-rampart/rampart-bundle.zip

Under three megabytes. Upload it as a Kaggle Dataset once; kernels attach it.

Kaggle expands a zip on upload, so the dataset holds the tree rather than the
archive. `kaggle_r3c.py` accepts both arrivals, and both are covered by the
resolution test, because finding out on Kaggle costs a run.

## Running it

    export KAGGLE_API_TOKEN=KGAT_...        # Settings -> API -> Create New Token
    scripts/kaggle/push_and_run.sh          # push, wait, print the log

The `KGAT_` token goes in that variable. It is not the older `kaggle.json` key and
the CLI will not read it from that file.

The kernel settings are declared in `kernel-metadata.json`, not clicked in the web
UI: `enable_gpu` with `machine_shape` T4, `enable_internet` for pip and TabPFN's
weights, and `dataset_sources` for the bundle. The cell stays a plain `.py` so it
is greppable and diffable; the runner generates the notebook from it on push.

It runs the two-minute guard first and the thirty-minute arm second. The guard
exercises the whole path -- panel loader, the chronological-order contract the
context cap depends on, the factory, a clean fit, absorption appending rows to an
already-capped frame, an injected arm appending more, and the registered
sensitivity rule. Seven Camber jobs died one at a time before it existed; it now
runs before anything expensive, anywhere.

## What this arm answers

On INEP the in-context models read ten thousand rows and the classical models read
thirty-eight thousand, so the absorption column is measured at two sample sizes
and the comparison across families is confounded. `RAMPART_CAP_ALL=1` caps every
model to the same ten thousand and the column becomes comparable.

It does not touch the conclusion that in-context absorption is not invariant in n.
That one compares the in-context models against themselves across panels, 400 rows
to 10,000, and needs no matching.
