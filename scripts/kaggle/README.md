# Running a probe on Kaggle

The Camber student plan ran out of CPU hours before the matched-context arm could
be submitted. Kaggle gives thirty hours of T4 a week, and this arm needs half of
one, so the code goes there. Nothing about the science changes; only where the
forward passes happen.

## Building the bundle

The panels live outside this repository and Kaggle has no way to reach them, so
code and data travel together:

    scripts/kaggle/bundle.sh          # writes kaggle-rampart/rampart-bundle.zip

Under three megabytes. Upload it as a Kaggle Dataset once; notebooks attach it.

## The notebook

Paste `kaggle_r3c.py` as a single cell, and in the notebook settings:

| setting | value | why |
|---|---|---|
| Accelerator | GPU T4 x1 | TabPFN on CPU is about ten times slower |
| Internet | On | pip, and TabPFN fetches its weights |
| Add Input | the dataset with the bundle | code and panels |

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
