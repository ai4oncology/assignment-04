# MIT-BIH Arrhythmia Database (compact channel-0 copy)

## What this is

`mitbih_ch0.npz` is a compact, lossless copy of **channel 0 (lead MLII)** of every
record in the MIT-BIH Arrhythmia Database, plus the beat annotations. It is the data
used by `week9_demo_timeseries.py` (the Normal vs PVC 1D-convolution demo).

## Source and citation

MIT-BIH Arrhythmia Database v1.0.0, distributed by PhysioNet:

- https://physionet.org/content/mitdb/1.0.0/
- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering
  in Medicine and Biology* 20(3):45-50, 2001.
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a New Research Resource for Complex Physiologic Signals.
  *Circulation* 101(23):e215-e220, 2000.

License: Open Data Commons Attribution License v1.0 (ODC-By 1.0).

## What we did and why

The original download is the full WFDB folder (`mit-bih-arrhythmia-database-1.0.0/`):
48 records, each with **two channels** stored in 12-bit packed format-212, plus
`.hea`/`.atr`/`.xws` files. That is about **107 MB**.

The demo only ever reads **channel 0**, so we:

1. Decoded format-212 and kept **only channel 0**, stored losslessly as `int16`.
2. Bundled the beat annotations (sample index + symbol) for each record.
3. Saved everything with `np.savez_compressed`.

Result: **107 MB to 29 MB** (about 3.7x smaller), it loads instantly with `np.load`
(no slow per-sample decoding), and the extracted beats and model results are
**byte-identical** to the full database (Normal 28488, PVC 7122, total 35610 beats).

The raw WFDB folder has been removed from the repo to save space.

## Layout of the npz

For each record `R` (e.g. `100`, `119`, ...):

- `R_sig`: channel-0 signal, `int16`, full length (~650000 samples at 360 Hz)
- `R_ann`: beat-annotation sample indices, `int32`
- `R_sym`: beat symbols, `<U1` (e.g. `N`, `V`, `L`, `R`, ...)

## Regenerating

Re-download the raw database from PhysioNet, then for each record: decode the WFDB
format-212 `.dat`, keep channel 0 as `int16`, parse the `.atr` beat annotations
(sample index + symbol), and write all arrays with `np.savez_compressed` under the
keys `R_sig` / `R_ann` / `R_sym`. The demo's loader cell expects exactly those keys.
