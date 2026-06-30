"""Part 1 - A: time-series data -- classify single heartbeats from the raw ECG.

You build the MIT-BIH Normal-vs-PVC beat classifier end to end: cut beats out of
the raw signal, a flat summary-statistic baseline, a shallow 1D CNN and a dilated
TCN (the WaveNet / GluNet family), and a leak-free train/test split. The lesson:
the waveform *shape* is the signal, a generic summary baseline is hard to beat for
the wrong reasons, and under a leak-free split a bigger receptive field helps only
a little.

Each beat carries the `record` (≈ patient) it was extracted from. If you split
*individual beats* at random, beats from the same record land in both train and
test, and a CNN can memorise that record's lead placement and baseline noise and
"recognise" it at test time, inflating the score. The fix is to hold out **whole
records**: every beat of a record stays on one side. This is the same group-aware
splitting idea used for the patient trajectories in `longitudinal_exercise.py`.

What to implement
-----------------
Seven auto-graded functions: `extract_beats` (cut z-scored windows around the
R-peaks), `ecg_summary_features` (the flat, shape-blind baseline), `make_beat_cnn`
and `make_tcn` (the two models), `train_cnn` (the training loop), `receptive_field`
(how far a dilated stack sees), and `split_by_record` (the leak-free split). See
the stubs below.

Run the tests with:

    pip install -r requirements.txt
    pytest -v tests/
"""
from __future__ import annotations

import numpy as np


def split_by_record(record_ids, test_frac: float = 0.2, seed: int = 0):
    """Leak-free train/test split for the Part 1-A ECG beats.

    `record_ids[i]` is the recording (≈ patient) that beat *i* came from. Return
    a **boolean** numpy array `is_test`, one entry per beat, that holds out
    **whole records**: no record may have beats on both sides. A per-beat random
    split lets a CNN memorise a record's lead placement and noise and recognise it
    at test time, inflating the score; grouping by record forces honest
    generalisation to unseen recordings.

    Put roughly `test_frac` of the *unique records* (not beats) in the test set,
    and use `seed` so the split is reproducible.
    """
    # TODO: choose a subset of the UNIQUE record ids for test (use `seed` via
    # np.random.default_rng), then mark every beat whose record is in that subset.
    # Return a bool array the same length as record_ids (not a list of indices).
    raise NotImplementedError


def ecg_summary_features(X):
    """Flat summary-statistics baseline for the ECG beats.

    `X` has shape (n_beats, 1, beat_len) (one channel). For each beat, return a
    fixed-size feature vector of generic statistics, **not** the raw waveform:
    `[mean, std, min, max, last_value]` over the beat's samples. Stack them into
    a (n_beats, 5) array.

    This is the deliberately shape-blind baseline: it throws away the QRS
    *morphology* that distinguishes a PVC, so the 1D CNN (which sees the shape)
    should beat it. Same idea as `summary_features` in the longitudinal task.
    """
    # TODO: take the single channel X[:, 0, :] and column-stack the five stats.
    raise NotImplementedError


def extract_beats(signal, peaks, before, after):
    """Cut one fixed-length heartbeat window around each R-peak.

    `signal` is a 1-D ECG trace; `peaks` are R-peak sample indices. For each peak
    `p`, take the window `signal[p - before : p + after]` (length `before + after`),
    z-score it (subtract its mean, divide by its std + 1e-6), and collect it. Skip
    any peak whose window runs off either end of the signal.

    Returns
    -------
    beats : float32 array, shape (n_kept, before + after) -- the z-scored windows.
    keep  : bool array, shape (len(peaks),) -- True where the window was in-bounds
            (so the caller can line `beats` back up with each peak's label).
    """
    # TODO: loop the peaks, drop out-of-bounds windows, z-score each kept window;
    # return (beats array, boolean keep mask aligned with `peaks`).
    raise NotImplementedError


def make_beat_cnn():
    """A shallow 1D CNN over a single beat: two conv layers then a linear head.

    Return a fresh `torch.nn.Module` taking input shape (batch, 1, beat_len) and
    returning one logit per beat (shape `(batch,)`). A small stack works well, e.g.
    Conv1d(1->16, k=5) - ReLU - Conv1d(16->32, k=5) - ReLU - global max-pool -
    Linear(32->1). Its receptive field is only a few samples, so it sees a local
    sliver of the waveform.
    """
    # TODO: build and return a small nn.Module; forward returns (batch,) logits.
    raise NotImplementedError


def make_tcn(channels: int = 24, dilations=(1, 2, 4, 8, 16)):
    """A dilated 1D CNN (TCN, the WaveNet / GluNet family from the lecture).

    Return a fresh `torch.nn.Module`, same input/output shape as `make_beat_cnn`,
    but built from a stack of **dilated** conv blocks (dilations 1, 2, 4, 8, 16).
    Doubling the dilation each layer grows the receptive field exponentially, so a
    few layers span the whole wide-QRS shape of a PVC. End with a global max-pool
    and a linear head.
    """
    # TODO: stack dilated conv blocks (optionally residual), pool, linear head;
    # forward returns (batch,) logits.
    raise NotImplementedError


def train_cnn(make, Xtr, ytr, Xte, epochs: int = 12, seed: int = 0):
    """Train a beat classifier and return its probabilities on the test beats.

    `make` is a zero-arg factory (e.g. `make_beat_cnn`) returning a fresh model.
    Seed torch with `seed`, train on `(Xtr, ytr)` for `epochs` (Adam + a
    class-weighted `BCEWithLogitsLoss`, since PVCs are rare), then return a 1-D
    numpy array of sigmoid probabilities for `Xte`.
    """
    # TODO: torch.manual_seed(seed); train make() with Adam + BCEWithLogitsLoss
    # (pos_weight for the imbalance); return sigmoid(model(Xte)) as numpy.
    raise NotImplementedError


def receptive_field(kernel: int, dilations) -> int:
    """How many input samples one output sees, for a stack of dilated convs.

    One convolution per entry in `dilations`. Each layer adds `(kernel - 1) *
    dilation` to the field, which starts at 1. So two k=5 layers (dilations 1, 1)
    reach 9 samples; a 1-2-4-8-16 stack of k=3 convs reaches 63.
    """
    # TODO: return 1 + sum((kernel - 1) * d for d in dilations).
    raise NotImplementedError
