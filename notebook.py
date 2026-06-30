"""Marimo walkthrough notebook for assignment-04 (Longitudinal & Predictive).

Open with:

    marimo edit notebook.py

Two parts, all graded:

  **Part 1 -- Clinical modeling**

  * **A. Time-series data**: a 1D convolution (and a dilated TCN) on raw ECG
    (MIT-BIH), where waveform shape is the whole signal. You build the
    Normal-vs-PVC beat classifier end to end (uses `data/mitbih_ch0.npz`),
    implementing the functions in `time_series_exercise.py`.
  * **B. Longitudinal data** (predictive): predict survival from a patient's lab
    trajectory on the PBC cohort. You implement the functions in
    `longitudinal_exercise.py`; the cells run three models on `data/pbcseq.csv`.

  **Part 2 -- Tracking**

  Chain unlabelled per-frame detections of moving cells into trajectories. You
  implement the functions in `tracking_exercise.py`; the cells run them on
  `data/detections.csv`.

The graded sections end in pen-and-paper questions whose answers auto-save to
`submission.json` for the autograder.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # Install only genuinely-missing deps (installing over a live kernel can
    # corrupt imports). On a fresh env this runs once; restart the kernel after.
    import importlib.util
    import subprocess
    import sys

    _need = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch"]
    _missing = [m for m in _need if importlib.util.find_spec(m) is None]
    if _missing:
        print(f"Installing missing dependencies: {', '.join(_missing)} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            check=False,
        )
        print("Done. Restart the kernel so imports pick up the new packages.")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    GREY = "#9AA0A6"
    return GREY, Path, np, plt


@app.cell
def _():
    # Helpers for the pen-and-paper questions (auto-save / restore answers).
    import json as _json
    from pathlib import Path as _Path

    _submission_path = _Path(__file__).with_name("submission.json")
    try:
        submission_data = (
            _json.loads(_submission_path.read_text()) if _submission_path.exists() else {}
        )
    except _json.JSONDecodeError:
        submission_data = {}


    def submission_default(key, default=None):
        return submission_data.get(key, default)


    def submission_radio_default(key, options, default=None):
        saved = submission_data.get(key, default)
        if saved in options:
            return saved
        for opt_key, opt_val in options.items():
            if opt_val == saved:
                return opt_key
        return default

    return submission_default, submission_radio_default


@app.cell
def _(mo):
    mo.md(r"""
    # Assignment 04: Longitudinal & Predictive Modeling

    Two parts on one theme, **reading signal out of data that changes over time**.

    **Part 1: Clinical modeling**

    - **A. Time-series data.** A 1D convolution (and a dilated TCN) on raw ECG
      (MIT-BIH), where waveform *shape* is the whole signal and a CNN beats a flat
      baseline. Build the beat classifier end to end. (Implement
      `time_series_exercise.py`.)
    - **B. Longitudinal data.** Predict survival from a patient's lab *trajectory*,
      and ask whether a Transformer earns its keep over a plain RNN and a logistic
      baseline. (Implement `longitudinal_exercise.py`.)

    **Part 2: Tracking**

    Frame-to-frame *data association*: chain unlabelled detections of moving cells
    into trajectories. (Implement `tracking_exercise.py`.)

    Work top to bottom. Each part ends in pen-and-paper questions that auto-save to
    `submission.json` (read by the autograder).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part 1 - A. Time-series data: MIT-BIH, where waveform shape is the whole signal

    The **MIT-BIH Arrhythmia Database** (Moody & Mark, 2001) contains 48 half-hour ECG
    recordings sampled at **360 Hz**, with every heartbeat annotated by two independent
    cardiologists. Total: ≈ **110 000 labelled beats** across 48 patients.

    **Task:** classify each beat as **Normal (N)** or **Premature Ventricular Contraction
    (V / PVC)**. A PVC produces a characteristically wide, bizarre QRS complex:
    a *local waveform shape* that varies beat to beat, invisible to a mean or standard
    deviation, and exactly what a 1D convolution detects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Medical excursus: the heart's electrical system and what a PVC is

    An **ECG** records the heart's electrical activity from skin electrodes. Each beat
    produces a stereotyped waveform:

    - **P wave**: atria contract; the SA node fires
    - **QRS complex**: ventricles contract; the main pump stroke
    - **T wave**: ventricles reset for the next beat

    In a healthy heart the impulse always follows the same fast-conducting pathway
    (SA node → AV node → Bundle of His → Purkinje fibres → muscle), so the QRS is
    **narrow** (< 120 ms) and looks identical beat to beat.

    A **PVC** is an ectopic beat: the impulse originates in the ventricular muscle itself,
    bypassing the fast pathway. The wave spreads slowly, cell to cell, producing a
    **wide (> 120 ms), morphologically bizarre QRS**: often inverted and asymmetric.
    PVCs are common in healthy adults but can indicate structural heart disease and, in
    a vulnerable myocardium, can trigger ventricular fibrillation.

    **The key point:** normal vs PVC is a *waveform shape* distinction: exactly the
    thing a 1D convolutional kernel is built to detect.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, GREY, RED, np, plt):
    from matplotlib.patches import (
        FancyBboxPatch as _FancyBboxPatch,
        FancyArrowPatch as _FancyArrowPatch,
    )

    _YEL = "#FFE863"
    fig_exc, (ax_path, ax_wave) = plt.subplots(1, 2, figsize=(11, 4.8))

    # left: conduction pathway
    ax_path.set_xlim(0, 5)
    ax_path.set_ylim(0, 6)
    ax_path.axis("off")
    _steps = [
        ("SA node fires\n(right atrium)", GREY),
        ("AV node  (gate + delay)", GREY),
        ("Bundle of His\n+ branch blocks", BLUE),
        ("Purkinje fibres → fast spread", BLUE),
        ("Ventricles contract → narrow QRS", BLUE),
    ]
    _ys = [5.1, 4.1, 3.1, 2.1, 1.1]
    _cx = 2.5
    for (_txt, _fc), _y in zip(_steps, _ys):
        ax_path.add_patch(
            _FancyBboxPatch(
                (_cx - 2.0, _y - 0.33),
                4.0,
                0.65,
                boxstyle="round,pad=0.06",
                facecolor=_fc,
                edgecolor="none",
                alpha=0.85,
                zorder=3,
            )
        )
        ax_path.text(
            _cx,
            _y,
            _txt,
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            zorder=4,
        )
    for _y0, _y1 in zip(_ys[:-1], _ys[1:]):
        ax_path.add_patch(
            _FancyArrowPatch(
                (_cx, _y0 - 0.33),
                (_cx, _y1 + 0.33),
                arrowstyle="-|>",
                mutation_scale=13,
                color="#555",
                lw=1.5,
                zorder=2,
            )
        )
    ax_path.add_patch(
        _FancyBboxPatch(
            (0.1, 2.27),
            1.6,
            0.7,
            boxstyle="round,pad=0.06",
            facecolor=RED,
            edgecolor="none",
            alpha=0.9,
            zorder=5,
        )
    )
    ax_path.text(
        0.9,
        2.62,
        "PVC:\nectopic focus",
        ha="center",
        va="center",
        color="white",
        fontsize=7.5,
        fontweight="bold",
        zorder=6,
    )
    ax_path.add_patch(
        _FancyArrowPatch(
            (1.7, 2.62),
            (_cx - 1.6, 2.62),
            arrowstyle="-|>",
            mutation_scale=11,
            color=RED,
            lw=1.8,
            connectionstyle="arc3,rad=0.3",
            zorder=5,
        )
    )
    ax_path.text(
        _cx,
        0.45,
        "bypasses fast pathway → wide QRS",
        ha="center",
        color=RED,
        fontsize=8.5,
        fontstyle="italic",
    )
    ax_path.set_title("Normal cardiac conduction\n(red = PVC ectopic origin)", fontsize=10)
    ax_path.text(
        0.0,
        1.0,
        " MEDICAL EXCURSUS ",
        transform=ax_path.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="black",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="square,pad=0.25", facecolor=_YEL, edgecolor="none"),
        zorder=8,
    )

    # right: schematic waveforms
    _t = np.linspace(0, 1, 300)


    def _g(t, mu, sig, a):
        return a * np.exp(-0.5 * ((t - mu) / sig) ** 2)


    _normal = (
        _g(_t, 0.20, 0.030, 0.30)
        + _g(_t, 0.50, 0.020, 1.00)
        - _g(_t, 0.52, 0.015, 0.60)
        + _g(_t, 0.54, 0.020, 0.90)
        + _g(_t, 0.72, 0.050, 0.35)
    )
    _pvc = (
        _g(_t, 0.45, 0.060, -0.50)
        + _g(_t, 0.50, 0.055, 1.40)
        + _g(_t, 0.58, 0.040, -0.80)
        - _g(_t, 0.73, 0.060, 0.40)
    )
    ax_wave.plot(_t, _normal, color=BLUE, lw=2.2, label="Normal beat")
    ax_wave.plot(_t, _pvc, color=RED, lw=2.2, label="PVC beat", ls="--")
    ax_wave.axvspan(0.46, 0.58, alpha=0.10, color=BLUE, label="Normal QRS")
    ax_wave.axvspan(0.40, 0.62, alpha=0.10, color=RED, label="PVC QRS (wider)")
    ax_wave.set_xlabel("time (normalised)")
    ax_wave.set_ylabel("amplitude (a.u.)")
    ax_wave.set_title("Schematic ECG: narrow normal QRS vs wide PVC QRS", fontsize=10)
    ax_wave.legend(fontsize=8, frameon=False, loc="upper left")
    ax_wave.axhline(0, color="k", lw=0.5, alpha=0.3)

    fig_exc.tight_layout()
    fig_exc
    return


@app.cell
def _(Path, mo, np):
    BEAT_BEFORE = 90  # samples before the R-peak
    BEAT_AFTER = 110  # samples after (200 total ≈ 0.55 s at 360 Hz)
    BEAT_LEN = BEAT_BEFORE + BEAT_AFTER
    MITBIH_FS = 360

    # Compact channel-0 store of the MIT-BIH database (29 MB, lossless for channel 0).
    # Bundled in this assignment under data/; provenance in data/README_mitbih.md.
    _dir = Path(__file__).with_name("data")
    _npz_path = next(
        (p for p in (_dir / "mitbih_ch0.npz", Path("mitbih_ch0.npz")) if p.exists()), None
    )

    MITBIH_OK = _npz_path is not None
    beats_N, beats_V = [], []
    X_ecg = y_ecg = rec_ecg = mitbih_record = None
    _note = None

    if not MITBIH_OK:
        _note = mo.md(
            "> ⚠️ **MIT-BIH data not found.** Part 1 needs `data/mitbih_ch0.npz` "
            "(≈ 29 MB); see `data/README_mitbih.md`. The cells below stay inert until "
            "the file is present."
        )
    else:
        try:
            import time_series_solution as _ts
        except ModuleNotFoundError:
            import time_series_exercise as _ts

        _Z = np.load(_npz_path)
        _records = sorted({k[:-4] for k in _Z.files if k.endswith("_sig")})

        def mitbih_record(rid):
            """One record: channel-0 signal (float32 ADC units), annotation samples, symbols."""
            return (_Z[f"{rid}_sig"].astype(np.float32), _Z[f"{rid}_ann"], _Z[f"{rid}_sym"])

        _rec_N, _rec_V = [], []  # which record each beat came from (for leak-free splits)
        for _rid in _records:
            _sig, _samp, _syms = mitbih_record(_rid)
            _nv = np.isin(_syms, ["N", "V"])  # keep only Normal / PVC beats
            # your beat extractor: z-scored windows around each R-peak
            _beats, _kept = _ts.extract_beats(_sig, _samp[_nv], BEAT_BEFORE, BEAT_AFTER)
            for _w, _sym in zip(_beats, _syms[_nv][_kept]):
                if _sym == "N":
                    beats_N.append(_w)
                    _rec_N.append(_rid)
                else:
                    beats_V.append(_w)
                    _rec_V.append(_rid)

        _rng_mb = np.random.default_rng(0)
        _n_keep = min(len(beats_N), 4 * len(beats_V))
        _keep = _rng_mb.choice(len(beats_N), _n_keep, replace=False)
        beats_N = [beats_N[i] for i in _keep]
        _rec_N = [_rec_N[i] for i in _keep]

        X_ecg = np.array(beats_N + beats_V, dtype=np.float32)[:, None, :]
        y_ecg = np.array([0] * len(beats_N) + [1] * len(beats_V), dtype=int)
        rec_ecg = np.array(_rec_N + _rec_V)  # record id per beat, aligned with X_ecg

        print(
            f"MIT-BIH (channel-0 npz): {len(_records)} records  |  "
            f"Normal: {len(beats_N)}  PVC: {len(beats_V)}  "
            f"Total beats: {len(y_ecg)}"
        )

    _note if _note is not None else None
    return (
        BEAT_BEFORE,
        BEAT_LEN,
        MITBIH_FS,
        MITBIH_OK,
        X_ecg,
        beats_N,
        beats_V,
        mitbih_record,
        rec_ecg,
        y_ecg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the raw ECG looks like

    Record 119 (a patient with frequent uniform PVCs): 20 seconds of lead II.
    Blue ticks mark normal beats; red stars mark PVCs. The PVC arrives early (premature),
    with a visibly wider and taller spike. The bottom panel zooms into a single N → PVC → N
    triplet.
    """)
    return


@app.cell
def _(BLUE, GREY, MITBIH_FS, MITBIH_OK, RED, mitbih_record, mo, np, plt):
    if not MITBIH_OK:
        out_raw = mo.md("*(MIT-BIH data not bundled: raw-ECG panel skipped.)*")
    else:
        _raw119, _samp119, _s119 = mitbih_record("119")
        _sig119 = _raw119 / 200.0
        _fs = MITBIH_FS

        _pvc_pos = [s for s, sym in zip(_samp119, _s119) if sym == "V"]
        _anchor = _pvc_pos[3]
        _lo = max(0, _anchor - 10 * _fs)
        _hi = min(len(_sig119), _anchor + 10 * _fs)
        _t_seg = np.arange(_hi - _lo) / _fs
        _ann_in = [
            (s - _lo, sym)
            for s, sym in zip(_samp119, _s119)
            if _lo <= s < _hi and sym in ("N", "V")
        ]

        _nv = [(s, sym) for s, sym in zip(_samp119, _s119) if sym in ("N", "V")]
        _triplet = next(
            (
                (_nv[k - 1][0], _nv[k][0], _nv[k + 1][0])
                for k in range(1, len(_nv) - 1)
                if _nv[k][1] == "V"
            ),
            None,
        )
        _zlo = _triplet[0] - int(0.3 * _fs)
        _zhi = _triplet[2] + int(0.4 * _fs)
        _t_zoom = np.arange(_zhi - _zlo) / _fs

        fig_raw, (ax_seg, ax_zoom) = plt.subplots(
            2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1.2]}
        )
        ax_seg.plot(_t_seg, _sig119[_lo:_hi], color=GREY, lw=0.7, alpha=0.9)
        for _s, _sym in _ann_in:
            _c, _mk, _ms = (RED, "*", 12) if _sym == "V" else (BLUE, "|", 8)
            ax_seg.plot(
                _s / _fs,
                _sig119[_lo + _s],
                marker=_mk,
                color=_c,
                ms=_ms,
                ls="none",
                zorder=3,
            )
        ax_seg.plot([], [], marker="|", color=BLUE, ls="none", ms=9, label="Normal (N)")
        ax_seg.plot([], [], marker="*", color=RED, ls="none", ms=12, label="PVC (V)")
        ax_seg.set_ylabel("mV")
        ax_seg.set_xlabel("seconds")
        ax_seg.set_title(
            "MIT-BIH record 119: 20-second ECG segment, lead II (MLII)", fontsize=10
        )
        ax_seg.legend(
            fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0)
        )

        ax_zoom.plot(_t_zoom, _sig119[_zlo:_zhi], color=GREY, lw=1.2)
        for (_bs, _sym), _c, _lbl in zip(
            [(_triplet[0], "N"), (_triplet[1], "V"), (_triplet[2], "N")],
            [BLUE, RED, BLUE],
            ["Normal", "PVC", "Normal"],
        ):
            ax_zoom.axvline(
                (_bs - _zlo) / _fs, color=_c, lw=1.3, ls="--", alpha=0.7, label=_lbl
            )
        ax_zoom.set_ylabel("mV")
        ax_zoom.set_xlabel("seconds")
        ax_zoom.set_title("Zoom: N → PVC → N  (dashed = R-peak annotation)", fontsize=10)
        ax_zoom.legend(
            fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0)
        )
        fig_raw.tight_layout(h_pad=1.0, rect=[0, 0, 0.84, 1])
        out_raw = fig_raw
    out_raw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extracted beat windows

    Each beat is a 200-sample window (≈ 0.55 s) centred on the annotated R-peak,
    z-scored. Normal beats (blue) are narrow and symmetric; PVC beats (red) are wide
    and morphologically diverse. This visual difference is what the CNN will learn.
    """)
    return


@app.cell
def _(
    BEAT_BEFORE,
    BEAT_LEN,
    GREY,
    MITBIH_OK,
    RED,
    beats_N,
    beats_V,
    mo,
    np,
    plt,
):
    if not MITBIH_OK:
        out_beats = mo.md("*(MIT-BIH data not bundled: beat-window panel skipped.)*")
    else:
        _n_show = 6
        _rng = np.random.default_rng(1)
        _t = np.arange(BEAT_LEN) / 360.0 * 1000  # milliseconds

        fig_beats, axes_b = plt.subplots(
            2, _n_show, figsize=(13, 4.5), sharey=True, sharex=True
        )
        for _col, _i in enumerate(_rng.choice(len(beats_N), _n_show, replace=False)):
            axes_b[0, _col].plot(_t, beats_N[_i], color=GREY, lw=1.3)
            axes_b[0, _col].axvline(
                BEAT_BEFORE / 360 * 1000, color="k", lw=0.6, ls="--", alpha=0.4
            )
            axes_b[0, _col].set_title(f"N #{_col + 1}", fontsize=9)
        for _col, _i in enumerate(_rng.choice(len(beats_V), _n_show, replace=False)):
            axes_b[1, _col].plot(_t, beats_V[_i], color=RED, lw=1.3)
            axes_b[1, _col].axvline(
                BEAT_BEFORE / 360 * 1000, color="k", lw=0.6, ls="--", alpha=0.4
            )
            axes_b[1, _col].set_title(f"PVC #{_col + 1}", fontsize=9, color=RED)
        axes_b[0, 0].set_ylabel("Normal\n(z-scored)", fontsize=9)
        axes_b[1, 0].set_ylabel("PVC\n(z-scored)", fontsize=9, color=RED)
        for _ax in axes_b[1]:
            _ax.set_xlabel("ms", fontsize=8)
        fig_beats.suptitle(
            "MIT-BIH beat waveforms: Normal vs PVC; dashed line = annotation-aligned beat center",
            fontsize=11,
        )
        fig_beats.tight_layout()
        out_beats = fig_beats
    out_beats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Your turn: split these beats without leaking

    Before benchmarking, you need a test set. **Implement `split_by_record` in
    `time_series_exercise.py`**: given `record_ids` (the MIT-BIH record each beat
    came from), return a boolean `is_test` mask that holds out *whole records*, so
    no record has beats on both sides. The check below turns green when it is
    leak-free; the autograder (`test_split_by_record_is_leak_free`) checks the same
    thing. Until you implement it, you will see the reminder below.
    """)
    return


@app.cell
def _(np, rec_ecg):
    # Uses your split from time_series_exercise.py (or the reference
    # solution if present). Reload so edits are picked up without restarting.
    import importlib as _il

    try:
        import time_series_solution as _pr_split
    except ModuleNotFoundError:
        import time_series_exercise as _pr_split
    _pr_split = _il.reload(_pr_split)

    try:
        _res = _pr_split.split_by_record(rec_ecg) if rec_ecg is not None else None
        is_test = np.asarray(_res) if _res is not None else None
        split_todo = False
    except NotImplementedError:
        is_test, split_todo = None, True
    return is_test, split_todo


@app.cell(hide_code=True)
def _(MITBIH_OK, is_test, mo, np, rec_ecg, split_todo, y_ecg):
    if not MITBIH_OK:
        out_split = mo.md("*(MIT-BIH data not bundled: split check skipped.)*")
    elif split_todo or is_test is None:
        out_split = mo.md(
            "> 🛠️ `split_by_record` is not implemented yet. Fill in the TODO in "
            "`time_series_exercise.py`, then re-run this cell."
        )
    elif np.asarray(is_test).dtype != bool or len(is_test) != len(rec_ecg):
        out_split = mo.md(
            "> ⚠️ `split_by_record` must return a **boolean** array, one entry per "
            "beat (not a list of indices)."
        )
    elif is_test.all() or not is_test.any():
        out_split = mo.md(
            "> ⚠️ Your split puts every beat on one side. Return a mix of True/False."
        )
    else:
        _train_recs = set(rec_ecg[~is_test])
        _test_recs = set(rec_ecg[is_test])
        _shared = _train_recs & _test_recs
        _n_rec = len(set(rec_ecg))
        _both_classes = (
            len(np.unique(y_ecg[~is_test])) == 2 and len(np.unique(y_ecg[is_test])) == 2
        )
        if _shared:
            out_split = mo.md(
                f"> ❌ **Leakage.** {len(_shared)} of {_n_rec} records have beats in "
                "**both** train and test. A 1D CNN can then memorise a record's lead "
                "placement and baseline noise and recognise it at test time, so the "
                "AUROC is inflated and will not hold on a new patient.\n>\n"
                "> **Fix:** split the *unique record ids*, not the beats: put whole "
                "records into train or test (e.g. `GroupKFold` / `GroupShuffleSplit` "
                "keyed on `record_ids`). Every beat of a record then stays on one side."
            )
        else:
            _msg = (
                f"> ✅ **Leak-free.** No record appears on both sides "
                f"({len(_train_recs)} train records, {len(_test_recs)} test records). "
                f"Note the test set is **{is_test.mean():.0%}** of the beats, not exactly "
                "20%: it follows whole records, so the count depends on which records you "
                "held out, not on a fixed beat fraction."
            )
            if not _both_classes:
                _msg += (
                    "  \n> (Heads up: one side is missing a class; rebalance which "
                    "records you hold out.)"
                )
            out_split = mo.md(_msg)
    out_split
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline vs shallow CNN vs dilated TCN

    Three models on the same leak-free (record-grouped) folds:

    **Baseline**: five generic summary statistics per beat (mean, std, min, max, last
    value), the same flat baseline as the MIMIC cell. They see all 200 samples but
    discard the *shape* of the QRS.

    **Shallow CNN**: two `k=5` convolutions. Its **receptive field is only ~9 of the
    200 samples**, so each feature sees a narrow window (slid across the beat by the
    final max-pool).

    **Dilated TCN**: the same idea, but with the **dilated / causal convolutions** from
    the lecture (WaveNet / GluNet / TCN family). Stacking dilations 1-2-4-8-16 grows the
    receptive field to **~63 samples**, enough to span the whole wide-QRS morphology that
    marks a PVC. The question: does seeing more of the beat at once actually help here?
    """)
    return


@app.cell
def _(BLUE, MITBIH_OK, RED, X_ecg, mo, np, plt, rec_ecg, y_ecg):
    if not MITBIH_OK:
        out_ecg = mo.md("*(MIT-BIH data not bundled: baseline-vs-CNN benchmark skipped.)*")
    else:
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.model_selection import GroupKFold as _GKF
        from sklearn.metrics import roc_auc_score as _auc

        # Your Part 1-A code: the flat baseline, the two models and the
        # training loop all come from time_series_exercise.py (or the reference).
        try:
            import time_series_solution as _ts
        except ModuleNotFoundError:
            import time_series_exercise as _ts

        F_ecg = _ts.ecg_summary_features(X_ecg)
        auc_ecg_lr, auc_ecg_cnn, auc_ecg_tcn = [], [], []
        # GroupKFold on the record id: every beat of a record stays on one side,
        # so the model is scored on patients/records it never trained on.
        for _tr, _te in _GKF(5).split(X_ecg, y_ecg, rec_ecg):
            _sc = _SS().fit(F_ecg[_tr])
            _lr = _LR(max_iter=1000, C=1.0, class_weight="balanced").fit(
                _sc.transform(F_ecg[_tr]), y_ecg[_tr]
            )
            auc_ecg_lr.append(
                _auc(y_ecg[_te], _lr.predict_proba(_sc.transform(F_ecg[_te]))[:, 1])
            )
            auc_ecg_cnn.append(
                _auc(
                    y_ecg[_te],
                    _ts.train_cnn(_ts.make_beat_cnn, X_ecg[_tr], y_ecg[_tr], X_ecg[_te]),
                )
            )
            auc_ecg_tcn.append(
                _auc(
                    y_ecg[_te],
                    _ts.train_cnn(_ts.make_tcn, X_ecg[_tr], y_ecg[_tr], X_ecg[_te]),
                )
            )

        print(
            f"MIT-BIH  logistic (5 features): {np.mean(auc_ecg_lr):.3f} ± {np.std(auc_ecg_lr):.3f}"
        )
        print(
            f"MIT-BIH  shallow CNN (RF~9):    {np.mean(auc_ecg_cnn):.3f} ± {np.std(auc_ecg_cnn):.3f}"
        )
        print(
            f"MIT-BIH  dilated TCN (RF~63):   {np.mean(auc_ecg_tcn):.3f} ± {np.std(auc_ecg_tcn):.3f}"
        )

        fig_ecg, ax_ecg = plt.subplots(figsize=(5.6, 4.5))
        _m = [np.mean(auc_ecg_lr), np.mean(auc_ecg_cnn), np.mean(auc_ecg_tcn)]
        _e = [np.std(auc_ecg_lr), np.std(auc_ecg_cnn), np.std(auc_ecg_tcn)]
        _bars = ax_ecg.bar(
            [
                "logistic\n(5 summary\nstats)",
                "shallow CNN\n(RF ~9)",
                "dilated TCN\n(RF ~63)",
            ],
            _m,
            yerr=_e,
            capsize=6,
            color=[BLUE, RED, "#2A9D8F"],
            alpha=0.85,
            error_kw={"elinewidth": 1.8},
        )
        ax_ecg.axhline(0.5, color="k", lw=0.9, ls="--", alpha=0.4)
        ax_ecg.set_ylim(0.5, 1.02)
        ax_ecg.set_ylabel("AUROC  (5-fold GroupKFold)")
        ax_ecg.set_title(
            f"MIT-BIH: Normal vs PVC  ({len(y_ecg):,} beats)\n"
            "leak-free CV: a bigger receptive field helps only a little",
            fontsize=10,
        )
        for _b, _v in zip(_bars, _m):
            ax_ecg.annotate(
                f"{_v:.3f}",
                (_b.get_x() + _b.get_width() / 2, _v + 0.005),
                ha="center",
                fontsize=13,
            )
        fig_ecg.tight_layout()
        out_ecg = fig_ecg
    out_ecg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The lesson

    Under leak-free evaluation the fancier model barely pulled ahead: the dilated
    TCN's larger receptive field bought only a point or two over the shallow CNN,
    well within the noise of a 47-record cohort. The recurring theme in clinical ML
    is not "reach for a bigger model": it is **understand the structure of the
    signal and of the data first**, then choose accordingly. The same reasoning runs
    through this lecture: **causal** vs acausal convolutions for online onset
    detection, the **sampling grid** for irregular visits, and the **loss** for
    class imbalance. And as the split exercise showed, an honest evaluation matters
    more than the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pen and paper: how should we split this dataset?

    The 48 records give **35610** beats in total. Suppose you evaluate the beat
    classifier with an **80/20 train/validation split**.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {"True": "true", "False": "false"}
    q_pp_ecg_split = mo.ui.radio(
        options=_opts,
        label="(A1) True or False: an 80/20 split puts exactly 0.8 x 35610 = "
        "28488 beats in train and the other 7122 in validation.",
        value=submission_radio_default("Q_PP_A_ECG_SPLIT", _opts),
    )
    q_pp_ecg_split
    return (q_pp_ecg_split,)


@app.cell
def _(mo):
    mo.md(r"""
    # Part 1 - B. Longitudinal data: leak-free landmark survival on the PBC cohort

    > **The setup.** The hepatology group hands you the **PBC** cohort
    > (primary biliary cirrhosis): 312 patients followed over years with repeated
    > liver-lab panels and follow-up (`futime` / `status`).
    >
    > *"From a patient's first 2 years of labs, can we flag who dies within 5
    > years? And is a Transformer worth it over a plain RNN here?"*

    The task is a **landmark prediction** that never leaks the future: features
    come only from visits in the first **2 years**, the label is death within
    **5 years**, patients must be alive at the 2-year landmark, and anyone whose
    5-year status is unknown is dropped (so no model ever sees post-window data).
    That leaves **257 patients, 55 deaths (~21%)**. You compare three models:

    1. a **logistic baseline** on summary statistics (mean, last, slope per lab),
    2. a **tiny RNN** (one small GRU layer),
    3. a **tiny Transformer** (one small encoder layer).

    Implement the functions in `longitudinal_exercise.py`, then run the cells below.
    """)
    return


@app.cell
def _(Path, np, plt):
    import pandas as pd

    PBC_BLUE, PBC_RED, PBC_GREEN, PBC_GREY = "#344A9A", "#C8323C", "#00A082", "#9AA0A6"
    plt.rcParams.update(
        {
            "font.size": 12,
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    _DATA = Path(__file__).with_name("data") / "pbcseq.csv"
    LABS = ["bili", "albumin", "alk.phos", "ast", "platelet", "protime"]
    LOGLABS = {"bili", "alk.phos", "ast", "protime"}
    PRETTY = {
        "bili": "bilirubin",
        "albumin": "albumin",
        "alk.phos": "alk. phos.",
        "ast": "AST",
        "platelet": "platelets",
        "protime": "prothr. time",
    }

    pbc_df = pd.read_csv(_DATA)
    pbc_df["yr"] = pbc_df["day"] / 365.25
    pbc_df["lb"] = np.log(pbc_df["bili"].clip(lower=1e-3))
    pbc_df["died"] = pbc_df.groupby("id")["status"].transform(lambda s: int((s == 2).any()))
    nvis = pbc_df.groupby("id").size()

    # Your graded lag-1 autocorrelation (why persistence is hard to beat).
    try:
        import longitudinal_solution as _ll
    except ModuleNotFoundError:
        import longitudinal_exercise as _ll
    lag1_corr = _ll.lag1_autocorr

    print(
        f"{pbc_df['id'].nunique()} patients, {len(pbc_df)} visits; "
        f"visits/patient median {int(nvis.median())} (range {nvis.min()}-{nvis.max()})"
    )
    return (
        LABS,
        PBC_BLUE,
        PBC_GREEN,
        PBC_GREY,
        PBC_RED,
        PRETTY,
        lag1_corr,
        nvis,
        pbc_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Meet the data: what is PBC?

    **Primary biliary cholangitis (PBC)** is a chronic **autoimmune** liver disease: the immune
    system attacks and slowly destroys the **small bile ducts** inside the liver. Bile can then
    no longer drain (**cholestasis**), so **bilirubin** and bile acids build up in the blood,
    causing **jaundice** and itching, and over years the scarring progresses to **cirrhosis**.

    That mechanism is why **serum bilirubin** climbs along a patient's trajectory and is the
    central marker tracked in the `pbcseq` cohort.
    """)
    return


@app.cell
def _(PBC_BLUE, PBC_GREY, Path, plt):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    _YEL = "#FFE863"
    figx, (axx, axy) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    axx.axis("off")
    axx.imshow(plt.imread(Path(__file__).with_name("pbc.png")))
    axx.text(
        0.0,
        1.0,
        " MEDICAL EXCURSUS ",
        transform=axx.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="black",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="square,pad=0.25", facecolor=_YEL, edgecolor="none"),
        zorder=8,
    )
    axy.set_xlim(0, 5)
    axy.set_ylim(0, 5)
    axy.axis("off")
    _steps = [
        ("Small bile ducts destroyed", PBC_GREY),
        ("Bile cannot drain (cholestasis)", PBC_GREY),
        ("Bilirubin builds up in the blood", PBC_BLUE),
        ("Jaundice & itch;\nscarring $\\rightarrow$ cirrhosis over years", PBC_GREY),
    ]
    _ys = [4.3, 3.25, 2.2, 1.0]
    _cx = 2.5
    for (_txt, _fc), _y in zip(_steps, _ys):
        _h = 0.62 if "\n" in _txt else 0.5
        axy.add_patch(
            FancyBboxPatch(
                (_cx - 2.1, _y - _h / 2),
                4.2,
                _h,
                boxstyle="round,pad=0.06",
                facecolor=_fc,
                edgecolor="none",
                alpha=0.92 if _fc != PBC_GREY else 0.8,
                zorder=3,
            )
        )
        axy.text(
            _cx,
            _y,
            _txt,
            ha="center",
            va="center",
            color="white",
            fontsize=9.5,
            fontweight=("bold" if _fc == PBC_BLUE else "normal"),
            zorder=4,
        )
    for _y0, _y1 in zip(_ys[:-1], _ys[1:]):
        axy.add_patch(
            FancyArrowPatch(
                (_cx, _y0 - 0.30),
                (_cx, _y1 + 0.30),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#555",
                lw=1.6,
                zorder=2,
            )
        )
    axy.text(
        _cx,
        0.32,
        "serum bilirubin is the marker tracked in pbcseq",
        ha="center",
        va="center",
        fontsize=8.8,
        color=PBC_BLUE,
        fontstyle="italic",
    )
    axy.set_title("Why bilirubin rises along the trajectory", fontsize=11.5)
    figx.tight_layout()
    figx
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The PBC cohort at a glance

    `pbcseq` is the Mayo Clinic PBC trial: **312 patients**, followed for over a decade, seen at
    **repeated visits** (every 6-12 months) where a panel of **liver labs** is measured, until
    **death**, **transplant**, or end of study (**censored**). Each line below is one patient,
    each dot a visit, the end marker their outcome: many visits per patient, **uneven** spacing
    and length, and shorter tracks ending in **death**.
    """)
    return


@app.cell
def _(PBC_BLUE, PBC_GREY, PBC_RED, pbc_df, plt):
    sw_order = pbc_df.groupby("id")["yr"].max().sort_values().index
    sw_ids = sw_order[:: max(1, len(sw_order) // 30)][:30]
    figc, axc = plt.subplots(figsize=(9.2, 5.2))
    for _row, _pid in enumerate(sw_ids):
        _g = pbc_df[pbc_df.id == _pid].sort_values("yr")
        _t = _g["yr"].values * 12.0
        axc.plot([0, _t.max()], [_row, _row], color=PBC_GREY, lw=0.8, alpha=0.5, zorder=1)
        axc.scatter(_t, [_row] * len(_t), s=16, color=PBC_BLUE, zorder=2)
        if (_g["status"] == 2).any():
            axc.scatter(_t.max(), _row, marker="X", s=55, color=PBC_RED, zorder=3)
        else:
            axc.scatter(
                _t.max(),
                _row,
                marker="o",
                s=34,
                facecolors="none",
                edgecolors=PBC_GREY,
                linewidths=1.3,
                zorder=3,
            )
    axc.set_yticks([])
    axc.set_xlabel("months since enrolment")
    axc.set_ylabel(f"patients (sample of {len(sw_ids)}, sorted by follow-up)")
    axc.set_title(
        f"The PBC cohort: {pbc_df['id'].nunique()} patients over ~{pbc_df['yr'].max() * 12:.0f} months"
    )
    axc.scatter([], [], s=16, color=PBC_BLUE, label="visit (labs measured)")
    axc.scatter([], [], marker="X", s=55, color=PBC_RED, label="death")
    axc.scatter(
        [],
        [],
        marker="o",
        s=34,
        facecolors="none",
        edgecolors=PBC_GREY,
        label="censored / transplant",
    )
    axc.legend(loc="lower right", fontsize=9, frameon=False)
    figc.tight_layout()
    figc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repeated measures

    Each patient contributes a *sequence* of visits (median 5, up to 16), and bilirubin is
    tracked along a personal trajectory. Patients who later die (red) tend to ride higher.
    """)
    return


@app.cell
def _(PBC_BLUE, PBC_RED, nvis, pbc_df, plt):
    fig1, (ov1, ov2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ov1.hist(nvis.values, bins=range(1, 18), color=PBC_BLUE, alpha=0.85, edgecolor="white")
    ov1.axvline(nvis.median(), color=PBC_RED, lw=2, ls="--")
    ov1.annotate(
        f"median {int(nvis.median())} visits",
        (nvis.median() + 0.4, ov1.get_ylim()[1] * 0.9),
        color=PBC_RED,
        fontsize=10,
    )
    ov1.set_xlabel("visits per patient")
    ov1.set_ylabel("patients")
    ov1.set_title(
        f"{pbc_df['id'].nunique()} patients, {len(pbc_df)} visits: repeated measures"
    )
    sample = pbc_df[pbc_df.groupby("id")["id"].transform("size") >= 4]
    for _pid, g in sample.groupby("id"):
        g = g.sort_values("yr")
        ov2.plot(
            g["yr"],
            g["lb"],
            "-",
            color=(PBC_RED if g["died"].iloc[0] else PBC_BLUE),
            lw=0.7,
            alpha=0.35,
        )
    ov2.set_xlabel("years since enrolment")
    ov2.set_ylabel("log serum bilirubin")
    ov2.set_title("One line per patient (red = later death)")
    ov2.set_xlim(0, 12)
    fig1.tight_layout()
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Within-patient correlation: the rows are not independent

    The defining property of longitudinal data. **Left:** the gap between two visits of the
    **same** patient (green) is tiny next to two **random** visits (grey). **Right:** the
    **autocorrelation**, the plain correlation between a visit and that patient's *next* visit;
    bilirubin is ~0.9. So treating pooled visits as independent rows, or using a random
    train/test split, leaks correlated information across the split.
    """)
    return


@app.cell
def _(LABS, PBC_BLUE, PBC_GREEN, PBC_GREY, PRETTY, lag1_corr, np, pbc_df, plt):
    rng = np.random.default_rng(0)
    sdf = pbc_df.dropna(subset=["lb"]).sort_values(["id", "day"])
    same = sdf.groupby("id")["lb"].diff().abs().dropna().values
    vals = sdf["lb"].values
    ids = sdf["id"].values
    perm = rng.permutation(len(vals))
    cross = np.abs(vals - vals[perm])[ids != ids[perm]]
    acf = {lab: lag1_corr(pbc_df, lab) for lab in LABS}
    fig2, (ic1, ic2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    bins = np.linspace(0, 4, 30)
    ic1.hist(
        same,
        bins=bins,
        density=True,
        color=PBC_GREEN,
        alpha=0.8,
        label=f"same patient (med {np.median(same):.2f})",
    )
    ic1.hist(
        cross,
        bins=bins,
        density=True,
        color=PBC_GREY,
        alpha=0.55,
        label=f"random pair (med {np.median(cross):.2f})",
    )
    ic1.set_xlabel(r"$|\Delta$ log-bilirubin$|$ between two visits")
    ic1.set_ylabel("density")
    ic1.set_title("Two visits of the SAME patient are far more alike")
    ic1.legend(fontsize=9, frameon=False)
    _order = sorted(acf, key=acf.get)
    ic2.barh(
        [PRETTY[lab] for lab in _order],
        [acf[lab] for lab in _order],
        color=PBC_BLUE,
        alpha=0.85,
    )
    ic2.set_xlim(0, 1)
    ic2.set_xlabel("correlation between a visit and the next (same patient)")
    ic2.set_title("Consecutive visits are strongly correlated")
    for _i, lab in enumerate(_order):
        ic2.annotate(
            f"{acf[lab]:.2f}",
            (acf[lab] + 0.02, _i),
            va="center",
            fontsize=9,
            color=PBC_BLUE,
        )
    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Irregular and informative sampling

    Longitudinal data rarely lands on a clean grid. **Left:** the gap $\Delta t$ between
    consecutive visits varies. **Right:** the *length* of a patient's series is itself
    informative: patients who die contribute **shorter** sequences (follow-up ends), so the
    sampling pattern is not independent of the outcome. This is the longitudinal face of
    "missingness": not values to impute, but a sampling pattern that carries signal.
    """)
    return


@app.cell
def _(PBC_BLUE, PBC_RED, np, nvis, pbc_df, plt):
    dts = pbc_df.sort_values(["id", "day"]).groupby("id")["yr"].diff().dropna().values
    died = pbc_df.groupby("id")["died"].first()
    fig3, (sp1, sp2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    sp1.hist(dts, bins=np.linspace(0, 3, 40), color=PBC_BLUE, alpha=0.85, edgecolor="white")
    sp1.set_xlabel(r"gap between consecutive visits, $\Delta t$ (years)")
    sp1.set_ylabel("visit pairs")
    sp1.set_title(f"Spacing varies (median {np.median(dts):.2f} yr, not a grid)")
    parts = [nvis[died == 0].values, nvis[died == 1].values]
    bp = sp2.boxplot(
        parts, tick_labels=["survived /\ncensored", "died"], patch_artist=True, widths=0.6
    )
    for patch, pcol in zip(bp["boxes"], [PBC_BLUE, PBC_RED]):
        patch.set_facecolor(pcol)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    sp2.set_ylabel("visits per patient")
    sp2.set_title("Series length is itself informative")
    fig3.tight_layout()
    fig3
    return


@app.cell
def _(Path):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Prefer the reference solution if present (instructor side); otherwise fall
    # back to the student stubs. You implement the TODOs in longitudinal_exercise.py.
    # reload so edits to the module are picked up without restarting the kernel.
    import importlib as _importlib

    try:
        import longitudinal_solution as pr
    except ModuleNotFoundError:
        import longitudinal_exercise as pr
    pr = _importlib.reload(pr)

    BLUE, RED, GREEN = "#344A9A", "#C8323C", "#00A082"
    DATA = Path(__file__).with_name("data") / "pbcseq.csv"
    X, mask, y = pr.load_sequences(str(DATA))
    print(
        f"patients={len(y)}  deaths={int(y.sum())} ({y.mean():.0%})  "
        f"feature dim per visit={X.shape[2]}  max visits={X.shape[1]}"
    )
    return (
        BLUE,
        GREEN,
        LogisticRegression,
        RED,
        StandardScaler,
        StratifiedKFold,
        X,
        mask,
        pr,
        roc_auc_score,
        y,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### The logistic regression baseline

    Collapse each patient to summary statistics (mean / last / slope per lab)
    and fit logistic regression, scored by 5-fold cross-validated AUROC. Any
    sequence model has to clear this bar.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    StratifiedKFold,
    X,
    mask,
    np,
    pr,
    roc_auc_score,
    y,
):
    F = pr.summary_features(X, mask)
    _auc = []
    for _tr, _te in StratifiedKFold(5, shuffle=True, random_state=42).split(F, y):
        _sc = StandardScaler().fit(F[_tr])
        _lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            _sc.transform(F[_tr]), y[_tr]
        )
        _auc.append(roc_auc_score(y[_te], _lr.predict_proba(_sc.transform(F[_te]))[:, 1]))
    auc_base = float(np.mean(_auc))
    print(f"logistic baseline: AUROC {auc_base:.3f} +/- {np.std(_auc):.3f}")
    return (auc_base,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The two tiny sequence models

    Both are deliberately tiny (a few hundred parameters). Note the parameter
    counts: that is the whole budget the model gets to learn temporal shape from
    ~250 training patients per fold.
    """)
    return


@app.cell
def _(X, mask, pr, y):
    print(f"RNN params:         {pr.count_parameters(pr.make_rnn(X.shape[2]))}")
    print(f"Transformer params: {pr.count_parameters(pr.make_transformer(X.shape[2]))}")


    auc_rnn, sd_rnn = pr.cross_val_auroc(lambda: pr.make_rnn(X.shape[2]), X, mask, y)
    auc_tfm, sd_tfm = pr.cross_val_auroc(
        lambda: pr.make_transformer(X.shape[2]), X, mask, y
    )
    print(f"RNN:         AUROC {auc_rnn:.3f} +/- {sd_rnn:.3f}")
    print(f"Transformer: AUROC {auc_tfm:.3f} +/- {sd_tfm:.3f}")
    return auc_rnn, auc_tfm


@app.cell
def _(BLUE, GREEN, RED, auc_base, auc_rnn, auc_tfm, plt):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _names = ["logistic\n(summary stats)", "tiny RNN", "tiny Transformer"]
    _vals = [auc_base, auc_rnn, auc_tfm]
    _bars = ax.bar(_names, _vals, color=[GREEN, BLUE, RED], alpha=0.85)
    ax.axhline(0.5, color="k", lw=0.9, ls="--", alpha=0.4)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_title("PBC death prediction: do the sequence models earn their keep?")
    for _b, _v in zip(_bars, _vals):
        ax.annotate(
            f"{_v:.3f}",
            (_b.get_x() + _b.get_width() / 2, _v + 0.005),
            ha="center",
            fontsize=12,
        )
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1 - B. Continued: forecasting the next visit's labs

    A regression flavour: instead of an outcome, predict the **next visit's lab
    vector** from the history (`load_forecasting`). The baseline is **persistence**
    (`persistence_forecast`): the next labs equal the last observed ones. Because the
    labs are slowly varying and highly autocorrelated, persistence is very hard to
    beat, and a tiny neural forecaster usually does not (it regresses toward the
    mean). Scored with MAE in standardized units.

    Each patient contributes many (history, next-visit) examples, so they are not
    independent: cross-validate with **`GroupKFold` keyed on the patient id**, or a
    patient leaks across train and test (the within-patient correlation issue from
    the lecture, made operational).
    """)
    return


@app.cell
def _(Path, np, pr):
    import torch as _torch
    import torch.nn as _nn

    Xf, maskf, Yf, gf = pr.load_forecasting(
        str(Path(__file__).with_name("data") / "pbcseq.csv")
    )
    print(f"forecast examples={len(Yf)}  patients={len(np.unique(gf))}")
    print(
        f"persistence (next = last)  MAE: "
        f"{pr.forecast_mae(Yf, pr.persistence_forecast(Xf, maskf)):.3f}"
    )


    # pre-given tiny forecasters: each outputs the next lab vector (dim = N_FEATURES)
    class _RNNForecast(_nn.Module):
        def __init__(self, d, h=8):
            super().__init__()
            self.g = _nn.GRU(d, h, batch_first=True)
            self.fc = _nn.Linear(h, d)

        def forward(self, x, m):
            o, _ = self.g(x)
            i = m.sum(1).clamp(min=1).long() - 1
            return self.fc(o[_torch.arange(len(x)), i])


    class _TFMForecast(_nn.Module):
        def __init__(self, d, dm=8, nh=2, ff=16):
            super().__init__()
            self.e = _nn.Linear(d, dm)
            layer = _nn.TransformerEncoderLayer(dm, nh, ff, batch_first=True)
            self.enc = _nn.TransformerEncoder(layer, 1)
            self.fc = _nn.Linear(dm, d)

        def forward(self, x, m):
            h = self.enc(self.e(x), src_key_padding_mask=~m)
            h = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            return self.fc(h)


    print(
        f"tiny RNN forecaster         MAE: "
        f"{pr.groupkfold_mae(lambda: _RNNForecast(Xf.shape[2]), Xf, maskf, Yf, gf):.3f}"
    )
    print(
        f"tiny Transformer forecaster MAE: "
        f"{pr.groupkfold_mae(lambda: _TFMForecast(Xf.shape[2]), Xf, maskf, Yf, gf):.3f}"
    )
    print("(persistence is the one to beat)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### pen-and-paper questions

    Answer the three questions below from what you saw. They auto-save to
    `submission.json` (read by the autograder).
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "It runs faster than using the whole record.": "a",
        '"Ever died" leaks the future: the last visit sits right before death '
        "and follow-up length encodes the outcome, inflating every model.": "b",
        "The first two years already contain all of each patient's visits.": "c",
    }
    q_pp_a_leak = mo.ui.radio(
        options=_opts,
        label="(B1) Why describe a patient by their first 2 years and predict "
        "death within 5 years, rather than whether they *ever* died?",
        value=submission_radio_default("Q_PP_B_LEAK", _opts),
    )
    q_pp_a_leak
    return (q_pp_a_leak,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "They are close (AUROC ~0.88-0.92); with only ~55 events the "
        "differences are within noise.": "a",
        "The logistic baseline dominates; the sequence models trail far behind.": "b",
        "The Transformer wins by a wide, clearly significant margin.": "c",
    }
    q_pp_a_compare = mo.ui.radio(
        options=_opts,
        label="(B2) On the leak-free landmark task, how do the three models compare?",
        value=submission_radio_default("Q_PP_B_COMPARE", _opts),
    )
    q_pp_a_compare
    return (q_pp_a_compare,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "AUROC collapses to chance: the model is now too small to learn.": "a",
        "AUROC stays about the same: with only a few hundred short sequences "
        "the extra capacity is not the bottleneck.": "b",
        "AUROC improves a lot: smaller is always better.": "c",
    }
    q_pp_a_size = mo.ui.radio(
        options=_opts,
        label="(B3) You shrink the RNN from hidden=16 to hidden=4. What happens?",
        value=submission_radio_default("Q_PP_B_SIZE", _opts),
    )
    q_pp_a_size
    return (q_pp_a_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Attention puzzle: diagnose four patients' models

    Imagine a **single-head** Transformer. Below are the attention matrices it
    produced for **four patients** (rows = query visit, columns = the visit
    attended to; each row sums to 1). Each visit's representation is the
    **weighted average** of all visits, using its row of weights.

    The four patients had very different clinical courses, and the attention map
    reflects what the model leaned on for each. Below are four short clinical
    situations. Read each matrix, work out the pattern it implies, and match it to
    the patient. Two patterns connect back to models you already built: the
    logistic **mean-summary** baseline and **persistence**.
    """)
    return


@app.cell(hide_code=True)
def _(np, plt):
    # Four synthetic single-head attention matrices (rows = query, cols = key;
    # each row sums to 1). Archetypes are scrambled across the patient numbers.
    _mats = {
        1: np.array(
            [
                [0.10, 0.10, 0.15, 0.65],  # recency: every row -> last visit
                [0.12, 0.08, 0.15, 0.65],
                [0.08, 0.10, 0.17, 0.65],
                [0.05, 0.08, 0.12, 0.75],
            ]
        ),
        2: np.array(
            [
                [0.79, 0.07, 0.07, 0.07],  # self/diagonal: no context mixing
                [0.07, 0.79, 0.07, 0.07],
                [0.07, 0.07, 0.79, 0.07],
                [0.07, 0.07, 0.07, 0.79],
            ]
        ),
        3: np.array(
            [
                [0.70, 0.10, 0.10, 0.10],  # baseline: every row -> first visit
                [0.68, 0.12, 0.10, 0.10],
                [0.71, 0.09, 0.12, 0.08],
                [0.72, 0.10, 0.10, 0.08],
            ]
        ),
        4: np.array(
            [
                [0.24, 0.26, 0.25, 0.25],  # uniform: averaging
                [0.26, 0.24, 0.25, 0.25],
                [0.25, 0.25, 0.26, 0.24],
                [0.25, 0.25, 0.24, 0.26],
            ]
        ),
    }
    _fig, _axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    _lbl = ["v1", "v2", "v3", "v4"]
    for _ax, _pid in zip(_axes, _mats):
        _A = _mats[_pid]
        _ax.imshow(_A, cmap="Oranges", vmin=0, vmax=1)
        _ax.set_xticks(range(4))
        _ax.set_yticks(range(4))
        _ax.set_xticklabels(_lbl, fontsize=8)
        _ax.set_yticklabels(_lbl, fontsize=8)
        _ax.set_title(f"Patient {_pid}", fontsize=11)
        for _r in range(4):
            for _c in range(4):
                _ax.text(
                    _c,
                    _r,
                    f"{_A[_r, _c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if _A[_r, _c] > 0.5 else "black",
                )
    _axes[0].set_ylabel("query visit", fontsize=8)
    _fig.supxlabel("key (visit attended to)", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {"Patient 1": "1", "Patient 2": "2", "Patient 3": "3", "Patient 4": "4"}
    q_pp_attn_avg = mo.ui.radio(
        options=_opts,
        label="(B4) A patient whose labs stay stable and unremarkable across "
        "every visit, with no single visit standing out. Which patient?",
        value=submission_radio_default("Q_PP_B_ATTN_AVG", _opts),
    )
    q_pp_attn_avg
    return (q_pp_attn_avg,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {"Patient 1": "1", "Patient 2": "2", "Patient 3": "3", "Patient 4": "4"}
    q_pp_attn_recency = mo.ui.radio(
        options=_opts,
        label="(B5) A patient who was stable for years, then deteriorated "
        "abruptly over the final visits. Whose attention map fits?",
        value=submission_radio_default("Q_PP_B_ATTN_RECENCY", _opts),
    )
    q_pp_attn_recency
    return (q_pp_attn_recency,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {"Patient 1": "1", "Patient 2": "2", "Patient 3": "3", "Patient 4": "4"}
    q_pp_attn_self = mo.ui.radio(
        options=_opts,
        label="(B6) A patient seen for unrelated one-off problems each time, "
        "with no connecting trajectory between visits. Which patient?",
        value=submission_radio_default("Q_PP_B_ATTN_SELF", _opts),
    )
    q_pp_attn_self
    return (q_pp_attn_self,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {"Patient 1": "1", "Patient 2": "2", "Patient 3": "3", "Patient 4": "4"}
    q_pp_attn_baseline = mo.ui.radio(
        options=_opts,
        label="(B7) A patient with a dramatic abnormality at the very first "
        "visit, a very high baseline bilirubin, that never recurred but "
        "still defines their prognosis. Which patient?",
        value=submission_radio_default("Q_PP_B_ATTN_BASELINE", _opts),
    )
    q_pp_attn_baseline
    return (q_pp_attn_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Spot the trustworthy result

    Four teams enter the PBC challenge. Every team does the honest part right:
    features from each patient's **first 2 years**, label = **death within 5
    years**, and folds **split by patient**. All four proudly report
    **AUROC ≈ 0.95**. But the reviewers know **three of them leaked** somewhere
    else in the pipeline. You are the reviewer: whose number would you actually
    trust?

    Reminder: leakage is any way that test-set information sneaks into training
    or preprocessing. Nothing else counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "Team A standardizes each lab using a separate, older cohort's mean and SD.": "a",
        "Team B drops patients whose labs are most extreme across the whole dataset.": "b",
        "Team C reports the best AUROC out of 20 random patient-level splits.": "c",
        "Team D adds each patient's total number of visits as a feature.": "d",
    }
    q_pp_leakfree = mo.ui.radio(
        options=_opts,
        label="(B8) Whose reported AUROC would you actually trust?",
        value=submission_radio_default("Q_PP_B_LEAKFREE", _opts),
    )
    q_pp_leakfree
    return (q_pp_leakfree,)


@app.cell
def _(mo):
    mo.md(r"""
    # Part 2: Tracking (from frames to trajectories)

    > **The setup.** Your lab images a dish of migrating cells under a
    > microscope. Every few minutes the segmentation pipeline spits out
    > a *frame*: a handful of $(x, y)$ detections, one per cell. The
    > catch: **the detector does not know which cell is which**. Frame
    > to frame, the dots are unlabelled and arrive in arbitrary order.
    >
    > A biologist wants *trajectories*: which dot in frame $t{+}1$ is the
    > same cell as which dot in frame $t$, chained all the way through
    > the movie. That linking step is **data association**, and it is the
    > whole job of this section.

    Your plan:

    - **A.** Treat one frame-to-frame step as an **assignment problem**:
      build a cost matrix of distances, then solve it two ways —
      **greedy nearest-neighbour** and the **Hungarian algorithm**. Do they
      agree on a clean step?
    - **B.** A **dish bump**: between two frames the microscope stage gets
      knocked and *every* detection jumps. How do greedy and Hungarian cope
      when the whole geometry shifts?
    - **C.** **Chain** one-step matches into full tracks and count
      **identity switches** for each matcher.
    - **D.** Identities can also break at a **crossing**. Add a
      constant-velocity motion model — **predict, then match** — and see
      which failures it fixes and which it doesn't.

    The cohort: **3 cells over 8 frames** in `data/detections.csv`. Two of
    them cross paths, and the dish gets bumped early on.
    """)
    return


@app.cell
def _(Path):
    # Prefer the reference solution if present (instructor side); otherwise fall
    # back to the student stubs. You implement the TODOs in tracking_exercise.py.
    # reload so edits to the module are picked up without restarting the kernel.
    import importlib as _importlib

    try:
        import tracking_solution as trk
    except ModuleNotFoundError:
        import tracking_exercise as trk
    trk = _importlib.reload(trk)

    DATA_PATH = Path(__file__).with_name("data") / "detections.csv"
    return DATA_PATH, trk


@app.cell
def _(DATA_PATH, mo, trk):
    df = trk.load_detections(DATA_PATH)
    pos_list, gt_list = trk.frames_as_list(df)
    mo.md(
        f"Loaded **{df['det_id'].max() + 1} detections × "
        f"{df['frame'].max() + 1} frames**. Columns: "
        f"`{', '.join(df.columns)}`. The `track_id` column is the "
        f"ground truth: used only to *score* a tracker, never to match."
    )
    return df, gt_list, pos_list


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What the detector sees vs. what is really there

    Left: the raw detections, coloured by **frame**. This is all a
    tracker gets. Right: the same points coloured by the **hidden
    ground-truth identity**, with the true trajectories drawn in. Notice
    two tracks heading straight for each other (a **crossing**), and the
    sharp **vertical jump** of every track at frame 2: the dish bump.
    """)
    return


@app.cell
def _(df, plt, pos_list):
    _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(9, 3.6))

    _n_frames = len(pos_list)
    for _t, _p in enumerate(pos_list):
        _axL.scatter(_p[:, 0], _p[:, 1], s=30, color=plt.cm.viridis(_t / (_n_frames - 1)))
    _axL.set_title("Detections coloured by frame\n(what the tracker sees)")
    _axL.set_xlabel("x")
    _axL.set_ylabel("y")

    for _tid, _g in df.groupby("track_id"):
        _g = _g.sort_values("frame")
        _axR.plot(
            _g["x"],
            _g["y"],
            "-o",
            ms=4,
            color=plt.cm.tab10(_tid % 10),
            label=f"track {_tid}",
        )
    _axR.set_title("Ground-truth trajectories\n(the hidden answer)")
    _axR.set_xlabel("x")
    _axR.set_ylabel("y")
    _axR.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gt_list, np, pos_list):
    # READING AID (scoring only): order each frame's detections by their
    # true identity, so row i and column i are the same cell and the
    # *correct* assignment is the diagonal {0:0, 1:1, 2:2}. A real tracker
    # never gets to do this: it only sees the scrambled det_id order.
    def aligned(t):
        order = np.argsort(gt_list[t])
        return pos_list[t][order]

    return (aligned,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2-A. One frame-to-frame step is an assignment problem

    Take frame 0 and frame 1. Each has the same three detections. To link
    them we score every (frame-0 dot, frame-1 dot) pair by how far apart
    they are (that is your **cost matrix** `cost_matrix(src, dst)`) and
    then pick a one-to-one matching.

    Two ways to pick:

    - **`greedy_nn`**: walk the rows top to bottom; each track grabs its
      nearest *still-free* detection. Fast, intuitive, order-dependent.
    - **`hungarian`**: minimise the *total* cost over all one-to-one
      matchings (via `scipy.optimize.linear_sum_assignment`), with a
      guarantee.

    Below we use the **real** frames 0 and 1 (ordered by identity so the
    correct answer is the diagonal). On this clean step, do the two
    methods pick the same matching?
    """)
    return


@app.cell
def _(aligned, trk):
    clean_cost = trk.cost_matrix(aligned(0), aligned(1))
    clean_greedy = trk.greedy_nn(clean_cost)
    clean_hung = trk.hungarian(clean_cost)
    return clean_cost, clean_greedy, clean_hung


@app.cell
def _(clean_cost, clean_greedy, clean_hung, mo, np):
    mo.md(
        f"**Clean step (frames 0 → 1)**: cost matrix (rounded):\n\n"
        f"`{np.round(clean_cost, 2).tolist()}`\n\n"
        f"- greedy : assignment `{clean_greedy[0]}`, total **{clean_greedy[1]:.2f}**, "
        f"collided = `{clean_greedy[2]}`\n"
        f"- hungarian: assignment `{clean_hung[0]}`, total **{clean_hung[1]:.2f}**\n\n"
        f"Both pick the diagonal `{{0:0, 1:1, 2:2}}`: when the nearest "
        f"neighbours don't conflict, greedy is already optimal."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2-B. The dish gets bumped

    Between frame 1 and frame 2 someone knocks the microscope stage, so
    **every detection jumps** by the same offset, and stays shifted from
    then on. The jump is large enough that, looking only at raw distance,
    one cell's new detection lands closer to a *different* cell's old spot.

    Watch what greedy does on the **real** frames 1 → 2, processing rows
    top to bottom: will it still grab the right partner? And can a solver
    that optimises the *total* cost globally — the **Hungarian**
    algorithm — do any better when the whole dish has shifted rigidly?
    """)
    return


@app.cell
def _(aligned, trk):
    drift_cost = trk.cost_matrix(aligned(1), aligned(2))
    drift_greedy = trk.greedy_nn(drift_cost)
    drift_hung = trk.hungarian(drift_cost)
    return drift_cost, drift_greedy, drift_hung


@app.cell
def _(drift_cost, drift_greedy, drift_hung, mo, np):
    mo.md(
        f"**Drift step (frames 1 → 2)**: cost matrix (rounded):\n\n"
        f"`{np.round(drift_cost, 2).tolist()}`\n\n"
        f"| method | assignment | total cost |\n"
        f"|---|---|---|\n"
        f"| greedy | `{drift_greedy[0]}` | **{drift_greedy[1]:.2f}** |\n"
        f"| hungarian | `{drift_hung[0]}` | **{drift_hung[1]:.2f}** |\n\n"
        f"The correct answer is the diagonal `{{0:0, 1:1, 2:2}}`. Greedy "
        f"returns `{drift_greedy[0]}` (collided = `{drift_greedy[2]}`): "
        f"wrong, and pays **{drift_greedy[1] - drift_hung[1]:.2f}** extra "
        f"cost over Hungarian, which recovers the diagonal. *Same data, "
        f"same cost function: a better solver is what saved us here.*"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2-C. Chain one-step matches into tracks

    A trajectory is just one-step matches stitched together: seed labels
    from frame 0, then for every later frame match the previous detections
    to the current ones and carry each label forward. `link_tracks(pos,
    matcher)` does exactly that: pass it either matcher.

    We run it with **greedy** and with **Hungarian**, plot the recovered
    trajectories, and score them with `count_id_switches` against the
    ground truth. An **identity switch** is a track whose label flips from
    one frame to the next; **0** means every identity was held throughout.
    """)
    return


@app.cell
def _(plt):
    def _gather(pos_seq, key_seq):
        """Group detection positions over time by a per-detection key (label/gt)."""
        traj = {}
        for _t, (_p, _k) in enumerate(zip(pos_seq, key_seq)):
            for _i in range(len(_p)):
                traj.setdefault(int(_k[_i]), []).append((_t, _p[_i, 0], _p[_i, 1]))
        for _v in traj.values():
            _v.sort()
        return traj


    def plot_tracks(pos_seq, labels_seq, title, ax, gt_seq=None):
        """Recovered trajectories (coloured by predicted label), with the
        ground truth drawn underneath as faint grey dashed paths so an ID
        switch shows up as a coloured line hopping between true tracks."""
        if gt_seq is not None:
            gtraj = _gather(pos_seq, gt_seq)
            for _j, (_gid, _pts) in enumerate(sorted(gtraj.items())):
                ax.plot(
                    [q[1] for q in _pts],
                    [q[2] for q in _pts],
                    "--",
                    color="0.6",
                    lw=3.5,
                    alpha=0.7,
                    zorder=1,
                    label="ground truth" if _j == 0 else None,
                )
        traj = _gather(pos_seq, labels_seq)
        for _lid, _pts in sorted(traj.items()):
            ax.plot(
                [q[1] for q in _pts],
                [q[2] for q in _pts],
                "-o",
                ms=4,
                color=plt.cm.tab10(_lid % 10),
                zorder=2,
                label=f"id {_lid}",
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)

    return (plot_tracks,)


@app.cell
def _(gt_list, plot_tracks, plt, pos_list, trk):
    greedy_labels = trk.link_tracks(pos_list, lambda C: trk.greedy_nn(C)[0])
    hung_labels = trk.link_tracks(pos_list, lambda C: trk.hungarian(C)[0])

    sw_greedy = trk.count_id_switches(greedy_labels, gt_list)
    sw_hung = trk.count_id_switches(hung_labels, gt_list)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    plot_tracks(
        pos_list,
        greedy_labels,
        f"greedy linking: {sw_greedy} ID switches",
        _a1,
        gt_seq=gt_list,
    )
    plot_tracks(
        pos_list,
        hung_labels,
        f"Hungarian linking: {sw_hung} ID switches",
        _a2,
        gt_seq=gt_list,
    )
    _fig.tight_layout()
    _fig
    return sw_greedy, sw_hung


@app.cell
def _(mo, sw_greedy, sw_hung):
    mo.md(
        f"Greedy: **{sw_greedy}** ID switches. Hungarian: **{sw_hung}**. "
        f"Greedy trips **twice**: once at the dish bump, once at the "
        f"crossing. Hungarian fixes the bump (global optimisation), so it "
        f"trips only at the **crossing**. Those remaining switches are a "
        f"different kind of failure: there, distance itself is the wrong "
        f"signal: even the optimal matching swaps identities, because once "
        f"two cells have crossed, the swap *is* the cheapest assignment. "
        f"That is what part D fixes."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2-D. Predict, then match

    At a crossing the two cells are momentarily nearest the *wrong*
    neighbour, so any position-only cost mislinks them. A natural idea:
    **don't match on where a cell is, match on where it is going.** Each
    track carries a velocity; predict its next position
    (`pred = cur + (cur − prev)`) and match the *predictions* to the new
    detections. This is the core idea behind SORT-style trackers. Does it
    carry identities through the crossing — and does it survive the bump?

    `predict_then_match` does one such step; `link_tracks_motion` chains
    it (bootstrapping the first step with a plain Hungarian position match).
    """)
    return


@app.cell
def _(aligned, mo, trk):
    # The crossing step (frames 3 → 4), ordered by identity so the diagonal
    # is correct. Distance-only (Hungarian) swaps; prediction holds.
    cross_dist = trk.hungarian(trk.cost_matrix(aligned(3), aligned(4)))[0]
    cross_pred = trk.predict_then_match(aligned(2), aligned(3), aligned(4))
    mo.md(
        f"**Crossing step (frames 3 → 4).** Distance-only Hungarian: "
        f"`{cross_dist}`: a swap (should be the diagonal). "
        f"Predict-then-match: `{cross_pred}`: identities held. The "
        f"velocity carried each cell *through* the crossing."
    )
    return


@app.cell
def _(gt_list, plot_tracks, plt, pos_list, trk):
    motion_labels = trk.link_tracks_motion(pos_list)
    sw_motion = trk.count_id_switches(motion_labels, gt_list)

    _fig, _ax = plt.subplots(figsize=(5, 3.8))
    plot_tracks(
        pos_list,
        motion_labels,
        f"predict-then-match: {sw_motion} ID switches",
        _ax,
        gt_seq=gt_list,
    )
    _fig.tight_layout()
    _fig
    return (sw_motion,)


@app.cell
def _(mo, sw_greedy, sw_hung, sw_motion):
    mo.md(
        f"""**No single cost wins everywhere.** Tally over the whole movie:

    | linker | ID switches | fails at |
    |---|---|---|
    | greedy | **{sw_greedy}** | dish bump *and* crossing |
    | Hungarian (position) | **{sw_hung}** | crossing only |
    | predict-then-match | **{sw_motion}** | dish bump only |

    Predict-then-match sails through the **crossing**, but it *also* trips,
    at the **dish bump**: a constant-velocity model assumes smooth motion and
    cannot anticipate a sudden jump, so its prediction lands every cell off by
    the same offset and the match breaks (just like greedy).
    This is an inherent limitation of a simple motion model.

    So the two failure modes need *different* fixes: global assignment
    (Hungarian) rescues the discontinuity, motion prediction rescues the
    crossing. Real trackers (e.g. SORT) combine **both**: a motion model to
    build the cost *and* a global solver to assign it."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### pen-and-paper: run the Hungarian algorithm by hand

    Work this **3×3** cost matrix (tracks = rows, detections = columns)
    with pen and paper.

    $$
    C \;=\;
    \begin{bmatrix}
    3 & 7 & 4 \\
    1 & 7 & 7 \\
    4 & 3 & 2
    \end{bmatrix}
    $$

    Rows are tracks **1, 2, 3**; columns are detections **1, 2, 3**.

    Recall the steps from the lecture: **(1)** subtract each row's
    minimum, **(2)** subtract each column's minimum, **(3)** cover all
    zeros with the fewest lines, **(4)** if fewer than 3 lines, subtract
    the smallest uncovered value from all uncovered entries and add it to
    every doubly-covered entry, then re-cover, **(5)** read off the
    zero-cost assignment.

    Your answers save to `submission.json` via the last cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_greedy_total = mo.ui.number(
        start=0,
        stop=100,
        step=1,
        label="(C1) Greedy NN (rows top→bottom): what TOTAL cost does it pick?",
        value=submission_default("Q_PP_C_GREEDY_TOTAL"),
    )
    q_pp_greedy_total
    return (q_pp_greedy_total,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_opt_total = mo.ui.number(
        start=0,
        stop=100,
        step=1,
        label="(C2) Optimal (Hungarian) TOTAL cost:",
        value=submission_default("Q_PP_C_OPT_TOTAL"),
    )
    q_pp_opt_total
    return (q_pp_opt_total,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _q_strand_opts = {
        "Track 1": "t1",
        "Track 2": "t2",
        "Track 3": "t3",
    }
    q_pp_strand = mo.ui.radio(
        options=_q_strand_opts,
        label=(
            "(C3) Which track does greedy strand: forced off its own "
            "nearest detection onto an expensive one?"
        ),
        value=submission_radio_default("Q_PP_C_STRAND", _q_strand_opts),
    )
    q_pp_strand
    return (q_pp_strand,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_lines = mo.ui.number(
        start=1,
        stop=3,
        step=1,
        label=(
            "(C4) After the row- and column-reductions, what is the "
            "MINIMUM number of lines that cover all zeros?"
        ),
        value=submission_default("Q_PP_C_LINES"),
    )
    q_pp_lines
    return (q_pp_lines,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_adjust = mo.ui.number(
        start=0,
        stop=20,
        step=1,
        label=(
            "(C5) In the adjust step, what is the smallest UNCOVERED "
            "value you subtract / add?"
        ),
        value=submission_default("Q_PP_C_ADJUST"),
    )
    q_pp_adjust
    return (q_pp_adjust,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _q_assign_opts = {
        "track1→det3, track2→det1, track3→det2": "a",
        "track1→det1, track2→det2, track3→det3": "b",
        "track1→det2, track2→det1, track3→det3": "c",
        "track1→det3, track2→det2, track3→det1": "d",
    }
    q_pp_assign = mo.ui.radio(
        options=_q_assign_opts,
        label="(C6) What is the optimal (minimum-cost) one-to-one assignment?",
        value=submission_radio_default("Q_PP_C_ASSIGN", _q_assign_opts),
    )
    q_pp_assign
    return (q_pp_assign,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _q_cross_opts = {
        "After they cross, each cell sits closer to the other's detection, "
        "so the cheapest matching genuinely is the swap.": "a",
        "Their velocities are nearly equal there, so the cost rows tie and "
        "the solver resolves the match arbitrarily.": "b",
        "Detection noise is largest where the cells overlap, swamping the "
        "true inter-frame distances.": "c",
        "The within-frame shuffling of det_ids hides the order the solver "
        "needs to keep identities straight.": "d",
    }
    q_pp_cross = mo.ui.radio(
        options=_q_cross_opts,
        label=(
            "(C7) Conceptual: at the crossing, both greedy and Hungarian swap "
            "the two identities. What is the underlying reason?"
        ),
        value=submission_radio_default("Q_PP_C_CROSS", _q_cross_opts),
    )
    q_pp_cross
    return (q_pp_cross,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One last exercise: grade *us*

    Please fill out the **course teaching-evaluation form** on ILIAS. It is the
    one task where *you* hold the pen and *we* get the score, so be honest about
    what worked and what did not. It genuinely shapes next year's course. Tick
    the box below once you have submitted it.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {"Yes, I did": "yes"}
    q_pp_eval = mo.ui.radio(
        options=_opts,
        label="Have you filled out the course teaching-evaluation form on ILIAS?",
        value=submission_radio_default("Q_PP_EVAL", _opts),
    )
    q_pp_eval
    return (q_pp_eval,)


@app.cell
def _(
    mo,
    q_pp_a_compare,
    q_pp_a_leak,
    q_pp_a_size,
    q_pp_adjust,
    q_pp_assign,
    q_pp_attn_avg,
    q_pp_attn_baseline,
    q_pp_attn_recency,
    q_pp_attn_self,
    q_pp_cross,
    q_pp_ecg_split,
    q_pp_eval,
    q_pp_greedy_total,
    q_pp_leakfree,
    q_pp_lines,
    q_pp_opt_total,
    q_pp_strand,
):
    # Collect BOTH sections' pen-and-paper answers into a single submission.json
    # for the autograder.
    import json as _json
    from pathlib import Path as _Path

    _submission = {
        # Part 1 - A -- ECG (data splitting)
        "Q_PP_A_ECG_SPLIT": q_pp_ecg_split.value,
        # Part 1 - B -- sequence models
        "Q_PP_B_LEAK": q_pp_a_leak.value,
        "Q_PP_B_COMPARE": q_pp_a_compare.value,
        "Q_PP_B_SIZE": q_pp_a_size.value,
        "Q_PP_B_ATTN_AVG": q_pp_attn_avg.value,
        "Q_PP_B_ATTN_RECENCY": q_pp_attn_recency.value,
        "Q_PP_B_ATTN_SELF": q_pp_attn_self.value,
        "Q_PP_B_ATTN_BASELINE": q_pp_attn_baseline.value,
        "Q_PP_B_LEAKFREE": q_pp_leakfree.value,
        # Part 2 -- tracking
        "Q_PP_C_GREEDY_TOTAL": q_pp_greedy_total.value,
        "Q_PP_C_OPT_TOTAL": q_pp_opt_total.value,
        "Q_PP_C_STRAND": q_pp_strand.value,
        "Q_PP_C_LINES": q_pp_lines.value,
        "Q_PP_C_ADJUST": q_pp_adjust.value,
        "Q_PP_C_ASSIGN": q_pp_assign.value,
        "Q_PP_C_CROSS": q_pp_cross.value,
        # Course feedback
        "Q_PP_EVAL": q_pp_eval.value,
    }
    _path = _Path(__file__).with_name("submission.json")
    _path.write_text(_json.dumps(_submission, indent=2))
    _n_unanswered = sum(1 for v in _submission.values() if v is None)
    mo.md(
        f"**Answers auto-saved to `{_path.name}`** "
        f"({len(_submission)} questions, {_n_unanswered} unanswered)."
    )
    return


if __name__ == "__main__":
    app.run()
