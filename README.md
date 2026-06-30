# Assignment 04: Time-series, Predictive & Longitudinal Modeling

**ML4Health 2026**

This assignment has two parts on one theme, reading signal out of data that
changes over time.

**Part 1: Clinical modeling** *(graded)*

- **A. Time-series data (ECG, MIT-BIH)**: a Normal-vs-PVC heartbeat classifier
  built end to end, a 1D convolution and a dilated TCN on raw heartbeats where the
  waveform *shape* is the whole signal, with a flat summary-stat baseline and a
  leak-free train/test split (`split_by_record`).
- **B. Longitudinal data (PBC)**: leak-free landmark risk prediction from repeated
  liver-lab visits, using a tiny RNN vs Transformer vs a logistic baseline, with
  leakage-aware evaluation (ROC / AUC).

**Part 2: Tracking moving cells (video)** *(graded)*

Multi-object **tracking**: chaining unlabelled per-frame detections into
trajectories (data association).

Everything runs from `notebook.py` (a marimo walkthrough); all graded code is
checked by `tests/test_assignment.py`.

## Set up your environment

Reuse the shared `ml4health` conda env (the same one from assignments 01-03).
Create it once if you do not have it yet, and **activate it before installing
anything**, so packages land in the env and not in your base environment:

```bash
conda create -n ml4health python=3.11   # once per machine; skip if it exists
conda activate ml4health                # in every new shell
pip install -r requirements.txt
```

> Then run the tests for **both** graded sections with:
>
> ```bash
> conda activate ml4health
> pytest -v tests/
> ```
>
> Part 1 needs `torch` (in `requirements.txt`); CPU is fine, since the models
> are tiny and 5-fold cross-validation runs in seconds.

---

## Part 1 - A: a 1D CNN on ECG (time-series)

Data: `data/mitbih_ch0.npz` (channel 0 of the MIT-BIH Arrhythmia Database).
You build a Normal-vs-PVC heartbeat classifier end to end: extract single beats
from the raw ECG and classify them with a small **1D convolution** (and a dilated
**TCN**, the WaveNet/GluNet family from the lecture), where the waveform *shape*
is the whole signal. It shows why a generic summary-statistics baseline is hard to
beat for the wrong reasons, and that under leak-free evaluation a bigger receptive
field helps only a little.

You implement **seven graded functions** in `time_series_exercise.py`:
`extract_beats` (cut z-scored windows around the R-peaks), `ecg_summary_features`
(the flat mean/std/min/max/last baseline that deliberately throws away waveform
shape), `make_beat_cnn` and `make_tcn` (the shallow CNN and the dilated TCN),
`train_cnn` (the training loop), `receptive_field` (how far a dilated stack sees),
and `split_by_record` (the leak-free split that holds out whole records, so no
record's beats land on both sides). The notebook runs all of them and the tests
check each.

## Part 1 - B: leak-free landmark survival on the PBC cohort

File: `longitudinal_exercise.py`. Data: `data/pbcseq.csv` (the pbcseq cohort:
312 patients with primary biliary cirrhosis, repeated liver-lab panels, with
`futime` / `status` follow-up).

The task is a **landmark prediction** that never leaks the future: describe each
patient only by their visits in the **first 2 years**, and predict whether they
die **within 5 years**. Patients must be alive at the 2-year landmark, and any
patient whose 5-year status is unknown (censored before then) is dropped. That
leaves **257 patients, 55 deaths-by-5y (~21%)**. You then compare three models:

- a **logistic baseline** on per-patient summary statistics (mean, last, slope),
- a **tiny RNN** (one small GRU layer),
- a **tiny Transformer** (one small encoder layer).

You implement fourteen auto-graded functions: `load_sequences`, `load_forecasting`,
`summary_features`, `count_parameters`, `make_rnn`, `make_transformer`,
`train_model`, `predict_proba`, `evaluate_auroc`, `persistence_forecast`,
`forecast_mae`, the two leak-free cross-validation loops `cross_val_auroc`
(StratifiedKFold, one row per patient) and `groupkfold_mae` (GroupKFold keyed on
the patient id), and the `lag1_autocorr` diagnostic (why persistence is hard to
beat). The models are kept deliberately small (the tests cap each at under 500
parameters).

**The point is the discipline, not the leaderboard.** With features fixed to the
first 2 years and a 5-year horizon, no model ever sees post-window data. On this
honest setup the three models are close (AUROC ~0.88-0.92) and, with only ~55
events, differences are within noise. The naive alternative ("did the patient
*ever* die?") scores higher but cheats: the last visit sits right before death,
and follow-up length itself encodes the outcome.

**Part 1 - B, continued: next-visit forecasting** (`load_forecasting`, `persistence_forecast`,
`forecast_mae`). A regression task: predict the *next visit's lab vector* from the
history, scored by MAE. Each patient yields many (history, next-visit) examples,
so cross-validate with **`GroupKFold` keyed on the patient id**. The baseline is
**persistence** (next = last observed); because the labs are slowly varying and
highly autocorrelated, it is very hard to beat, and a tiny neural forecaster
typically does not. Same lesson as part 1, now for regression.

## Part 2: tracking (from frames to trajectories)

File: `tracking_exercise.py`. Data: `data/detections.csv` (a small synthetic
cohort of 3 moving cells over 8 frames; columns `frame, det_id, x, y,
track_id`, where `track_id` is the hidden ground truth used only to *score* a
tracker, never to match).

The detector emits unlabelled `(x, y)` dots per frame; your job is to link them
into trajectories. You implement six auto-graded functions
(`cost_matrix`, `greedy_nn`, `hungarian`, `link_tracks`, `count_id_switches`,
`predict_then_match`) and watch the assignment problem play out:

- one frame-to-frame step is an **assignment problem**: greedy
  nearest-neighbour vs the **Hungarian** algorithm (they agree on a clean step),
- a **dish bump** (every detection jumps) breaks greedy but not Hungarian,
- chaining matches into tracks exposes **identity switches**,
- a **crossing** breaks even Hungarian, and **predict-then-match**
  (constant-velocity motion) fixes it, but *not* the bump.

**The point: no single cost wins everywhere.** Global assignment rescues the
discontinuity; motion prediction rescues the crossing; real trackers combine
both.

---

## Layout

```text
.
|-- .github/workflows/tests.yml
|-- notebook.py                  # marimo walkthrough (Sections A, B, C)
|-- time_series_exercise.py      # Part 1-A: ECG beat classifier (CNN/TCN + leak-free split)
|-- longitudinal_exercise.py     # Part 1-B: tiny RNN vs Transformer vs baseline
|-- tracking_exercise.py         # Part 2: data association / tracking
|-- data/mitbih_ch0.npz          # Part 1-A: MIT-BIH ECG (channel 0)
|-- data/pbcseq.csv              # Part 1-B: PBC cohort
|-- data/detections.csv          # Part 2: moving-cell detections
|-- tests/test_assignment.py     # local + autograder tests
|-- requirements.txt
|-- experiences.md
`-- README.md
```

## Feedback

Leave one paragraph in `experiences.md` about what helped and what confused you.

And one last exercise: fill out the **course teaching-evaluation form** on
ILIAS. It is the one task where you grade us, so please be honest; it shapes
next year's course.
