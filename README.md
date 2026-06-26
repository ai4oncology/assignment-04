# Assignment 04: Longitudinal & Predictive Modeling

**ML4Health 2026**

This assignment covers two consecutive course topics, in the order the lecture
presents them:

1. **Predictive Modeling**: risk stratification from a patient's trajectory,
   leakage-aware evaluation, ROC / AUC.
2. **Longitudinal Modeling**: following the same units over time. Here:
   multi-object **tracking** — chaining unlabelled per-frame detections of
   moving cells into trajectories (data association).

Both sections are driven by `notebook.py` (a marimo walkthrough) and graded by
`tests/test_assignment.py`.

> Run the tests for **both** sections with:
>
> ```bash
> pip install -r requirements.txt
> pytest -v tests/
> ```
>
> Section A needs `torch` (in `requirements.txt`); CPU is fine — the models are
> tiny and 5-fold cross-validation runs in seconds.

---

## Section A: leak-free landmark survival on the PBC cohort

File: `prediction_exercise.py`. Data: `data/pbcseq.csv` (the pbcseq cohort:
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

You implement eleven auto-graded functions: `load_sequences`, `load_forecasting`,
`summary_features`, `count_parameters`, `make_rnn`, `make_transformer`,
`train_model`, `predict_proba`, `evaluate_auroc`, `persistence_forecast`,
`forecast_mae`. The models are kept deliberately small (the tests cap each at
under 500 parameters).

**The point is the discipline, not the leaderboard.** With features fixed to the
first 2 years and a 5-year horizon, no model ever sees post-window data. On this
honest setup the three models are close (AUROC ~0.88-0.92) and, with only ~55
events, differences are within noise. The naive alternative ("did the patient
*ever* die?") scores higher but cheats: the last visit sits right before death,
and follow-up length itself encodes the outcome.

**Part 2: next-visit forecasting** (`load_forecasting`, `persistence_forecast`,
`forecast_mae`). A regression task: predict the *next visit's lab vector* from the
history, scored by MAE. Each patient yields many (history, next-visit) examples,
so cross-validate with **`GroupKFold` keyed on the patient id**. The baseline is
**persistence** (next = last observed); because the labs are slowly varying and
highly autocorrelated, it is very hard to beat, and a tiny neural forecaster
typically does not. Same lesson as part 1, now for regression.

## Section B: tracking — from frames to trajectories

File: `tracking_exercise.py`. Data: `data/detections.csv` (a small synthetic
cohort of 3 moving cells over 8 frames; columns `frame, det_id, x, y,
track_id`, where `track_id` is the hidden ground truth used only to *score* a
tracker, never to match).

The detector emits unlabelled `(x, y)` dots per frame; your job is to link them
into trajectories. You implement six auto-graded functions —
`cost_matrix`, `greedy_nn`, `hungarian`, `link_tracks`, `count_id_switches`,
`predict_then_match` — and watch the assignment problem play out:

- one frame-to-frame step is an **assignment problem**: greedy
  nearest-neighbour vs the **Hungarian** algorithm (they agree on a clean step),
- a **dish bump** (every detection jumps) breaks greedy but not Hungarian,
- chaining matches into tracks exposes **identity switches**,
- a **crossing** breaks even Hungarian, and **predict-then-match**
  (constant-velocity motion) fixes it — but *not* the bump.

**The point: no single cost wins everywhere.** Global assignment rescues the
discontinuity; motion prediction rescues the crossing; real trackers combine
both.

---

## Layout

```text
.
|-- .github/workflows/tests.yml
|-- notebook.py                  # marimo walkthrough (both sections)
|-- prediction_exercise.py       # Section A: tiny RNN vs Transformer vs baseline
|-- tracking_exercise.py         # Section B: data association / tracking
|-- data/pbcseq.csv              # Section A: PBC cohort
|-- data/detections.csv          # Section B: moving-cell detections
|-- tests/test_assignment.py     # local + autograder tests
|-- requirements.txt
|-- experiences.md
`-- README.md
```

## Feedback

Leave one paragraph in `experiences.md` about what helped and what confused you.
