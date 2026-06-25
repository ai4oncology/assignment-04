# Assignment 04: Longitudinal & Predictive Modeling

**ML4Health 2026**

This assignment covers two consecutive course topics:

1. **Longitudinal Modeling**: handling repeated measurements per patient over
   time (within-subject correlation, time-to-event, survival basics).
2. **Predictive Modeling**: risk stratification from a patient's trajectory,
   leakage-aware evaluation, ROC / AUC.

---

## Section B (ready): tiny sequence models on the PBC cohort

File: `prediction_exercise.py`. Data: `data/pbcseq.csv` (the pbcseq cohort:
312 patients with primary biliary cirrhosis, repeated liver-lab panels, with a
death / transplant / censoring outcome).

You predict **death** from each patient's sequence of lab visits and compare
three models:

- a **logistic baseline** on per-patient summary statistics (mean, last, slope),
- a **tiny RNN** (one small GRU layer),
- a **tiny Transformer** (one small encoder layer).

You implement eight auto-graded functions: `load_sequences`, `summary_features`,
`count_parameters`, `make_rnn`, `make_transformer`, `train_model`,
`predict_proba`, `evaluate_auroc`. The models are kept deliberately small (the
tests cap each at under 3000 parameters).

**The point is not to win.** On ~300 patients you should find the RNN and the
Transformer roughly **tie each other**, and **neither reliably beats the
logistic baseline**. Small clinical cohorts reward small models and strong
baselines; a Transformer's extra capacity is wasted (and overfits) with only a
few hundred labelled sequences. Reading off a leaderboard without a baseline
would have hidden that.

Run the tests:

```bash
pip install -r requirements.txt
pytest -v tests/
```

> Note: this section needs `torch` (in `requirements.txt`). CPU is fine; the
> models are tiny and 5-fold cross-validation runs in seconds.

## Section A (under construction)

File: `longitudinal_exercise.py`. Within-subject correlation and time-to-event
basics on the same cohort. Released once the longitudinal lecture is finalised.

---

## Layout

```text
.
|-- .github/workflows/tests.yml
|-- notebook.py                  # marimo walkthrough
|-- longitudinal_exercise.py     # Section A: longitudinal modeling (placeholder)
|-- prediction_exercise.py       # Section B: tiny RNN vs Transformer vs baseline
|-- data/pbcseq.csv              # PBC cohort
|-- tests/test_assignment.py     # local + autograder tests
|-- requirements.txt
|-- experiences.md
`-- README.md
```

## Feedback

Leave one paragraph in `experiences.md` about what helped and what confused you.
