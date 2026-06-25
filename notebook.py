"""Marimo walkthrough notebook for assignment-04 (Section B).

Open with:

    marimo edit notebook.py

You implement the functions in `prediction_exercise.py`; the cells below load
them, run the three models on the PBC cohort, show the comparison, and collect
your pen-and-paper answers (auto-saved to `submission.json` for the autograder).
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

    _need = ["numpy", "pandas", "sklearn", "matplotlib", "torch"]
    _missing = [m for m in _need if importlib.util.find_spec(m) is None]
    if _missing:
        print(f"Installing missing dependencies: {', '.join(_missing)} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "-r", "requirements.txt"], check=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Helpers for the pen-and-paper questions (auto-save / restore answers).
    import json as _json
    from pathlib import Path as _Path

    _submission_path = _Path(__file__).with_name("submission.json")
    try:
        submission_data = (
            _json.loads(_submission_path.read_text())
            if _submission_path.exists() else {}
        )
    except _json.JSONDecodeError:
        submission_data = {}

    def submission_radio_default(key, options, default=None):
        saved = submission_data.get(key, default)
        if saved in options:
            return saved
        for opt_key, opt_val in options.items():
            if opt_val == saved:
                return opt_key
        return default

    return (submission_radio_default,)


@app.cell
def _(mo):
    mo.md(r"""
    # Assignment 04, Section B: tiny sequence models on the PBC cohort

    > **The setup.** The hepatology group hands you the **PBC** cohort
    > (primary biliary cirrhosis): 312 patients, each followed over years with
    > repeated liver-lab panels, and an outcome (death / transplant / censored).
    >
    > *"Can a sequence model read a patient's lab trajectory and flag who will
    > die? And is a Transformer worth it over a plain RNN here?"*

    You compare three models on the same task (predict **death** from the visit
    sequence):

    1. a **logistic baseline** on summary statistics (mean, last, slope per lab),
    2. a **tiny RNN** (one small GRU layer),
    3. a **tiny Transformer** (one small encoder layer).

    Implement the functions in `prediction_exercise.py`, then run the cells below.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Prefer the reference solution if present (instructor side); otherwise fall
    # back to the student stubs. You implement the TODOs in prediction_exercise.py;
    # the cells below run against your code once they are filled in.
    try:
        import prediction_solution as pr
    except ModuleNotFoundError:
        import prediction_exercise as pr

    BLUE, RED, GREEN = "#344A9A", "#C8323C", "#00A082"
    DATA = Path(__file__).with_name("data") / "pbcseq.csv"
    X, mask, y = pr.load_sequences(str(DATA))
    print(f"patients={len(y)}  deaths={int(y.sum())} ({y.mean():.0%})  "
          f"feature dim per visit={X.shape[2]}  max visits={X.shape[1]}")
    return (BLUE, GREEN, LogisticRegression, RED, StandardScaler,
            StratifiedKFold, X, mask, np, plt, pr, roc_auc_score, y)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: the logistic baseline

    Collapse each patient to summary statistics (mean / last / slope per lab)
    and fit logistic regression, scored by 5-fold cross-validated AUROC. Any
    sequence model has to clear this bar.
    """)
    return


@app.cell
def _(LogisticRegression, StandardScaler, StratifiedKFold, X, mask, np,
      pr, roc_auc_score, y):
    F = pr.summary_features(X, mask)
    _auc = []
    for _tr, _te in StratifiedKFold(5, shuffle=True, random_state=42).split(F, y):
        _sc = StandardScaler().fit(F[_tr])
        _lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            _sc.transform(F[_tr]), y[_tr])
        _auc.append(roc_auc_score(y[_te], _lr.predict_proba(_sc.transform(F[_te]))[:, 1]))
    auc_base = float(np.mean(_auc))
    print(f"logistic baseline: AUROC {auc_base:.3f} +/- {np.std(_auc):.3f}")
    return (auc_base,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: the two tiny sequence models

    Both are deliberately tiny (a few hundred parameters). Note the parameter
    counts: that is the whole budget the model gets to learn temporal shape from
    ~250 training patients per fold.
    """)
    return


@app.cell
def _(StratifiedKFold, X, mask, np, pr, roc_auc_score, y):
    print(f"RNN params:         {pr.count_parameters(pr.make_rnn(X.shape[2]))}")
    print(f"Transformer params: {pr.count_parameters(pr.make_transformer(X.shape[2]))}")

    def cv_auroc(make):
        scores = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            model = pr.train_model(make(), X[tr], mask[tr], y[tr], epochs=60, seed=42)
            scores.append(roc_auc_score(y[te], pr.predict_proba(model, X[te], mask[te])))
        return float(np.mean(scores)), float(np.std(scores))

    auc_rnn, sd_rnn = cv_auroc(lambda: pr.make_rnn(X.shape[2]))
    auc_tfm, sd_tfm = cv_auroc(lambda: pr.make_transformer(X.shape[2]))
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
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_title("PBC death prediction: do the sequence models earn their keep?")
    for _b, _v in zip(_bars, _vals):
        ax.annotate(f"{_v:.3f}", (_b.get_x() + _b.get_width() / 2, _v + 0.005),
                    ha="center", fontsize=12)
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pen-and-paper questions

    Answer the three questions below from what you saw. They auto-save to
    `submission.json` (read by the autograder).
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "The Transformer clearly wins (attention beats recurrence).":          "a",
        "The RNN clearly wins.":                                               "b",
        "RNN and Transformer roughly tie, and neither clearly beats the "
        "logistic baseline.":                                                  "c",
    }
    q_pp_b_winner = mo.ui.radio(
        options=_opts,
        label="(B1) How do the three models compare on this cohort?",
        value=submission_radio_default("Q_PP_B_WINNER", _opts),
    )
    q_pp_b_winner
    return (q_pp_b_winner,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "AUROC collapses to chance: the model is now too small to learn.":     "a",
        "AUROC stays roughly the same: n is too small to use the extra "
        "capacity, so a 150-parameter RNN matches a 1000-parameter one.":      "b",
        "AUROC improves a lot: smaller is always better.":                     "c",
    }
    q_pp_b_size = mo.ui.radio(
        options=_opts,
        label="(B2) You shrink the RNN from hidden=16 to hidden=4. What happens?",
        value=submission_radio_default("Q_PP_B_SIZE", _opts),
    )
    q_pp_b_size
    return (q_pp_b_size,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "Transformers cannot model time series.":                              "a",
        "With only ~300 labelled sequences, a high-capacity attention model "
        "overfits, and a strong summary-statistic baseline is hard to beat.":  "b",
        "The labs carry no information about death.":                          "c",
    }
    q_pp_b_why = mo.ui.radio(
        options=_opts,
        label="(B3) Why does the Transformer not pull ahead here?",
        value=submission_radio_default("Q_PP_B_WHY", _opts),
    )
    q_pp_b_why
    return (q_pp_b_why,)


@app.cell
def _(mo, q_pp_b_size, q_pp_b_why, q_pp_b_winner):
    import json as _json
    from pathlib import Path as _Path

    _submission = {
        "Q_PP_B_WINNER": q_pp_b_winner.value,
        "Q_PP_B_SIZE":   q_pp_b_size.value,
        "Q_PP_B_WHY":    q_pp_b_why.value,
    }
    _path = _Path(__file__).with_name("submission.json")
    _path.write_text(_json.dumps(_submission, indent=2))
    _n_unanswered = sum(1 for v in _submission.values() if v is None)
    mo.md(f"**Answers auto-saved to `{_path.name}`** "
          f"({len(_submission)} questions, {_n_unanswered} unanswered).")
    return


if __name__ == "__main__":
    app.run()
