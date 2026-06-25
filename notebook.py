"""Marimo walkthrough notebook for assignment-04 (Longitudinal & Predictive).

Open with:

    marimo edit notebook.py

Two sections, one cohort-of-problems each:

  * **Section A -- Sequence models** (predictive): predict death from a
    patient's lab trajectory on the PBC cohort. You implement the functions in
    `prediction_exercise.py`; the cells run three models on `data/pbcseq.csv`.
  * **Section B -- Tracking** (longitudinal): chain unlabelled per-frame
    detections of moving cells into trajectories. You implement the functions
    in `tracking_exercise.py`; the cells run them on `data/detections.csv`.

Both sections end in pen-and-paper questions whose answers auto-save to
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

    return Path, np, plt


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
    # Assignment 04 — Longitudinal & Predictive Modeling

    Two short sections on the same theme — **reading signal out of data that
    changes over time** — but from opposite ends:

    - **Section A — Sequence models.** Predict an outcome from a patient's
      lab *trajectory*, and ask whether a Transformer earns its keep over a
      plain RNN and a logistic baseline. (Implement `prediction_exercise.py`.)
    - **Section B — Tracking.** Frame-to-frame *data association*: chain
      unlabelled detections of moving cells into trajectories. (Implement
      `tracking_exercise.py`.)

    Work top to bottom. Each section ends in pen-and-paper questions that
    auto-save to `submission.json` (read by the autograder).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Section A — tiny sequence models on the PBC cohort

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
def _(Path):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    # Prefer the reference solution if present (instructor side); otherwise fall
    # back to the student stubs. You implement the TODOs in prediction_exercise.py.
    try:
        import prediction_solution as pr
    except ModuleNotFoundError:
        import prediction_exercise as pr

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
    ## Step 1: the logistic baseline

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
    ### Section A pen-and-paper questions

    Answer the three questions below from what you saw. They auto-save to
    `submission.json` (read by the autograder).
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "The Transformer clearly wins (attention beats recurrence).": "a",
        "The RNN clearly wins.": "b",
        "RNN and Transformer roughly tie, and neither clearly beats the "
        "logistic baseline.": "c",
    }
    q_pp_b_winner = mo.ui.radio(
        options=_opts,
        label="(A1) How do the three models compare on this cohort?",
        value=submission_radio_default("Q_PP_B_WINNER", _opts),
    )
    q_pp_b_winner
    return (q_pp_b_winner,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "AUROC collapses to chance: the model is now too small to learn.": "a",
        "AUROC stays roughly the same: n is too small to use the extra "
        "capacity, so a 150-parameter RNN matches a 1000-parameter one.": "b",
        "AUROC improves a lot: smaller is always better.": "c",
    }
    q_pp_b_size = mo.ui.radio(
        options=_opts,
        label="(A2) You shrink the RNN from hidden=16 to hidden=4. What happens?",
        value=submission_radio_default("Q_PP_B_SIZE", _opts),
    )
    q_pp_b_size
    return (q_pp_b_size,)


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _opts = {
        "Transformers cannot model time series.": "a",
        "With only ~300 labelled sequences, a high-capacity attention model "
        "overfits, and a strong summary-statistic baseline is hard to beat.": "b",
        "The labs carry no information about death.": "c",
    }
    q_pp_b_why = mo.ui.radio(
        options=_opts,
        label="(A3) Why does the Transformer not pull ahead here?",
        value=submission_radio_default("Q_PP_B_WHY", _opts),
    )
    q_pp_b_why
    return (q_pp_b_why,)


@app.cell
def _(mo):
    mo.md(r"""
    # Section B — Tracking: from frames to trajectories

    > **The setup.** Your lab images a dish of migrating cells under a
    > microscope. Every few minutes the segmentation pipeline spits out
    > a *frame*: a handful of $(x, y)$ detections, one per cell. The
    > catch — **the detector does not know which cell is which**. Frame
    > to frame, the dots are unlabelled and arrive in arbitrary order.
    >
    > A biologist wants *trajectories*: which dot in frame $t{+}1$ is the
    > same cell as which dot in frame $t$, chained all the way through
    > the movie. That linking step is **data association**, and it is the
    > whole job of this section.

    Your plan:

    - **A.** Treat one frame-to-frame step as an **assignment problem** —
      build a cost matrix of distances, then solve it two ways:
      **greedy nearest-neighbour** and the **Hungarian algorithm**. On a
      clean step they agree.
    - **B.** A **dish bump**: between two frames the microscope stage gets
      knocked and *every* detection jumps. Greedy strands a track and pays
      a higher cost; **Hungarian** recovers the right identities.
    - **C.** **Chain** one-step matches into full tracks and count
      **identity switches** — greedy vs Hungarian.
    - **D.** Identities also **break at a crossing**. **Predict-then-match**
      (constant-velocity motion) carries them through the crossing — but
      *not* through the bump. **No single cost wins everywhere.**

    The cohort: **3 cells over 8 frames** in `data/detections.csv`. Two of
    them cross paths, and the dish gets bumped early on.
    """)
    return


@app.cell
def _(Path):
    # Prefer the reference solution if present (instructor side); otherwise fall
    # back to the student stubs. You implement the TODOs in tracking_exercise.py.
    try:
        import tracking_solution as trk
    except ModuleNotFoundError:
        import tracking_exercise as trk

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
        f"ground truth — used only to *score* a tracker, never to match."
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

    Left: the raw detections, coloured by **frame** — this is all a
    tracker gets. Right: the same points coloured by the **hidden
    ground-truth identity**, with the true trajectories drawn in. Notice
    two tracks heading straight for each other (a **crossing**), and the
    sharp **vertical jump** of every track at frame 2 — the dish bump.
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
    # never gets to do this — it only sees the scrambled det_id order.
    def aligned(t):
        order = np.argsort(gt_list[t])
        return pos_list[t][order]

    return (aligned,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A — one frame-to-frame step is an assignment problem

    Take frame 0 and frame 1. Each has the same three detections. To link
    them we score every (frame-0 dot, frame-1 dot) pair by how far apart
    they are — that is your **cost matrix** `cost_matrix(src, dst)` — and
    then pick a one-to-one matching.

    Two ways to pick:

    - **`greedy_nn`** — walk the rows top to bottom; each track grabs its
      nearest *still-free* detection. Fast, intuitive, order-dependent.
    - **`hungarian`** — minimise the *total* cost over all one-to-one
      matchings (via `scipy.optimize.linear_sum_assignment`), with a
      guarantee.

    Below we use the **real** frames 0 and 1 (ordered by identity so the
    correct answer is the diagonal). On this clean step the two agree.
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
        f"**Clean step (frames 0 → 1)** — cost matrix (rounded):\n\n"
        f"`{np.round(clean_cost, 2).tolist()}`\n\n"
        f"- greedy : assignment `{clean_greedy[0]}`, total **{clean_greedy[1]:.2f}**, "
        f"collided = `{clean_greedy[2]}`\n"
        f"- hungarian: assignment `{clean_hung[0]}`, total **{clean_hung[1]:.2f}**\n\n"
        f"Both pick the diagonal `{{0:0, 1:1, 2:2}}` — when the nearest "
        f"neighbours don't conflict, greedy is already optimal."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## B — the dish gets bumped (drift breaks greedy)

    Between frame 1 and frame 2 someone knocks the microscope stage, so
    **every detection jumps** by the same offset — and stays shifted from
    then on. The jump is large enough that, looking only at raw distance,
    one cell's new detection lands closer to a *different* cell's old spot.

    Watch what greedy does on the **real** frames 1 → 2. Processing rows
    top to bottom, an early row grabs a cheap-looking dot that truly
    belongs to another track, and a later row is stranded with an
    expensive scrap: the result is **wrong** (identities mislinked) *and*
    **cost-suboptimal**. The **Hungarian** algorithm optimises the total
    globally — and because a rigid shift preserves the relative geometry,
    it recovers the correct identities.
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
        f"**Drift step (frames 1 → 2)** — cost matrix (rounded):\n\n"
        f"`{np.round(drift_cost, 2).tolist()}`\n\n"
        f"| method | assignment | total cost |\n"
        f"|---|---|---|\n"
        f"| greedy | `{drift_greedy[0]}` | **{drift_greedy[1]:.2f}** |\n"
        f"| hungarian | `{drift_hung[0]}` | **{drift_hung[1]:.2f}** |\n\n"
        f"The correct answer is the diagonal `{{0:0, 1:1, 2:2}}`. Greedy "
        f"returns `{drift_greedy[0]}` (collided = `{drift_greedy[2]}`) — "
        f"wrong — and pays **{drift_greedy[1] - drift_hung[1]:.2f}** extra "
        f"cost over Hungarian, which recovers the diagonal. *Same data, "
        f"same cost function — a better solver is what saved us here.*"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C — chain one-step matches into tracks

    A trajectory is just one-step matches stitched together: seed labels
    from frame 0, then for every later frame match the previous detections
    to the current ones and carry each label forward. `link_tracks(pos,
    matcher)` does exactly that — pass it either matcher.

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
        f"greedy linking — {sw_greedy} ID switches",
        _a1,
        gt_seq=gt_list,
    )
    plot_tracks(
        pos_list,
        hung_labels,
        f"Hungarian linking — {sw_hung} ID switches",
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
        f"Greedy trips **twice** — once at the dish bump, once at the "
        f"crossing. Hungarian fixes the bump (global optimisation), so it "
        f"trips only at the **crossing**. Those remaining switches are a "
        f"different kind of failure: there, distance itself is the wrong "
        f"signal — even the optimal matching swaps identities, because once "
        f"two cells have crossed, the swap *is* the cheapest assignment. "
        f"That is what part D fixes."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## D — predict, then match (and where it *doesn't* help)

    At a crossing the two cells are momentarily nearest the *wrong*
    neighbour, so any position-only cost mislinks them. The fix:
    **don't match on where a cell is, match on where it is going.** Each
    track carries a velocity; predict its next position
    (`pred = cur + (cur − prev)`) and match the *predictions* to the new
    detections. This is the core idea behind SORT-style trackers.

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
        f"`{cross_dist}` — a swap (should be the diagonal). "
        f"Predict-then-match: `{cross_pred}` — identities held. The "
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
        f"predict-then-match — {sw_motion} ID switches",
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

    Predict-then-match sails through the **crossing** — but it *also* trips,
    at the **dish bump**: a constant-velocity model assumes smooth motion and
    cannot anticipate a sudden jump, so its prediction lands every cell off by
    the same offset and the match breaks (just like greedy). That is not a bug
    to hide — it is the price of a simple motion model.

    So the two failure modes need *different* fixes: global assignment
    (Hungarian) rescues the discontinuity, motion prediction rescues the
    crossing. Real trackers (e.g. SORT) combine **both** — a motion model to
    build the cost *and* a global solver to assign it — plus gating/re-init to
    catch the discontinuities neither handles alone."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Section B pen-and-paper — run the Hungarian algorithm by hand

    Work this **3×3** cost matrix (tracks = rows, detections = columns)
    with pen and paper. It is a *fresh* matrix — not one from the lecture —
    so you have to actually turn the crank.

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
        label="(B-a) Greedy NN (rows top→bottom): what TOTAL cost does it pick?",
        value=submission_default("Q_PP_GREEDY_TOTAL"),
    )
    q_pp_greedy_total
    return (q_pp_greedy_total,)


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_pp_opt_total = mo.ui.number(
        start=0,
        stop=100,
        step=1,
        label="(B-b) Optimal (Hungarian) TOTAL cost:",
        value=submission_default("Q_PP_OPT_TOTAL"),
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
            "(B-c) Which track does greedy strand — forced off its own "
            "nearest detection onto an expensive one?"
        ),
        value=submission_radio_default("Q_PP_STRAND", _q_strand_opts),
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
            "(B-d) After the row- and column-reductions, what is the "
            "MINIMUM number of lines that cover all zeros?"
        ),
        value=submission_default("Q_PP_LINES"),
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
            "(B-e) In the adjust step, what is the smallest UNCOVERED "
            "value you subtract / add?"
        ),
        value=submission_default("Q_PP_ADJUST"),
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
        label="(B-f) What is the optimal (minimum-cost) one-to-one assignment?",
        value=submission_radio_default("Q_PP_ASSIGN", _q_assign_opts),
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
            "(B-g) Conceptual: at the crossing, both greedy and Hungarian swap "
            "the two identities. What is the underlying reason?"
        ),
        value=submission_radio_default("Q_PP_CROSS", _q_cross_opts),
    )
    q_pp_cross
    return (q_pp_cross,)


@app.cell
def _(
    mo,
    q_pp_adjust,
    q_pp_assign,
    q_pp_b_size,
    q_pp_b_why,
    q_pp_b_winner,
    q_pp_cross,
    q_pp_greedy_total,
    q_pp_lines,
    q_pp_opt_total,
    q_pp_strand,
):
    # Collect BOTH sections' pen-and-paper answers into a single submission.json
    # for the autograder.
    import json as _json
    from pathlib import Path as _Path

    _submission = {
        # Section A -- sequence models
        "Q_PP_B_WINNER": q_pp_b_winner.value,
        "Q_PP_B_SIZE": q_pp_b_size.value,
        "Q_PP_B_WHY": q_pp_b_why.value,
        # Section B -- tracking
        "Q_PP_GREEDY_TOTAL": q_pp_greedy_total.value,
        "Q_PP_OPT_TOTAL": q_pp_opt_total.value,
        "Q_PP_STRAND": q_pp_strand.value,
        "Q_PP_LINES": q_pp_lines.value,
        "Q_PP_ADJUST": q_pp_adjust.value,
        "Q_PP_ASSIGN": q_pp_assign.value,
        "Q_PP_CROSS": q_pp_cross.value,
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
