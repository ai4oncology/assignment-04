"""Local + autograder tests for assignment-04.

Run from the repo root:

    pip install -r requirements.txt
    pytest -v tests/

The tests are intentionally lightweight: shape/dtype checks, a few cohort
numbers pinned to the data, parameter-count caps that enforce "tiny" models,
and loose learning checks (a trained model must clearly beat chance). They do
not lock you into one particular implementation; any correct one passes.

The neural-net checks use a fixed seed (42) but assert *ranges*, not exact
values, so they are stable across torch versions / CPU vs GPU.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import prediction_exercise as pr  # noqa: E402
import tracking_exercise as trk  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "pbcseq.csv")
DETECTIONS_PATH = os.path.join(ROOT, "data", "detections.csv")


# ===========================================================================
# Section B -- tracking
# ===========================================================================
@pytest.fixture(scope="module")
def frames():
    """Per-frame (pos, gt) lists from the detections cohort."""
    df = trk.load_detections(DETECTIONS_PATH)
    return trk.frames_as_list(df)


def _aligned(pos_list, gt_list, t):
    """Order a frame's detections by ground-truth id (scoring aid only):
    then the correct assignment is the identity {0:0, 1:1, 2:2}."""
    order = np.argsort(gt_list[t])
    return pos_list[t][order]


def test_cost_matrix_shape_and_metrics():
    src = np.array([[0.0, 0.0], [1.0, 1.0]])
    dst = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
    C = trk.cost_matrix(src, dst)
    assert C.shape == (2, 3)
    assert C[0, 0] == pytest.approx(5.0)          # euclidean 3-4-5
    assert C[0, 1] == pytest.approx(0.0)
    assert C[1, 2] == pytest.approx(0.0)
    C1 = trk.cost_matrix(src, dst, metric="l1")
    assert C1[0, 0] == pytest.approx(7.0)         # |3| + |4|


def test_clean_step_greedy_equals_hungarian(frames):
    pos_list, gt_list = frames
    C = trk.cost_matrix(_aligned(pos_list, gt_list, 0), _aligned(pos_list, gt_list, 1))
    g_assign, g_total, g_collided = trk.greedy_nn(C)
    h_assign, h_total = trk.hungarian(C)
    # clean step: both recover the identity, agree on cost, no collision
    assert g_assign == {0: 0, 1: 1, 2: 2}
    assert h_assign == {0: 0, 1: 1, 2: 2}
    assert g_collided is False
    assert g_total == pytest.approx(h_total)


def test_drift_step_hungarian_beats_greedy(frames):
    pos_list, gt_list = frames
    C = trk.cost_matrix(_aligned(pos_list, gt_list, 1), _aligned(pos_list, gt_list, 2))
    g_assign, g_total, g_collided = trk.greedy_nn(C)
    h_assign, h_total = trk.hungarian(C)
    # the dish bump: greedy collides + mislinks + overpays; Hungarian recovers
    assert g_collided is True
    assert g_assign != {0: 0, 1: 1, 2: 2}
    assert h_assign == {0: 0, 1: 1, 2: 2}
    assert h_total < g_total


def test_link_tracks_id_switches(frames):
    pos_list, gt_list = frames
    greedy_labels = trk.link_tracks(pos_list, lambda C: trk.greedy_nn(C)[0])
    hung_labels = trk.link_tracks(pos_list, lambda C: trk.hungarian(C)[0])
    # pinned to this cohort: greedy trips at bump AND crossing; Hungarian
    # only at the crossing.
    assert trk.count_id_switches(greedy_labels, gt_list) == 4
    assert trk.count_id_switches(hung_labels, gt_list) == 2
    # a perfect linking against itself has zero switches
    assert trk.count_id_switches(gt_list, gt_list) == 0


def test_predict_then_match_holds_through_crossing(frames):
    pos_list, gt_list = frames
    a = trk.predict_then_match(
        _aligned(pos_list, gt_list, 2),
        _aligned(pos_list, gt_list, 3),
        _aligned(pos_list, gt_list, 4),
    )
    # distance-only Hungarian swaps here; the velocity prediction holds identity
    assert a == {0: 0, 1: 1, 2: 2}
    C = trk.cost_matrix(_aligned(pos_list, gt_list, 3), _aligned(pos_list, gt_list, 4))
    assert trk.hungarian(C)[0] != {0: 0, 1: 1, 2: 2}


def test_motion_linker_trips_only_on_bump(frames):
    pos_list, gt_list = frames
    motion_labels = trk.link_tracks_motion(pos_list)
    # constant-velocity carries the crossing but not the discontinuity
    assert trk.count_id_switches(motion_labels, gt_list) == 2


# ---------------------------------------------------------------------------
# Section A -- data: padded sequences + mask + label
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seqs():
    return pr.load_sequences(DATA_PATH)


def test_load_shapes_and_dtypes(seqs):
    X, mask, y = seqs
    assert X.shape == (312, 16, pr.N_FEATURES), f"got {X.shape}"
    assert mask.shape == (312, 16)
    assert y.shape == (312,)
    assert X.dtype == np.float32
    assert mask.dtype == bool


def test_label_prevalence(seqs):
    _, _, y = seqs
    assert set(np.unique(y)) == {0, 1}
    assert int(y.sum()) == 140          # 140 / 312 patients die (~45%)


def test_mask_matches_real_visits(seqs):
    X, mask, y = seqs
    # every patient has at least one visit, at most max_len
    counts = mask.sum(axis=1)
    assert counts.min() >= 1 and counts.max() <= 16
    # padded positions are exactly zero
    assert np.allclose(X[~mask], 0.0)
    # real positions are not all zero (the labs vary)
    assert np.abs(X[mask]).sum() > 0


def test_standardized(seqs):
    X, mask, _ = seqs
    # z-scored across real visits: roughly mean 0, std 1 per channel
    vals = X[mask]                      # (n_real_visits, N_FEATURES)
    assert np.allclose(vals.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(vals.std(axis=0), 1.0, atol=0.1)


def test_summary_features(seqs):
    X, mask, _ = seqs
    F = pr.summary_features(X, mask)
    assert F.shape == (312, 3 * pr.N_FEATURES)
    assert np.all(np.isfinite(F))


# ---------------------------------------------------------------------------
# Section A -- the two tiny models
# ---------------------------------------------------------------------------
def test_rnn_is_tiny_module(seqs):
    model = pr.make_rnn(pr.N_FEATURES)
    assert isinstance(model, torch.nn.Module)
    assert pr.count_parameters(model) < 3000, "keep the RNN tiny"


def test_transformer_is_tiny_module(seqs):
    model = pr.make_transformer(pr.N_FEATURES)
    assert isinstance(model, torch.nn.Module)
    assert pr.count_parameters(model) < 3000, "keep the Transformer tiny"


@pytest.mark.parametrize("factory", ["make_rnn", "make_transformer"])
def test_forward_returns_logits(seqs, factory):
    X, mask, _ = seqs
    model = getattr(pr, factory)(pr.N_FEATURES)
    out = model(torch.tensor(X), torch.tensor(mask))
    assert out.shape == (X.shape[0],), "forward must return one logit per patient"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Section A -- training / evaluation
# ---------------------------------------------------------------------------
def test_evaluate_auroc_edge_cases():
    y = np.array([0, 0, 1, 1])
    assert pr.evaluate_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert pr.evaluate_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_predict_proba_range(seqs):
    X, mask, y = seqs
    model = pr.make_rnn(pr.N_FEATURES)
    p = pr.predict_proba(model, X, mask)
    assert p.shape == (X.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0


@pytest.mark.parametrize("factory", ["make_rnn", "make_transformer"])
def test_training_beats_chance(seqs, factory):
    # A tiny model trained on the full cohort should fit it well above chance.
    X, mask, y = seqs
    model = pr.make_rnn(pr.N_FEATURES) if factory == "make_rnn" \
        else pr.make_transformer(pr.N_FEATURES)
    model = pr.train_model(model, X, mask, y, epochs=60, seed=42)
    auroc = pr.evaluate_auroc(y, pr.predict_proba(model, X, mask))
    assert auroc > 0.65, f"{factory} trained AUROC {auroc:.3f} is too close to chance"


# ---------------------------------------------------------------------------
# Pen-and-paper answers (submission.json, written by notebook.py)
# ---------------------------------------------------------------------------
import json  # noqa: E402

SUBMISSION_PATH = os.path.join(ROOT, "submission.json")


@pytest.fixture(scope="module")
def submission():
    if not os.path.exists(SUBMISSION_PATH):
        pytest.fail(
            "submission.json not found -- open notebook.py in marimo, answer "
            "the pen-and-paper widgets (Sections A and B), and run all cells."
        )
    with open(SUBMISSION_PATH) as f:
        return json.load(f)


def _require(submission, key):
    assert key in submission, f"{key} missing from submission.json"
    val = submission[key]
    assert val is not None, f"{key} is unanswered (None). Answer it in notebook.py."
    return val


# --- Section B: the 3x3 matrix C = [[3,7,4],[1,7,7],[4,3,2]] (worked by hand) ---
def test_q_pp_greedy_total(submission):
    # greedy picks the diagonal: 3 + 7 + 2 = 12
    assert _require(submission, "Q_PP_GREEDY_TOTAL") == 12


def test_q_pp_opt_total(submission):
    # optimal assignment {0:2, 1:0, 2:1}: 4 + 1 + 3 = 8
    assert _require(submission, "Q_PP_OPT_TOTAL") == 8


def test_q_pp_strand(submission):
    # track 2 wants det 1 (cost 1) but greedy already took it for track 1
    assert _require(submission, "Q_PP_STRAND") == "t2"


def test_q_pp_lines(submission):
    # after row/col reduction, 2 lines cover all zeros (< 3 -> needs adjust)
    assert _require(submission, "Q_PP_LINES") == 2


def test_q_pp_adjust(submission):
    # smallest uncovered value in the adjust step is 1
    assert _require(submission, "Q_PP_ADJUST") == 1


def test_q_pp_assign(submission):
    # optimal one-to-one assignment: track1->det3, track2->det1, track3->det2
    assert _require(submission, "Q_PP_ASSIGN") == "a"


def test_q_pp_cross(submission):
    # after the crossing each cell is nearest the other's detection
    assert _require(submission, "Q_PP_CROSS") == "a"


# --- Section A ---
def test_q_pp_b_winner(submission):
    # RNN and Transformer tie, and neither clearly beats the logistic baseline.
    assert _require(submission, "Q_PP_B_WINNER") == "c"


def test_q_pp_b_size(submission):
    # Shrinking the RNN leaves AUROC roughly unchanged: n is too small to use
    # the extra capacity.
    assert _require(submission, "Q_PP_B_SIZE") == "b"


def test_q_pp_b_why(submission):
    # Too few labelled sequences: high-capacity attention overfits, strong
    # baseline is hard to beat.
    assert _require(submission, "Q_PP_B_WHY") == "b"
