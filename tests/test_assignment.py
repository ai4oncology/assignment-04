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
import longitudinal_exercise as pr  # noqa: E402
import time_series_exercise as ts  # noqa: E402
import tracking_exercise as trk  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "pbcseq.csv")
DETECTIONS_PATH = os.path.join(ROOT, "data", "detections.csv")


# ===========================================================================
# Part 2 -- tracking
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
# Part 1-B -- data: padded sequences + mask + label
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seqs():
    return pr.load_sequences(DATA_PATH)


def test_load_shapes_and_dtypes(seqs):
    X, mask, y = seqs
    assert X.shape == (257, 8, pr.N_FEATURES), f"got {X.shape}"
    assert mask.shape == (257, 8)
    assert y.shape == (257,)
    assert X.dtype == np.float32
    assert mask.dtype == bool


def test_label_prevalence(seqs):
    _, _, y = seqs
    assert set(np.unique(y)) == {0, 1}
    assert int(y.sum()) == 55           # 55 / 257 die within the 5-year horizon (~21%)


def test_mask_matches_real_visits(seqs):
    X, mask, y = seqs
    # every patient has at least one visit, at most max_len
    counts = mask.sum(axis=1)
    assert counts.min() >= 1 and counts.max() <= 8
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
    assert F.shape == (257, 3 * pr.N_FEATURES)
    assert np.all(np.isfinite(F))


# ---------------------------------------------------------------------------
# Part 1-B -- the two tiny models
# ---------------------------------------------------------------------------
def test_rnn_is_tiny_module(seqs):
    model = pr.make_rnn(pr.N_FEATURES)
    assert isinstance(model, torch.nn.Module)
    assert pr.count_parameters(model) < 500, "keep the RNN tiny"


def test_transformer_is_tiny_module(seqs):
    model = pr.make_transformer(pr.N_FEATURES)
    assert isinstance(model, torch.nn.Module)
    assert pr.count_parameters(model) < 500, "keep the Transformer tiny"


@pytest.mark.parametrize("factory", ["make_rnn", "make_transformer"])
def test_forward_returns_logits(seqs, factory):
    X, mask, _ = seqs
    model = getattr(pr, factory)(pr.N_FEATURES)
    out = model(torch.tensor(X), torch.tensor(mask))
    assert out.shape == (X.shape[0],), "forward must return one logit per patient"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Part 1-B -- training / evaluation
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
# Part 1-B (continued) -- next-visit forecasting (regression)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def forecast():
    return pr.load_forecasting(DATA_PATH)


def test_forecast_shapes(forecast):
    X, mask, Y, groups = forecast
    assert X.shape == (1558, 16, pr.N_FEATURES), f"got {X.shape}"
    assert mask.shape == (1558, 16)
    assert Y.shape == (1558, pr.N_FEATURES)
    assert groups.shape == (1558,)
    assert len(np.unique(groups)) == 283


def test_persistence_is_last_visit(forecast):
    X, mask, Y, _ = forecast
    pred = pr.persistence_forecast(X, mask)
    assert pred.shape == Y.shape
    last = mask.sum(1) - 1                       # each row's last real visit
    assert np.allclose(pred, X[np.arange(len(X)), last])


def test_forecast_mae(forecast):
    X, mask, Y, _ = forecast
    assert pr.forecast_mae(Y, Y) == pytest.approx(0.0)
    mae = pr.forecast_mae(Y, pr.persistence_forecast(X, mask))
    assert 0.3 < mae < 0.7                       # persistence ~0.46 in standardized units


# ---------------------------------------------------------------------------
# Part 1-B -- leak-free cross-validation loops + autocorrelation diagnostic
# ---------------------------------------------------------------------------
def test_lag1_autocorr():
    import pandas as pd
    # within each patient the lab marches 1,2,3 -> consecutive (prev,cur) pairs are
    # perfectly correlated; "albumin" is not log-transformed.
    df = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "day": [0, 1, 2, 0, 1, 2],
        "albumin": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    })
    assert pr.lag1_autocorr(df, "albumin") == pytest.approx(1.0)
    # on the real cohort the labs are strongly autocorrelated (why persistence wins)
    real = pd.read_csv(DATA_PATH)
    assert pr.lag1_autocorr(real, "bili") > 0.7


def test_cross_val_auroc(seqs):
    X, mask, y = seqs
    mean, std = pr.cross_val_auroc(lambda: pr.make_rnn(pr.N_FEATURES), X, mask, y, epochs=25)
    assert 0.0 <= mean <= 1.0 and std >= 0.0
    assert mean > 0.6                            # cross-validated, clearly beats chance


def test_groupkfold_mae(forecast):
    X, mask, Y, groups = forecast

    class _LastLinear(torch.nn.Module):
        # tiny forecaster: read the last real visit, map it linearly to the next
        def __init__(self, d):
            super().__init__()
            self.fc = torch.nn.Linear(d, d)

        def forward(self, x, m):
            last = m.sum(1).clamp(min=1).long() - 1
            return self.fc(x[torch.arange(len(x)), last])

    mae = pr.groupkfold_mae(lambda: _LastLinear(pr.N_FEATURES), X, mask, Y, groups, epochs=15)
    assert np.isfinite(mae) and 0.0 < mae < 1.5


# ---------------------------------------------------------------------------
# Part 1-A -- leak-free ECG split (time_series_exercise.split_by_record)
# ---------------------------------------------------------------------------
def test_split_by_record_is_leak_free():
    # 12 records, 7 beats each (record id repeated along the beats).
    rec = np.repeat(np.arange(12), 7)
    is_test = np.asarray(ts.split_by_record(rec, test_frac=0.25, seed=0))

    # must be a boolean mask, one entry per beat
    assert is_test.dtype == bool
    assert is_test.shape == (len(rec),)
    assert is_test.any() and not is_test.all()           # both sides non-empty

    # the whole point: no record may appear on both sides
    train_recs = set(rec[~is_test].tolist())
    test_recs = set(rec[is_test].tolist())
    assert not (train_recs & test_recs), "a record's beats leak across the split"

    # roughly the requested fraction of RECORDS (not beats), and reproducible
    assert 1 <= len(test_recs) <= 5
    again = np.asarray(ts.split_by_record(rec, test_frac=0.25, seed=0))
    assert np.array_equal(is_test, again)                # same seed -> same split


def test_ecg_summary_features():
    # two beats, one channel, four samples each: (n_beats, 1, beat_len)
    X = np.array([[[1.0, 2.0, 3.0, 4.0]],
                  [[-1.0, 0.0, 0.0, 5.0]]], dtype=np.float32)
    F = ts.ecg_summary_features(X)
    assert F.shape == (2, 5)                              # [mean, std, min, max, last]
    assert F[0, 0] == pytest.approx(2.5)                  # mean of 1..4
    assert F[0, 2] == pytest.approx(1.0)                  # min
    assert F[0, 3] == pytest.approx(4.0)                  # max
    assert F[0, 4] == pytest.approx(4.0)                  # last value
    assert F[1, 4] == pytest.approx(5.0)


def test_extract_beats():
    sig = np.arange(20, dtype=np.float32)
    beats, keep = ts.extract_beats(sig, [5, 10, 18], before=3, after=3)
    assert keep.tolist() == [True, True, False]          # last window runs off the end
    assert beats.shape == (2, 6)                          # before + after = 6 samples
    assert np.allclose(beats.mean(axis=1), 0.0, atol=1e-5)   # each window z-scored
    assert np.allclose(beats.std(axis=1), 1.0, atol=1e-3)


def test_receptive_field():
    assert ts.receptive_field(5, (1, 1)) == 9            # the shallow CNN (~9 samples)
    assert ts.receptive_field(3, (1, 2, 4, 8, 16)) == 63  # the dilated TCN (~63 samples)


@pytest.mark.parametrize("factory", ["make_beat_cnn", "make_tcn"])
def test_beat_model_forward(factory):
    model = getattr(ts, factory)()
    assert isinstance(model, torch.nn.Module)
    out = model(torch.zeros(4, 1, 200))                  # (batch, 1 channel, beat_len)
    assert out.shape == (4,), "forward must return one logit per beat"
    assert torch.isfinite(out).all()


def test_train_cnn_learns_separable():
    rng = np.random.default_rng(0)
    n, L = 64, 32
    ramp = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    X = np.zeros((2 * n, 1, L), dtype=np.float32)
    X[:n, 0, :] = rng.normal(0, 0.1, (n, L)) + ramp      # class 0: rising
    X[n:, 0, :] = rng.normal(0, 0.1, (n, L)) - ramp      # class 1: falling
    y = np.array([0] * n + [1] * n)
    p = ts.train_cnn(ts.make_beat_cnn, X, y, X, epochs=8, seed=0)
    assert p.shape == (2 * n,)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert pr.evaluate_auroc(y, p) > 0.7                 # clearly learns this trivial split


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
            "the pen-and-paper widgets (Sections A, B and C), and run all cells."
        )
    with open(SUBMISSION_PATH) as f:
        return json.load(f)


def _require(submission, key):
    assert key in submission, f"{key} missing from submission.json"
    val = submission[key]
    assert val is not None, f"{key} is unanswered (None). Answer it in notebook.py."
    return val


# --- Part 2: the 3x3 matrix C = [[3,7,4],[1,7,7],[4,3,2]] (worked by hand) ---
def test_q_pp_c_greedy_total(submission):
    # greedy picks the diagonal: 3 + 7 + 2 = 12
    assert _require(submission, "Q_PP_C_GREEDY_TOTAL") == 12


def test_q_pp_c_opt_total(submission):
    # optimal assignment {0:2, 1:0, 2:1}: 4 + 1 + 3 = 8
    assert _require(submission, "Q_PP_C_OPT_TOTAL") == 8


def test_q_pp_c_strand(submission):
    # track 2 wants det 1 (cost 1) but greedy already took it for track 1
    assert _require(submission, "Q_PP_C_STRAND") == "t2"


def test_q_pp_c_lines(submission):
    # after row/col reduction, 2 lines cover all zeros (< 3 -> needs adjust)
    assert _require(submission, "Q_PP_C_LINES") == 2


def test_q_pp_c_adjust(submission):
    # smallest uncovered value in the adjust step is 1
    assert _require(submission, "Q_PP_C_ADJUST") == 1


def test_q_pp_c_assign(submission):
    # optimal one-to-one assignment: track1->det3, track2->det1, track3->det2
    assert _require(submission, "Q_PP_C_ASSIGN") == "a"


def test_q_pp_c_cross(submission):
    # after the crossing each cell is nearest the other's detection
    assert _require(submission, "Q_PP_C_CROSS") == "a"


# --- Part 1-B: sequence models (leak-free landmark task) ---
def test_q_pp_b_leak(submission):
    # "ever died" leaks: last visit is near death, follow-up length encodes outcome.
    assert _require(submission, "Q_PP_B_LEAK") == "b"


def test_q_pp_b_compare(submission):
    # leak-free, the three models are close (~0.88-0.92), within noise on ~55 events.
    assert _require(submission, "Q_PP_B_COMPARE") == "a"


def test_q_pp_b_size(submission):
    # shrinking the RNN leaves AUROC about the same: capacity is not the bottleneck.
    assert _require(submission, "Q_PP_B_SIZE") == "b"


def test_q_pp_a_ecg_split(submission):
    # False: beats must be split by record/patient (GroupKFold), so fold sizes
    # follow which records land where, not a fixed 80% of the beat count.
    assert _require(submission, "Q_PP_A_ECG_SPLIT") == "false"


def test_q_pp_b_attn_avg(submission):
    # Patient 4: uniform attention -> representation = average of visits (mean baseline).
    assert _require(submission, "Q_PP_B_ATTN_AVG") == "4"


def test_q_pp_b_attn_recency(submission):
    # Patient 1: every row's mass on the last visit -> recency / last-value.
    assert _require(submission, "Q_PP_B_ATTN_RECENCY") == "1"


def test_q_pp_b_attn_self(submission):
    # Patient 2: diagonal -> each visit attends to itself, no context mixing.
    assert _require(submission, "Q_PP_B_ATTN_SELF") == "2"


def test_q_pp_b_attn_baseline(submission):
    # Patient 3: every row's mass on the first visit -> anchored to baseline.
    assert _require(submission, "Q_PP_B_ATTN_BASELINE") == "3"


def test_q_pp_b_leakfree(submission):
    # The trick: the honest team looks the most suspicious. Team A is leak-free:
    # an external, separate cohort's stats use NO information from this test set
    # (distribution shift, maybe, but not leakage). The innocent-sounding ones
    # leak: Team B's outlier filter is computed over the whole dataset incl. test
    # (test-dependent sample selection), Team C cherry-picks the best of 20 splits
    # (optimistic selection), Team D's total visit count encodes follow-up length.
    assert _require(submission, "Q_PP_B_LEAKFREE") == "a"


def test_q_pp_eval(submission):
    # Tick "yes" once the course teaching-evaluation form is in.
    assert _require(submission, "Q_PP_EVAL") == "yes"
