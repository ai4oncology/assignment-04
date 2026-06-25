"""Section A: Longitudinal modelling -- multi-object tracking.

Paired with the longitudinal/tracking beat of the assignment narrative (Week-9
lecture):

  > Your lab images a dish of migrating cells. Every few minutes the
  > segmentation pipeline emits a *frame*: a handful of (x, y) detections,
  > one per cell -- but the detector does NOT know which dot is which.
  > "How do we chain the unlabelled dots, frame to frame, into the
  > trajectory of each individual cell?"

That chaining step is **data association**, and it is the whole job of this
section:

    detect per frame  ->  *associate* across frames (an assignment problem)
    ->  chain the per-frame matches into trajectories  ->  watch identities
    break at a crossing  ->  fix it by predicting motion before matching.

Data
----
`data/detections.csv`: a small synthetic cohort of moving cells (columns
`frame, det_id, x, y, track_id`). `track_id` is the hidden ground truth -- used
only to *score* a tracker, never to match.

What to implement
-----------------
Six auto-graded functions, marked with ``# TODO`` below:
`cost_matrix`, `greedy_nn`, `hungarian`, `link_tracks`, `count_id_switches`,
`predict_then_match`. The pre-given helpers (loading, `link_tracks_motion`) are
shipped to you intact.

Run the tests with:

    pip install -r requirements.txt
    pytest -v tests/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ===========================================================================
# Pre-given helpers (use as-is)
# ===========================================================================


def load_detections(path) -> pd.DataFrame:
    """Load the detections CSV (columns: frame, det_id, x, y, track_id)."""
    return pd.read_csv(path).sort_values(["frame", "det_id"]).reset_index(drop=True)


def frame_points(df: pd.DataFrame, frame: int):
    """Return detections of one frame, in det_id order.

    Returns
    -------
    pos : (n, 2) float array of (x, y) detection positions.
    gt  : (n,)   int array of ground-truth track_id (for SCORING only --
          a tracker must never look at this when matching).
    """
    sub = df[df["frame"] == frame].sort_values("det_id")
    pos = sub[["x", "y"]].to_numpy(dtype=float)
    gt = sub["track_id"].to_numpy(dtype=int)
    return pos, gt


def frames_as_list(df: pd.DataFrame):
    """Convenience: per-frame (pos, gt) lists over all frames in order."""
    frames = sorted(df["frame"].unique())
    pos_list, gt_list = [], []
    for f in frames:
        pos, gt = frame_points(df, f)
        pos_list.append(pos); gt_list.append(gt)
    return pos_list, gt_list


# ===========================================================================
# Part 1 -- one-step association
# ===========================================================================


def cost_matrix(src, dst, metric: str = "euclidean") -> np.ndarray:
    """Pairwise cost between source points (rows) and destination points (cols).

    `src` is (n, 2) track positions at frame t, `dst` is (m, 2) detection
    positions at frame t+1.  Entry [i, j] is the distance from track i to
    detection j: Euclidean for ``metric="euclidean"``, sum of absolute
    differences for ``metric="l1"``.
    """
    src = np.asarray(src, dtype=float); dst = np.asarray(dst, dtype=float)
    # TODO (cost_matrix): build the (n, m) distance matrix.
    # Hint: broadcast src[:, None, :] - dst[None, :, :], then reduce the last
    # axis (np.sqrt of the squared sum for "euclidean", abs-sum for "l1").
    raise NotImplementedError


def greedy_nn(cost):
    """Greedy nearest-neighbour assignment, row by row.

    Process tracks (rows) top to bottom; each takes its nearest detection
    (column) that is still free.  Fast and intuitive, but order-dependent
    and not cost-optimal.

    Returns
    -------
    assignment : dict {row_index -> col_index}
    total      : float, summed cost of the chosen pairs
    collided   : bool, True if any row could NOT take its own argmin because
                 that column was already taken (the tell-tale of greedy failure)
    """
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    # TODO (greedy_nn): for each row, pick its lowest-cost column that is not
    # already used; set `collided=True` if a row is forced off its own argmin.
    # Return (assignment dict, total float, collided bool).
    raise NotImplementedError


def hungarian(cost):
    """Globally cost-optimal one-to-one assignment (Hungarian algorithm).

    Use ``scipy.optimize.linear_sum_assignment`` -- it minimises the TOTAL
    cost over all valid one-to-one matchings, with a guarantee, in O(n^3).

    Returns
    -------
    assignment : dict {row_index -> col_index}
    total      : float, the minimum achievable total cost
    """
    cost = np.asarray(cost, dtype=float)
    # TODO (hungarian): call linear_sum_assignment(cost), package the row/col
    # pairs into a dict and sum the chosen entries for the total.
    raise NotImplementedError


# ===========================================================================
# Part 2 -- chain one-step matches into tracks
# ===========================================================================


def link_tracks(frame_pos, matcher):
    """Chain per-frame matches into trajectories (position-only).

    `frame_pos` is a list of (n_t, 2) detection-position arrays, one per
    frame, in det_id order.  `matcher(cost)` returns a dict {prev_idx ->
    cur_idx} (pass e.g. ``lambda C: greedy_nn(C)[0]`` or
    ``lambda C: hungarian(C)[0]``).

    Frame 0's detections seed the track labels (label = det index).  For
    every later frame, match the previous frame's detections to the current
    ones and carry each matched track's label forward.

    Returns
    -------
    labels : list of (n_t,) int arrays -- the track label assigned to each
             detection (in det_id order), one array per frame.
    """
    # TODO (link_tracks): seed labels at frame 0, then for each later frame
    # build the cost matrix between consecutive frames, call `matcher`, and
    # carry the previous label of each matched track onto the current detection.
    raise NotImplementedError


def count_id_switches(pred_labels, gt_labels) -> int:
    """Count identity switches of a linking against ground truth.

    For every ground-truth track, follow it across consecutive frames and
    count how often the predicted label assigned to it *changes*.  Summed
    over all tracks, this is the standard MOT "ID switch" count: 0 means
    every identity was held throughout.

    `pred_labels` and `gt_labels` are both lists of (n_t,) int arrays in the
    same det_id order (so `pred_labels[t][k]` and `gt_labels[t][k]` describe
    the same detection).
    """
    # TODO (count_id_switches): for each consecutive frame pair, map each
    # ground-truth id to its predicted label; count an id whose predicted
    # label differs from the previous frame's.
    raise NotImplementedError


# ===========================================================================
# Part 3 -- predict, then match (the crossing fix)
# ===========================================================================


def predict_then_match(prev, cur, det):
    """Match using a constant-velocity prediction instead of raw position.

    `prev` (n, 2) and `cur` (n, 2) are the SAME tracks' positions at frames
    t-1 and t (row i is the same identity in both).  `det` (m, 2) are the new
    detections at frame t+1.  Estimate each track's velocity (cur - prev),
    predict its next position (cur + velocity), and Hungarian-match the
    PREDICTIONS to the detections.

    Returns
    -------
    assignment : dict {track_index -> det_index}
    """
    prev = np.asarray(prev, float); cur = np.asarray(cur, float)
    # TODO (predict_then_match): predict pred = cur + (cur - prev), then return
    # the Hungarian assignment of cost_matrix(pred, det).
    raise NotImplementedError


def link_tracks_motion(frame_pos):
    """Pre-given: chain with predict-then-match (uses your `predict_then_match`).

    Frame 1 has no velocity yet, so it bootstraps with a plain Hungarian
    position match; from frame 2 on it predicts each track's next position
    from its previous two and matches the predictions to the detections.
    (Assumes a constant set of tracks -- no births/deaths.)
    """
    labels = [np.arange(len(frame_pos[0]), dtype=int)]
    # bootstrap frame 1 with position-only Hungarian
    a0 = hungarian(cost_matrix(frame_pos[0], frame_pos[1]))[0]
    lab1 = np.full(len(frame_pos[1]), -1, dtype=int)
    for pi, ci in a0.items():
        lab1[ci] = labels[0][pi]
    labels.append(lab1)

    for t in range(2, len(frame_pos)):
        prev_by_label = {int(l): frame_pos[t - 2][k] for k, l in enumerate(labels[t - 2])}
        cur_by_label = {int(l): frame_pos[t - 1][k] for k, l in enumerate(labels[t - 1])}
        common = [l for l in map(int, labels[t - 1]) if l in prev_by_label]
        prev = np.array([prev_by_label[l] for l in common])
        cur = np.array([cur_by_label[l] for l in common])
        a = predict_then_match(prev, cur, frame_pos[t])   # common_idx -> det_idx
        lab = np.full(len(frame_pos[t]), -1, dtype=int)
        for ki, ci in a.items():
            lab[ci] = common[ki]
        labels.append(lab)
    return labels
