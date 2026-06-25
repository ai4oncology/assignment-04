"""Reference solution -- Tracking part of the Longitudinal section (Assignment 04).

Multi-object tracking on a small synthetic cohort of moving cells
(`data/detections.csv`).  The narrative mirrors the Week-9 lecture:

  detect per frame  ->  *associate* across frames (an assignment problem)
  ->  chain the per-frame matches into trajectories  ->  watch identities
  break at a crossing  ->  fix it by predicting motion before matching.

This file is the **solved reference**.  Every region a student must write is
wrapped in a marker pair::

    # >>> STUDENT TODO (<name>): <one-line instruction>
    ...reference solution...
    # <<< END STUDENT TODO

To produce the student-facing `tracking_exercise.py`, replace the body of each
marked block with ``raise NotImplementedError``.  Everything outside the markers
(imports, docstrings, the pre-given helpers) is shipped to students unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ===========================================================================
# Pre-given helpers (shipped to students as-is)
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
    # >>> STUDENT TODO (cost_matrix): build the (n, m) distance matrix
    diff = src[:, None, :] - dst[None, :, :]
    if metric == "l1":
        return np.abs(diff).sum(-1)
    return np.sqrt((diff ** 2).sum(-1))
    # <<< END STUDENT TODO


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
    # >>> STUDENT TODO (greedy_nn): row-wise argmin, forbidding reuse
    used: set[int] = set()
    assignment: dict[int, int] = {}
    collided = False
    for i in range(n):
        order = np.argsort(cost[i], kind="stable")
        pick = next(int(j) for j in order if int(j) not in used)
        if pick != int(order[0]):
            collided = True
        assignment[i] = pick
        used.add(pick)
    total = float(sum(cost[i, assignment[i]] for i in assignment))
    return assignment, total, collided
    # <<< END STUDENT TODO


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
    # >>> STUDENT TODO (hungarian): call linear_sum_assignment, package result
    rows, cols = linear_sum_assignment(cost)
    assignment = {int(r): int(c) for r, c in zip(rows, cols)}
    total = float(cost[rows, cols].sum())
    return assignment, total
    # <<< END STUDENT TODO


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
    # >>> STUDENT TODO (link_tracks): propagate labels frame to frame
    labels = [np.arange(len(frame_pos[0]), dtype=int)]
    for t in range(1, len(frame_pos)):
        cost = cost_matrix(frame_pos[t - 1], frame_pos[t])
        assign = matcher(cost)                       # prev_idx -> cur_idx
        lab = np.full(len(frame_pos[t]), -1, dtype=int)
        for prev_idx, cur_idx in assign.items():
            lab[cur_idx] = labels[t - 1][prev_idx]
        labels.append(lab)
    return labels
    # <<< END STUDENT TODO


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
    # >>> STUDENT TODO (count_id_switches): compare label continuity per gt track
    switches = 0
    n_frames = len(gt_labels)
    for t in range(1, n_frames):
        prev_map = {int(g): int(p) for g, p in zip(gt_labels[t - 1], pred_labels[t - 1])}
        cur_map = {int(g): int(p) for g, p in zip(gt_labels[t], pred_labels[t])}
        for g, p in cur_map.items():
            if g in prev_map and prev_map[g] != p:
                switches += 1
    return switches
    # <<< END STUDENT TODO


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
    # >>> STUDENT TODO (predict_then_match): predict next position, then match
    velocity = cur - prev
    pred = cur + velocity
    assignment, _ = hungarian(cost_matrix(pred, det))
    return assignment
    # <<< END STUDENT TODO


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


# ===========================================================================
# Self-check (instructor): run `python tracking_solution.py`
# ===========================================================================

if __name__ == "__main__":
    from pathlib import Path

    df = load_detections(Path(__file__).with_name("data") / "detections.csv")
    pos_list, gt_list = frames_as_list(df)

    # align each frame by ground-truth identity so per-step assignments can be
    # compared against the identity {0:0, 1:1, 2:2} (SCORING only -- a tracker
    # never does this).
    def by_gt(t):
        order = np.argsort(gt_list[t]); return pos_list[t][order]

    # --- the two engineered steps, on the real frames ---
    print("== clean step 0->1 ==")
    c01 = cost_matrix(by_gt(0), by_gt(1))
    print(" greedy:", greedy_nn(c01)[:2], "| hungarian:", hungarian(c01))
    print("== drift step 1->2 (dish bumped) ==")
    c12 = cost_matrix(by_gt(1), by_gt(2))
    print(" greedy:", greedy_nn(c12)[:2], "| hungarian:", hungarian(c12),
          "  <- greedy strays + pays more; Hungarian recovers identity")
    print("== crossing step 3->4 ==")
    c34 = cost_matrix(by_gt(3), by_gt(4))
    print(" greedy:", greedy_nn(c34)[:2], "| hungarian:", hungarian(c34),
          "  <- BOTH swap; predicted:", predict_then_match(by_gt(2), by_gt(3), by_gt(4)))

    # --- chaining on the real detections: greedy vs hungarian vs motion ---
    greedy_labels = link_tracks(pos_list, lambda C: greedy_nn(C)[0])
    hung_labels = link_tracks(pos_list, lambda C: hungarian(C)[0])
    motion_labels = link_tracks_motion(pos_list)
    print("ID switches  greedy :", count_id_switches(greedy_labels, gt_list),
          "(drift + crossing)")
    print("ID switches  hungar.:", count_id_switches(hung_labels, gt_list),
          "(crossing only -- global optimisation rescues the drift)")
    print("ID switches  motion :", count_id_switches(motion_labels, gt_list),
          "(drift only -- linear motion cannot anticipate a discontinuity)")

    # --- pen-and-paper reference matrix [[3,7,4],[1,7,7],[4,3,2]] ---
    pp = np.array([[3, 7, 4], [1, 7, 7], [4, 3, 2]])
    print("pen&paper greedy:", greedy_nn(pp)[:2], "| hungarian:", hungarian(pp))
