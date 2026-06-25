"""Generate the synthetic cell-detection dataset for the tracking exercise.

Run once to (re)create ``detections.csv`` next to this file::

    python make_detections.py

Three cells move right over eight frames.  Two events are baked in so the
tracker comparison has something to chew on:

* **A dish bump (drift).**  Between frame 1 and frame 2 the microscope stage
  is knocked, so from frame 2 on *every* detection is shifted by ``DRIFT``
  (a one-off global translation that then persists).  At that step the
  apparent jump is large enough that **greedy** nearest-neighbour strands a
  track and pays a higher cost, while the **Hungarian** algorithm -- a rigid
  translation preserves the relative geometry -- still recovers the right
  identities.  Constant-velocity **motion prediction** *also* trips here: it
  cannot anticipate a discontinuity.
* **A crossing.**  Track 0 drifts up, track 1 drifts down, and their paths
  cross *between* frames 3 and 4.  There **both** greedy and Hungarian swap
  identities (after a crossing the swap *is* the minimum-distance matching),
  and only motion prediction carries the identities through.

So no single cost wins everywhere: Hungarian fixes the bump, motion fixes the
crossing.  Net identity switches: greedy 4, Hungarian 2, motion 2.

* **track 2** stays well above the others and is never involved.

Each detection is jittered by small Gaussian noise, and the detections are
**shuffled within every frame** so that the per-frame ``det_id`` order carries
no identity information -- recovering identity is the whole point of the
exercise.  The ground-truth ``track_id`` is stored only so the exercise can
*score* a tracker (it must not be used for matching).

CSV columns: ``frame, det_id, x, y, track_id``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_FRAMES = 8
NOISE_STD = 0.15

# The dish bump: from DRIFT_FROM_FRAME onward, every detection is translated
# by DRIFT (and stays translated -- the stage does not move back).
DRIFT_FROM_FRAME = 2
DRIFT = (0.0, -7.0)

# Ground-truth linear motion: start position + per-frame velocity.
TRACKS = {
    0: {"start": (0.0, 2.0), "vel": (2.0, 1.5)},    # up-right
    1: {"start": (0.0, 13.0), "vel": (2.0, -1.5)},  # down-right -> crosses track 0
    2: {"start": (0.0, 19.0), "vel": (2.0, 0.0)},   # separate, flat, well above
}


def make_detections(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for frame in range(N_FRAMES):
        sx, sy = DRIFT if frame >= DRIFT_FROM_FRAME else (0.0, 0.0)
        true = {
            tid: (
                t["start"][0] + frame * t["vel"][0] + sx,
                t["start"][1] + frame * t["vel"][1] + sy,
            )
            for tid, t in TRACKS.items()
        }
        track_ids = list(true.keys())
        rng.shuffle(track_ids)  # det_id order != track_id
        for det_id, tid in enumerate(track_ids):
            x = true[tid][0] + rng.normal(0.0, NOISE_STD)
            y = true[tid][1] + rng.normal(0.0, NOISE_STD)
            rows.append(
                {"frame": frame, "det_id": det_id,
                 "x": round(float(x), 3), "y": round(float(y), 3),
                 "track_id": tid}
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_detections()
    out = Path(__file__).with_name("detections.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {df.frame.nunique()} frames, "
          f"{df.track_id.nunique()} tracks)")
    print(df.to_string(index=False))
