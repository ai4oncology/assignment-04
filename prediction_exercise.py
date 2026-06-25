"""Section B: predictive modelling -- tiny sequence models on the PBC cohort.

Paired with the predictive-modelling beat of the assignment narrative:

  > The hepatology group has the PBC (primary biliary cirrhosis) cohort:
  > 312 patients, each followed over years with repeated liver-lab panels.
  > "Can a sequence model read a patient's lab *trajectory* and flag who
  > will die? And is a Transformer worth it over a plain RNN here?"

You build two **deliberately tiny** sequence classifiers and compare them
against a simple, strong baseline:

  1. **Logistic regression** on per-patient summary statistics (mean, last
     value, slope of each lab). It throws away fine temporal shape, but it
     is the bar any sequence model must clear.
  2. **A tiny RNN** (one GRU layer, small hidden size). A learned recurrent
     state summarising the visit sequence.
  3. **A tiny Transformer** (one encoder layer, small model dim). Self-attention
     over the visit sequence, mean-pooled.

The point of the exercise is NOT to win a leaderboard. On ~300 patients you
should find that the RNN and Transformer roughly **tie each other**, and that
neither reliably **beats the logistic baseline**. Small clinical cohorts reward
small models and strong baselines; a Transformer's capacity is wasted (and
overfits) when there are only a few hundred labelled sequences.

Data
----
`data/pbcseq.csv` (the pbcseq cohort, R `survival` package). Repeated rows per
`id`, with `day` (days since enrolment) and `status` (0 censored, 1 transplant,
2 death). We predict **death** (status == 2 ever) from the lab trajectory.

What to implement
-----------------
Eight auto-graded functions (see the stubs below). Keep the two models tiny:
the tests cap each at under 3000 parameters, so use small `hidden` / `d_model`
(the defaults are fine). On this cohort a 393-parameter RNN matches a much
larger one, so bigger is not better here.

Run the tests with:

    pip install -r requirements.txt
    pytest -v tests/
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The six liver-lab channels used as the per-visit feature vector.
LABS = ["bili", "albumin", "alk.phos", "ast", "platelet", "protime"]
# Right-skewed labs that should be log-transformed before standardizing.
LOG_LABS = {"bili", "alk.phos", "ast", "protime"}
N_FEATURES = len(LABS)


# ---------------------------------------------------------------------------
# Data: variable-length visit sequences -> padded tensor + mask + label
# ---------------------------------------------------------------------------
def load_sequences(csv_path, max_len: int = 16):
    """Build padded per-patient lab sequences and the death label.

    Steps:
      1. Read `csv_path`; drop rows with any missing value in `LABS`.
      2. Log-transform the `LOG_LABS` columns (use `np.log` on values clipped
         to be >= 1e-3), then **z-score every lab** across all visit rows
         (mean 0, std 1 per column).
      3. Sort by (`id`, `day`). For each patient, take their first `max_len`
         visits as a sequence of `LABS` vectors.
      4. Pad every sequence up to `max_len` time steps with zeros, and build a
         boolean mask (True where a real visit sits, False on padding).
      5. Label `y` is 1 if the patient ever has `status == 2` (death), else 0.

    Returns
    -------
    X : float32 array, shape (n_patients, max_len, N_FEATURES)
        Zero-padded standardized lab sequences.
    mask : bool array, shape (n_patients, max_len)
        True at real visits, False at padding.
    y : int array, shape (n_patients,)
        Death label (1 = died).
    """
    # TODO: implement steps 1-5 and return (X, mask, y).
    raise NotImplementedError


def summary_features(X, mask):
    """Collapse each padded sequence to baseline summary statistics.

    For every patient, over their **real** time steps only (use the mask),
    compute three statistics per lab and concatenate them:

        [ mean over visits , last visit value , slope over visits ]

    The slope is the least-squares slope of the lab against the visit index
    0, 1, 2, ... (use `np.polyfit(t, values, 1)[0]`; if a patient has a single
    visit, use slope 0). The result has 3 * N_FEATURES columns.

    Returns
    -------
    F : float32 array, shape (n_patients, 3 * N_FEATURES)
    """
    # TODO: loop over patients, slice X by the mask, stack [mean, last, slope].
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Models: two tiny torch sequence classifiers
# ---------------------------------------------------------------------------
def count_parameters(model) -> int:
    """Total number of trainable parameters in a torch module."""
    # TODO: sum p.numel() over model.parameters() where p.requires_grad.
    raise NotImplementedError


def make_rnn(input_dim: int, hidden: int = 8):
    """A tiny GRU classifier.

    Return a `torch.nn.Module` whose `forward(x, mask)` takes:
        x    : float tensor (batch, time, input_dim)
        mask : bool tensor  (batch, time), True at real steps
    and returns a 1-D tensor of length `batch` of **logits** (pre-sigmoid).

    Architecture: a single `nn.GRU(input_dim, hidden, batch_first=True)`,
    then read the hidden state at each patient's **last real time step**
    (use the mask to find it), then a `nn.Linear(hidden, 1)`.

    Keep `hidden` small (default 8, about 400 parameters) so the model stays tiny.
    """
    # TODO: define and return the module (subclass nn.Module).
    raise NotImplementedError


def make_transformer(input_dim: int, d_model: int = 8, nhead: int = 2,
                     dim_feedforward: int = 16):
    """A tiny Transformer-encoder classifier.

    Return a `torch.nn.Module` with the same `forward(x, mask)` contract as
    `make_rnn` (returns `batch` logits).

    Architecture:
        nn.Linear(input_dim, d_model)                       # embed each visit
        one nn.TransformerEncoderLayer(d_model, nhead,
            dim_feedforward=dim_feedforward, batch_first=True)
        masked **mean-pool** over real time steps (use the mask)
        nn.Linear(d_model, 1)

    Pass the padding mask to the encoder via `src_key_padding_mask=~mask`
    (the layer expects True where positions should be ignored). Keep
    `d_model` small (default 8, about 700 parameters).
    """
    # TODO: define and return the module (subclass nn.Module).
    raise NotImplementedError


def train_model(model, X, mask, y, epochs: int = 60, lr: float = 5e-3,
                weight_decay: float = 1e-3, seed: int = 42):
    """Train a sequence classifier with full-batch Adam.

    - Seed torch with `seed` (call `torch.manual_seed(seed)`) for reproducibility.
    - Convert `X`, `mask`, `y` to tensors (y as float32).
    - Loss: `nn.BCEWithLogitsLoss` with `pos_weight` = (#neg / #pos) in `y`,
      to handle class imbalance.
    - Optimizer: `torch.optim.Adam(model.parameters(), lr=lr,
      weight_decay=weight_decay)`.
    - For `epochs` steps: zero grads, forward `model(X, mask)`, BCE loss,
      backward, step (full batch; no minibatching needed at this size).

    Returns the trained `model`.
    """
    # TODO: implement the full-batch training loop and return the model.
    raise NotImplementedError


def predict_proba(model, X, mask):
    """Return death probabilities for each patient (numpy array, length n).

    Put the model in eval mode, run a no-grad forward pass, apply a sigmoid
    to the logits, and return a 1-D numpy array.
    """
    # TODO: model.eval(); with torch.no_grad(): sigmoid(model(X, mask)) -> numpy.
    raise NotImplementedError


def evaluate_auroc(y_true, y_proba) -> float:
    """AUROC of predicted probabilities against binary labels.

    Thin wrapper around `sklearn.metrics.roc_auc_score`, returned as a float.
    """
    # TODO: return float(roc_auc_score(y_true, y_proba)).
    raise NotImplementedError
