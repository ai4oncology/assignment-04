"""Part 1 - B: predictive modelling -- leak-free landmark survival on the PBC cohort.

Paired with the predictive-modelling beat of the assignment narrative:

  > The hepatology group has the PBC (primary biliary cirrhosis) cohort:
  > 312 patients followed over years with repeated liver-lab panels.
  > "From a patient's first couple of years of labs, can we flag who will die
  > within five years? And is a Transformer worth it over a plain RNN?"

The hard part is doing this **without leaking the future**. Each patient is
described only by visits in the first `landmark_years`, and the label is death
within `horizon_years`, so no value measured after the prediction window ever
reaches any model. You then compare three models:

  1. **Logistic regression** on per-patient summary statistics (mean, last value,
     slope of each lab): a strong, cheap baseline.
  2. **A tiny RNN** (one GRU layer, small hidden size).
  3. **A tiny Transformer** (one encoder layer, small model dim), mean-pooled.

On this honest landmark setup the three are close (AUROC ~0.88-0.92), and with
only ~55 deaths the differences are within noise. So the lesson is the evaluation
**discipline** (a fixed prediction window, a future horizon, and dropping patients
whose outcome is unknown), not a leaderboard. The naive alternative ("did the
patient ever die?") scores higher but cheats: the last visit sits right before
death, and follow-up length itself encodes the outcome.

Data
----
`data/pbcseq.csv` (pbcseq, R `survival` package). Repeated rows per `id`, with
`day` (days since enrolment), `futime` (days of follow-up), and `status`
(0 censored, 1 transplant, 2 death). `load_sequences` turns this into the
leak-free landmark dataset described below.

What to implement
-----------------
The auto-graded functions below: the landmark dataset + the three models +
their training / evaluation, then next-visit forecasting, plus the leak-free
cross-validation loops (`cross_val_auroc`, `groupkfold_mae`) and a `lag1_autocorr`
diagnostic. Keep the two models tiny:
the tests cap each at under 500 parameters, so use small `hidden` / `d_model`
(the defaults are fine). On this cohort a 150-parameter RNN matches a much
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
def load_sequences(csv_path, landmark_years: float = 2.0,
                   horizon_years: float = 5.0, max_len: int = 8):
    """Build a leakage-free landmark dataset: early visits in, future label out.

    The rule that prevents leakage: a patient is described **only** by the visits
    in their first `landmark_years`, and the label is whether they die within
    `horizon_years`. Nothing measured after the landmark is ever used, by any model.

    Steps:
      1. Read `csv_path`; drop rows with any missing value in `LABS`. Compute each
         visit's time `yr = day / 365.25` and follow-up `futy = futime / 365.25`.
      2. Log-transform the `LOG_LABS` columns (`np.log` on values clipped >= 1e-3).
      3. Sort by (`id`, `day`). For each patient, keep the visits with
         `yr <= landmark_years` (the prediction window). Keep the patient only if:
            - they are **alive at the landmark** (`futy > landmark_years`), and
            - their status at the horizon is **known**: either they die by the
              horizon (`status == 2 and futy <= horizon_years`, label 1) or they
              are still followed past it (`futy > horizon_years`, label 0).
         Drop anyone censored in `(landmark_years, horizon_years]` (unknown label).
      4. **Z-score every lab using only the in-window visits you kept** (so no
         out-of-window value ever touches the features), then pad each sequence to
         `max_len` steps with zeros and build a boolean mask.

    Returns
    -------
    X : float32 array, shape (n_patients, max_len, N_FEATURES)  -- padded, standardized.
    mask : bool array, shape (n_patients, max_len)  -- True at real visits.
    y : int array, shape (n_patients,)  -- 1 if dead within `horizon_years`.
    """
    # TODO: implement the landmark rule above and return (X, mask, y).
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


def make_rnn(input_dim: int, hidden: int = 4):
    """A tiny GRU classifier.

    Return a `torch.nn.Module` whose `forward(x, mask)` takes:
        x    : float tensor (batch, time, input_dim)
        mask : bool tensor  (batch, time), True at real steps
    and returns a 1-D tensor of length `batch` of **logits** (pre-sigmoid).

    Architecture: a single `nn.GRU(input_dim, hidden, batch_first=True)`,
    then read the hidden state at each patient's **last real time step**
    (use the mask to find it), then a `nn.Linear(hidden, 1)`.

    Keep `hidden` small (default 4, about 150 parameters) so the model stays tiny.
    """
    # TODO: define and return the module (subclass nn.Module).
    raise NotImplementedError


def make_transformer(input_dim: int, d_model: int = 4, nhead: int = 2,
                     dim_feedforward: int = 8):
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
    `d_model` small (default 4, about 200 parameters).
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


# ---------------------------------------------------------------------------
# Part 1-B (continued): forecasting the next visit's labs (regression)
# ---------------------------------------------------------------------------
def load_forecasting(csv_path, max_len: int = 16):
    """Build a next-visit *forecasting* dataset (predict values, not an outcome).

    Load / log-transform / z-score the labs as in `load_sequences`, but here each
    patient yields many examples and the target is the next visit's lab *vector*,
    not a death label. For each patient with `k` visits and each `j = 1..k-1`: the
    input is the visits so far (the last `max_len`), and the target is the lab
    vector at visit `j+1`. Return `groups` (patient id) for GroupKFold.

    Returns
    -------
    X : float32 array, shape (n_examples, max_len, N_FEATURES)  -- padded history.
    mask : bool array, shape (n_examples, max_len).
    Y : float32 array, shape (n_examples, N_FEATURES)  -- the next visit's labs.
    groups : array, shape (n_examples,)  -- patient id (use GroupKFold).
    """
    # TODO: build (history -> next-visit-labs) examples; return (X, mask, Y, groups).
    raise NotImplementedError


def persistence_forecast(X, mask):
    """Baseline forecaster: predict the next visit equals the **last observed** visit.

    For each example, return the lab vector at its last real time step (use the
    mask to find it). Shape (n_examples, N_FEATURES). This "tomorrow = today"
    baseline is strong because the labs are highly autocorrelated.
    """
    # TODO: pick each row's last real time step via the mask and return those vectors.
    raise NotImplementedError


def forecast_mae(Y_true, Y_pred) -> float:
    """Mean absolute error over all labs and examples (standardized units)."""
    # TODO: return float(np.mean(np.abs(Y_true - Y_pred))).
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Leak-free cross-validation + an autocorrelation diagnostic
# ---------------------------------------------------------------------------
def lag1_autocorr(frame, col) -> float:
    """Lag-1 autocorrelation of a lab across consecutive visits of the same patient.

    For each patient, pair every visit's value with that patient's *next* visit's
    value (sorted by `day`), then return the Pearson correlation over all such
    (prev, cur) pairs, as a float. Log-transform the right-skewed labs first (the
    ones in `LOG_LABS`), matching `load_sequences`.

    A high value is exactly why **persistence** ("next = last") is so hard to beat:
    consecutive lab values barely move, so yesterday is an excellent guess for today.
    """
    # TODO: build per-patient (prev, cur) pairs sorted by day (use groupby().shift()),
    # drop NaNs, and return float(np.corrcoef(prev, cur)[0, 1]).
    raise NotImplementedError


def cross_val_auroc(make, X, mask, y, n_splits: int = 5, epochs: int = 60, seed: int = 42):
    """Stratified k-fold cross-validated AUROC for the landmark classifier.

    Here each patient is **one row** (one sequence, one label), so a plain
    `StratifiedKFold` already keeps a patient on a single side of the split: no
    GroupKFold needed (contrast `groupkfold_mae`, where one patient yields many
    rows). `make` is a zero-arg factory returning a *fresh* model per fold (so
    folds do not share trained weights). For each fold: train on the train rows
    with `train_model`, score the held-out rows with `evaluate_auroc` on
    `predict_proba`. Return `(mean_auroc, std_auroc)` across folds, as floats.
    """
    # TODO: loop StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y);
    # train a fresh make() per fold, collect held-out AUROC, return (mean, std).
    raise NotImplementedError


def groupkfold_mae(make, X, mask, Y, groups, n_splits: int = 5,
                   epochs: int = 40, seed: int = 42) -> float:
    """GroupKFold (keyed on patient id) cross-validated next-visit MAE.

    Unlike the landmark task, one patient produces **many** (history, next-visit)
    examples, so a random split would put the same patient in train and test and
    leak. `GroupKFold(...).split(X, Y, groups)` with `groups` = patient id keeps
    every example of a patient on one side. `make` returns a fresh forecaster
    mapping `(X, mask) -> next-visit lab vector`. Per fold: train it (Adam +
    `nn.MSELoss`, `epochs` steps, seed `seed`), then record the absolute error on
    the held-out rows. Return the mean absolute error over all examples, as a float.
    """
    # TODO: for tr, te in GroupKFold(n_splits).split(X, Y, groups): train make() on
    # the train rows, fill err[te] = |pred - Y[te]|; return float(err.mean()).
    raise NotImplementedError
