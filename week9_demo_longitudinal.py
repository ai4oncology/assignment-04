import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Week 9 demo: Longitudinal data, a cohort, not independent rows

    Companion notebook for the *longitudinal data* slides. One running example, the
    **PBC** cohort (`pbcseq`: 312 patients with repeated liver-lab panels), used to make the
    three properties that define longitudinal data concrete:

    * **Repeated measures**: each patient is a *sequence of visits*, not one row;
    * **Within-patient correlation**: two visits of the same patient are far more alike
      than two random visits, so the **i.i.d. assumption breaks** (this is exactly why
      pooling visits into a single PCA misleads; see the brain teaser);
    * **Irregular / informative sampling**: visit spacing varies and is not a grid, and
      *how long* a patient is followed is itself informative.

    Missing-value imputation is **not** covered here; that is the Week 8 (robustness)
    topic. This notebook only shows the parts that are *specific to longitudinal data*.
    Figures are written to `figure/intro/`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    BLUE = "#344A9A"; RED = "#C8323C"; GREEN = "#00A082"; GREY = "#9AA0A6"
    plt.rcParams.update({"font.size": 12, "figure.dpi": 120,
                         "axes.spines.top": False, "axes.spines.right": False})

    DATA = Path("progression_demo/pbcseq.csv")
    OUT = Path("figure/intro"); OUT.mkdir(parents=True, exist_ok=True)

    LABS = ["bili", "albumin", "alk.phos", "ast", "platelet", "protime"]
    LOGLABS = {"bili", "alk.phos", "ast", "protime"}      # skewed -> log scale
    PRETTY = {"bili": "bilirubin", "albumin": "albumin", "alk.phos": "alk. phos.",
              "ast": "AST", "platelet": "platelets", "protime": "prothr. time"}

    df = pd.read_csv(DATA)
    df["yr"] = df["day"] / 365.25
    df["lb"] = np.log(df["bili"].clip(lower=1e-3))                 # log serum bilirubin
    df["died"] = df.groupby("id")["status"].transform(lambda s: int((s == 2).any()))
    nvis = df.groupby("id").size()

    def lag1_corr(frame, col):
        """Lag-1 autocorrelation: the plain Pearson correlation between a visit and the
        same patient's *next* visit. r near 0 = independent, near 1 = nearly identical."""
        y = np.log(frame[col].clip(lower=1e-3)) if col in LOGLABS else frame[col]
        s = pd.DataFrame({"id": frame["id"], "day": frame["day"], "y": y}).dropna()
        s = s.sort_values(["id", "day"])
        pair = pd.DataFrame({"prev": s.groupby("id")["y"].shift(), "cur": s["y"]}).dropna()
        return np.corrcoef(pair["prev"], pair["cur"])[0, 1]

    print(f"{df['id'].nunique()} patients, {len(df)} visits; "
          f"visits/patient median {int(nvis.median())} (range {nvis.min()}-{nvis.max()})")
    return (
        BLUE,
        GREEN,
        GREY,
        LABS,
        OUT,
        PRETTY,
        Path,
        RED,
        df,
        lag1_corr,
        np,
        nvis,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Medical excursus: what is PBC?

    **Primary biliary cholangitis (PBC)** is a chronic **autoimmune** liver disease: the immune
    system attacks and slowly destroys the **small bile ducts** inside the liver. Bile can then
    no longer drain (**cholestasis**), so **bilirubin** and bile acids build up in the blood,
    causing **jaundice** and itching, and over years the scarring progresses to **cirrhosis**.

    That mechanism is why **serum bilirubin** climbs along a patient's trajectory and is the
    central marker tracked in the `pbcseq` cohort.
    """)
    return


@app.cell
def _(BLUE, GREY, OUT, plt):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    _YEL = "#FFE863"
    figx, (axx, axy) = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

    # left: real PBC anatomy illustration (normal vs cirrhotic liver)
    axx.axis("off")
    axx.imshow(plt.imread("figure/pbc.png"))
    axx.set_title("PBC: bile-duct damage scars the liver", fontsize=11.5)
    axx.text(0.0, 1.0, " MEDICAL EXCURSUS ", transform=axx.transAxes,
             fontsize=8.5, fontweight="bold", color="black", va="bottom", ha="left",
             bbox=dict(boxstyle="square,pad=0.25", facecolor=_YEL, edgecolor="none"), zorder=8)

    # right: mechanism chain
    axy.set_xlim(0, 5); axy.set_ylim(0, 5); axy.axis("off")
    _steps = [("Small bile ducts destroyed", GREY), ("Bile cannot drain (cholestasis)", GREY),
              ("Bilirubin builds up in the blood", BLUE),
              ("Jaundice & itch;\nscarring $\\rightarrow$ cirrhosis over years", GREY)]
    _ys = [4.3, 3.25, 2.2, 1.0]; _cx = 2.5
    for (_txt, _fc), _y in zip(_steps, _ys):
        _h = 0.62 if "\n" in _txt else 0.5
        axy.add_patch(FancyBboxPatch((_cx - 2.1, _y - _h / 2), 4.2, _h, boxstyle="round,pad=0.06",
                      facecolor=_fc, edgecolor="none", alpha=.92 if _fc != GREY else .8, zorder=3))
        axy.text(_cx, _y, _txt, ha="center", va="center", color="white",
                 fontsize=9.5, fontweight=("bold" if _fc == BLUE else "normal"), zorder=4)
    for _y0, _y1 in zip(_ys[:-1], _ys[1:]):
        axy.add_patch(FancyArrowPatch((_cx, _y0 - 0.30), (_cx, _y1 + 0.30), arrowstyle="-|>",
                      mutation_scale=14, color="#555", lw=1.6, zorder=2))
    axy.text(_cx, 0.32, "serum bilirubin is the marker tracked in pbcseq", ha="center",
             va="center", fontsize=8.8, color=BLUE, fontstyle="italic")
    axy.set_title("Why bilirubin rises along the trajectory", fontsize=11.5)

    figx.tight_layout()
    figx
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The PBC cohort at a glance

    **Primary biliary cholangitis (PBC)** is a chronic autoimmune liver disease that slowly
    scars the bile ducts. `pbcseq` is the Mayo Clinic PBC trial: **312 patients**, enrolled and
    followed for over a decade (up to ~14 years), seen at **repeated visits** (every 6--12 months) where
    a panel of **liver labs** (bilirubin, albumin, AST, platelets, ...) is measured, until
    **death**, **transplant**, or the end of the study (**censored**).

    Each line below is one patient, each dot a visit, the end marker their outcome. The three
    longitudinal traits show up at once: many visits per patient, **uneven** spacing and length,
    and shorter tracks ending in **death**.
    """)
    return


@app.cell
def _(BLUE, GREY, OUT, RED, df, plt):
    sw_order = df.groupby("id")["yr"].max().sort_values().index      # short -> long follow-up
    sw_ids = sw_order[:: max(1, len(sw_order) // 30)][:30]           # ~30 patients spanning the range

    figc, axc = plt.subplots(figsize=(9.2, 5.2))
    for _row, _pid in enumerate(sw_ids):
        _g = df[df.id == _pid].sort_values("yr")
        _t = _g["yr"].values * 12.0                       # months since enrolment
        axc.plot([0, _t.max()], [_row, _row], color=GREY, lw=0.8, alpha=.5, zorder=1)
        axc.scatter(_t, [_row] * len(_t), s=16, color=BLUE, zorder=2)
        if (_g["status"] == 2).any():
            axc.scatter(_t.max(), _row, marker="X", s=55, color=RED, zorder=3)
        else:
            axc.scatter(_t.max(), _row, marker="o", s=34, facecolors="none",
                        edgecolors=GREY, linewidths=1.3, zorder=3)
    axc.set_yticks([]); axc.set_xlabel("months since enrolment")
    axc.set_ylabel(f"patients (sample of {len(sw_ids)}, sorted by follow-up)")
    axc.set_title(f"The PBC cohort: {df['id'].nunique()} patients, repeated visits over ~{df['yr'].max() * 12:.0f} months")
    axc.scatter([], [], s=16, color=BLUE, label="visit (labs measured)")
    axc.scatter([], [], marker="X", s=55, color=RED, label="death")
    axc.scatter([], [], marker="o", s=34, facecolors="none", edgecolors=GREY, label="censored / transplant")
    axc.legend(loc="lower right", fontsize=9, frameon=False)

    figc.tight_layout()
    figc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. PBC is a cohort: repeated measures

    Each patient contributes a *sequence* of visits (median 5, up to 16), and bilirubin is
    tracked along a personal trajectory, the basic shape of longitudinal data. Patients
    who later die (red) tend to ride higher, drifting paths rather than scattered points.
    """)
    return


@app.cell
def _(BLUE, OUT, RED, df, nvis, plt):
    fig1, (ov1, ov2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ov1.hist(nvis.values, bins=range(1, 18), color=BLUE, alpha=.85, edgecolor="white")
    ov1.axvline(nvis.median(), color=RED, lw=2, ls="--")
    ov1.annotate(f"median {int(nvis.median())} visits", (nvis.median() + 0.4, ov1.get_ylim()[1] * 0.9),
                 color=RED, fontsize=10)
    ov1.set_xlabel("visits per patient"); ov1.set_ylabel("patients")
    ov1.set_title(f"{df['id'].nunique()} patients, {len(df)} visits: repeated measures")

    sample = df[df.groupby("id")["id"].transform("size") >= 4]
    for _pid, g in sample.groupby("id"):
        g = g.sort_values("yr")
        ov2.plot(g["yr"], g["lb"], "-", color=(RED if g["died"].iloc[0] else BLUE), lw=0.7, alpha=.35)
    ov2.set_xlabel("years since enrolment"); ov2.set_ylabel("log serum bilirubin")
    ov2.set_title("One line per patient (red = later death)"); ov2.set_xlim(0, 12)

    fig1.tight_layout()
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Within-patient correlation: the rows are not independent

    The defining property of longitudinal data. **Left:** the gap between two visits of the
    **same** patient (green) is tiny next to two **random** visits (grey): knowing one
    visit tells you a lot about the next. **Right:** the **autocorrelation**, the plain
    correlation between a visit and that patient's *next* visit; bilirubin is ~0.9.

    This is exactly the "repeated visits are **autocorrelated**" point from the brain teaser:
    a standard PCA over pooled visits treats these correlated rows as independent samples,
    which they are not.
    """)
    return


@app.cell
def _(BLUE, GREEN, GREY, LABS, OUT, PRETTY, df, lag1_corr, np, plt):
    rng = np.random.default_rng(0)
    sdf = df.dropna(subset=["lb"]).sort_values(["id", "day"])
    same = sdf.groupby("id")["lb"].diff().abs().dropna().values        # consecutive same-patient
    vals = sdf["lb"].values; ids = sdf["id"].values
    perm = rng.permutation(len(vals))
    cross = np.abs(vals - vals[perm])[ids != ids[perm]]                # random cross-patient pairs
    acf = {lab: lag1_corr(df, lab) for lab in LABS}                    # lag-1 autocorrelation

    fig2, (ic1, ic2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    bins = np.linspace(0, 4, 30)
    ic1.hist(same, bins=bins, density=True, color=GREEN, alpha=.8,
             label=f"same patient (med {np.median(same):.2f})")
    ic1.hist(cross, bins=bins, density=True, color=GREY, alpha=.55,
             label=f"random pair (med {np.median(cross):.2f})")
    ic1.set_xlabel(r"$|\Delta$ log-bilirubin$|$ between two visits"); ic1.set_ylabel("density")
    ic1.set_title("Two visits of the SAME patient are far more alike")
    ic1.legend(fontsize=9, frameon=False)

    order = sorted(acf, key=acf.get)
    ic2.barh([PRETTY[lab] for lab in order], [acf[lab] for lab in order], color=BLUE, alpha=.85)
    ic2.set_xlim(0, 1); ic2.set_xlabel("correlation between a visit and the next (same patient)")
    ic2.set_title("Consecutive visits are strongly correlated")
    for _i, lab in enumerate(order):
        ic2.annotate(f"{acf[lab]:.2f}", (acf[lab] + 0.02, _i), va="center", fontsize=9, color=BLUE)

    fig2.tight_layout()
    print("lag-1 autocorrelation:", {k: round(v, 2) for k, v in acf.items()})
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Irregular and informative sampling

    Longitudinal data rarely lands on a clean grid. **Left:** the gap $\Delta t$ between
    consecutive visits varies (a 6-month then yearly protocol, but spread out), so *when* a
    value appears differs across patients. **Right:** the *length* of a patient's series is
    itself informative: patients who die contribute **shorter** sequences (follow-up ends),
    so observation times are not independent of the outcome.

    This is the longitudinal-specific face of "missingness": not values to impute, but the
    sampling pattern carrying signal.
    """)
    return


@app.cell
def _(BLUE, RED, df, np, nvis, plt):
    dts = df.sort_values(["id", "day"]).groupby("id")["yr"].diff().dropna().values
    died = df.groupby("id")["died"].first()

    fig3, (sp1, sp2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    sp1.hist(dts, bins=np.linspace(0, 3, 40), color=BLUE, alpha=.85, edgecolor="white")
    sp1.set_xlabel(r"gap between consecutive visits, $\Delta t$ (years)"); sp1.set_ylabel("visit pairs")
    sp1.set_title(f"Spacing varies (median {np.median(dts):.2f} yr, not a grid)")

    parts = [nvis[died == 0].values, nvis[died == 1].values]
    bp = sp2.boxplot(parts, tick_labels=["survived /\ncensored", "died"], patch_artist=True, widths=.6)
    for patch, pcol in zip(bp["boxes"], [BLUE, RED]):
        patch.set_facecolor(pcol); patch.set_alpha(.7)
    for med in bp["medians"]:
        med.set_color("black")
    sp2.set_ylabel("visits per patient")
    sp2.set_title("Series length is itself informative")

    fig3.tight_layout()
    print(f"visits: died {nvis[died == 1].mean():.1f} vs survived/censored {nvis[died == 0].mean():.1f}")
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aside: a regular-grid series, where a 1D convolution fits

    PBC visits are irregular, so a model that expects a clean grid does not apply
    directly. ICU monitoring is the opposite extreme: the **MIMIC-IV demo** streams a few
    bedside vitals on a near-continuous clock. Resampled to a **1-hour grid**, one stay
    becomes a $(\text{time} \times \text{channels})$ array, exactly the shape a **1D
    convolution** reads.

    A 1D conv is a small **kernel** (here width 5 hours) that **slides along the time
    axis** with **shared weights**, so it responds to a *local temporal shape* wherever it
    occurs. The figure below shows three things on MIMIC stay 33281088: the four vitals on
    the grid, two **fixed, interpretable** kernels on heart rate (a moving average that
    reads the local level, a difference kernel that reads the local trend), and a
    **multivariate** kernel that mixes all four channels into feature maps.

    Caveat (and a callback to property 2): a *trained* 1D-CNN forecaster on a single stay
    does **not** beat a trivial "predict the last value" baseline here, because vitals are
    strongly autocorrelated and one stay is little data. Convolution is the right *operator*
    for grid time series; it earns its keep with many series and a real label, which is the
    Week 10 (predictive modelling) topic.
    """)
    return


@app.cell
def _(BLUE, GREEN, Path, RED, np, pd):
    import torch
    import torch.nn as nn

    MIMIC = Path("progression_demo/mimic_demo/chartevents.csv.gz")
    PURPLE = "#7C4D9F"
    STAY = 33281088            # a demo stay with all four vitals densely sampled
    HRS = 48                   # first two days
    # (label, itemids, plausible (lo, hi) range, color)
    VITALS = [("heart rate", (220045,), (20, 220), RED),
              ("blood pressure", (220052, 220181), (30, 180), BLUE),
              ("SpO$_2$", (220277,), (60, 100), GREEN),
              ("resp. rate", (220210,), (4, 60), PURPLE)]

    ce = pd.read_csv(MIMIC, usecols=["stay_id", "itemid", "charttime", "valuenum"])
    ce = ce[ce.stay_id == STAY].dropna(subset=["valuenum"]).copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["h"] = (ce["charttime"] - ce["charttime"].min()).dt.total_seconds() / 3600.0

    grid = np.arange(0, HRS + 1, 1.0)                 # 1-hour grid
    _cols = []
    for _name, _ids, (_lo, _hi), _c in VITALS:
        _d = ce[ce.itemid.isin(_ids) & ce.valuenum.between(_lo, _hi) & (ce.h <= HRS)]
        _s = pd.Series(_d.valuenum.values, index=np.floor(_d["h"].values)).groupby(level=0).mean()
        _s = _s[_s.index.isin(grid)]
        _v = pd.Series(index=grid, dtype=float)
        _v.loc[_s.index.values] = _s.values
        _cols.append(_v.ffill().bfill().values)
    X = np.array(_cols).T                             # (T, 4) multivariate series
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)          # per-channel z-score
    print(f"MIMIC stay {STAY}: {X.shape[0]} hourly steps x {X.shape[1]} channels")
    return HRS, VITALS, Xz, grid, nn, torch


@app.cell
def _(BLUE, HRS, RED, VITALS, Xz, grid, nn, plt, torch):
    K = 5                                             # kernel width (hours)
    _hr = torch.tensor(Xz[:, 0][None, None], dtype=torch.float32)   # heart-rate channel

    smooth = nn.Conv1d(1, 1, K, padding=K // 2, bias=False)
    smooth.weight.data[:] = torch.ones(1, 1, K) / K                 # moving average: local level
    slope = nn.Conv1d(1, 1, K, padding=K // 2, bias=False)
    slope.weight.data[:] = torch.tensor([-1., -1, 0, 1, 1]).view(1, 1, K) / 2   # local trend

    mix = nn.Conv1d(4, 3, K, padding=K // 2)          # one kernel reading all 4 channels
    torch.manual_seed(0)
    with torch.no_grad():
        f_smooth = smooth(_hr)[0, 0].numpy()
        f_slope = slope(_hr)[0, 0].numpy()
        f_mix = torch.relu(mix(torch.tensor(Xz.T[None], dtype=torch.float32)))[0].numpy()

    figk, axk = plt.subplots(3, 1, figsize=(8.6, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1]})
    for _i, (_name, _ids, _rng, _c) in enumerate(VITALS):
        axk[0].plot(grid, Xz[:, _i], color=_c, lw=1.3, label=_name)
    axk[0].set_ylim(top=axk[0].get_ylim()[1] + 1.4)
    axk[0].set_title(f"MIMIC-IV ICU stay: four vitals on a 1-hour grid, first {HRS} h (z-scored)", fontsize=10)
    axk[0].set_ylabel("z-score"); axk[0].legend(ncol=4, fontsize=8, frameon=False, loc="upper center")

    axk[1].plot(grid, Xz[:, 0], color="#9AA0A6", lw=1.0, label="heart rate (input)")
    axk[1].plot(grid, f_smooth, color=RED, lw=1.8, label="moving-average kernel (local level)")
    axk[1].plot(grid, f_slope, color=BLUE, lw=1.8, label="difference kernel (local trend)")
    axk[1].axhline(0, color="k", lw=.5, alpha=.3)
    axk[1].set_title("Two fixed width-5 kernels slide along time over one channel", fontsize=10)
    axk[1].set_ylabel("response"); axk[1].legend(fontsize=8, frameon=False, loc="upper right")

    for _j in range(f_mix.shape[0]):
        axk[2].plot(grid, f_mix[_j], lw=1.6, label=f"filter {_j + 1}")
    axk[2].set_title("A multivariate kernel mixes all four channels into feature maps (after ReLU)", fontsize=10)
    axk[2].set_ylabel("activation"); axk[2].set_xlabel("hours since ICU admission")
    axk[2].legend(ncol=3, fontsize=8, frameon=False, loc="upper right")

    figk.tight_layout(h_pad=0.6)
    figk
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From representation to prediction: a sequence-to-label 1D CNN

    The kernels above only *describe* the signal. To **predict**, we attach a classifier
    head, the same shape as **SeizureNet**: a window of physiological signal in, a clinical
    class out. Task here: from the **first 24 h** of the four vitals, predict whether an ICU
    stay will be **long or short** (length of stay above vs below the cohort median).

    The conv stack reads the raw 24-hour window and pools it to a single vector that feeds a
    logistic head. We score it with **5-fold cross-validation** against two references:
    **chance** (0.5) and a **logistic regression on hand-made summary statistics**
    (per-channel mean, sd, min, max, last value), the classic non-temporal baseline. To
    avoid label leakage we keep only stays that lasted past the 24 h window, so "having 24 h
    of data" cannot itself betray the label.

    Honest caveat: the open MIMIC-IV **demo** has fewer than 100 usable stays, so the numbers
    are modest and noisy. The point is the method and the comparison: reading temporal shape,
    the 1D-CNN edges out the summary-stat baseline (about 0.61 vs 0.52 AUROC). On full MIMIC
    with thousands of stays, both the gap and its stability grow. This is the Week 10
    (predictive modelling) bridge.
    """)
    return


@app.cell
def _(Path, np, pd):
    WIN_H = 24                 # predict from the first 24 h of vitals
    MIN_LOS = 1.5              # keep only stays that lasted past the window (no leakage)
    CLF_CH = [("HR", (220045,), (20, 220)), ("BP", (220052, 220181), (30, 180)),
              ("SpO2", (220277,), (60, 100)), ("RR", (220210,), (4, 60))]
    _allids = [i for _, ids, _ in CLF_CH for i in ids]

    icu = pd.read_csv(Path("progression_demo/mimic_demo/icustays.csv.gz"))
    icu = icu[icu["los"] >= MIN_LOS]
    cev = pd.read_csv(Path("progression_demo/mimic_demo/chartevents.csv.gz"),
                      usecols=["stay_id", "itemid", "charttime", "valuenum"])
    cev = cev[cev.stay_id.isin(icu.stay_id) & cev.itemid.isin(_allids)].dropna(subset=["valuenum"]).copy()
    cev["charttime"] = pd.to_datetime(cev["charttime"])

    _g = np.arange(0, WIN_H, 1.0)
    def _window(d):
        d = d.copy()
        d["h"] = (d["charttime"] - d["charttime"].min()).dt.total_seconds() / 3600.0
        chans = []
        for _, ids, (lo, hi) in CLF_CH:
            s = d[d.itemid.isin(ids) & d.valuenum.between(lo, hi) & (d.h < WIN_H)]
            if len(s) == 0:
                return None
            b = pd.Series(s.valuenum.values, index=np.floor(s["h"].values)).groupby(level=0).mean()
            b = b[b.index.isin(_g)]
            v = pd.Series(index=_g, dtype=float)
            v.loc[b.index.values] = b.values
            chans.append(v.ffill().bfill().values)
        return np.array(chans)                 # (4, WIN_H)

    _X, _los = [], []
    for _sid, _d in cev.groupby("stay_id"):
        _w = _window(_d)
        if _w is None or np.isnan(_w).any():
            continue
        _X.append(_w)
        _los.append(icu.loc[icu.stay_id == _sid, "los"].iloc[0])
    Xc = np.array(_X, dtype=np.float32)
    _los = np.array(_los)
    yc = (_los > np.median(_los)).astype(int)          # long (1) vs short (0)
    _mu = Xc.mean((0, 2), keepdims=True)
    _sd = Xc.std((0, 2), keepdims=True) + 1e-6
    Xc = (Xc - _mu) / _sd                              # per-channel z-score
    print(f"classification cohort: {len(Xc)} stays, {Xc.shape[1]} channels x {Xc.shape[2]} h; "
          f"{int(yc.sum())} long / {int((1 - yc).sum())} short (median los {np.median(_los):.1f} d)")
    return Xc, yc


@app.cell
def _(BLUE, GREY, RED, Xc, nn, np, plt, torch, yc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    def _feats(x):     # non-temporal summary stats per channel
        return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), x[:, :, -1]], axis=1)

    class _CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(4, 16, 3, padding=1), nn.ReLU(),
                nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(16, 1))

        def forward(s, x):
            return s.net(x).squeeze(-1)

    def _fit_cnn(Xtr, ytr, Xte, seed):
        torch.manual_seed(seed)
        m = _CNN()
        opt = torch.optim.Adam(m.parameters(), 3e-3, weight_decay=1e-3)
        lf = nn.BCEWithLogitsLoss()
        xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
        m.train()
        for _ in range(120):
            opt.zero_grad(); lf(m(xt), yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(Xte))).numpy()

    Fb = _feats(Xc)
    _auc_cnn, _auc_lr = [], []
    for _seed in (0, 1, 2):                              # repeated 5-fold CV for stability
        for _tr, _te in StratifiedKFold(5, shuffle=True, random_state=_seed).split(Xc, yc):
            _p = _fit_cnn(Xc[_tr], yc[_tr], Xc[_te], _seed)
            _auc_cnn.append(roc_auc_score(yc[_te], _p))
            _scal = StandardScaler().fit(Fb[_tr])
            _lr = LogisticRegression(max_iter=2000, C=0.5).fit(_scal.transform(Fb[_tr]), yc[_tr])
            _auc_lr.append(roc_auc_score(yc[_te], _lr.predict_proba(_scal.transform(Fb[_te]))[:, 1]))

    _names = ["chance", "logistic\n(summary stats)", "1D-CNN\n(raw window)"]
    _means = [0.5, np.mean(_auc_lr), np.mean(_auc_cnn)]
    _errs = [0.0, np.std(_auc_lr), np.std(_auc_cnn)]
    figp, axp = plt.subplots(figsize=(6.6, 4.3))
    _bars = axp.bar(_names, _means, yerr=_errs, capsize=5, color=[GREY, BLUE, RED], alpha=.85)
    axp.axhline(0.5, color="k", lw=.8, ls="--", alpha=.5)
    axp.set_ylim(0.4, 0.8); axp.set_ylabel("AUROC (3x 5-fold CV)")
    axp.set_title("Predicting long vs short ICU stay from the first 24 h of vitals", fontsize=11)
    for _b, _m in zip(_bars, _means):
        axp.annotate(f"{_m:.2f}", (_b.get_x() + _b.get_width() / 2, _m + 0.012), ha="center", fontsize=11)
    figp.tight_layout()
    print(f"AUROC  1D-CNN {np.mean(_auc_cnn):.3f}  logistic {np.mean(_auc_lr):.3f}  (chance 0.5)")
    figp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The same operator on a population time series: COVID-19 daily cases

    A convolution does not care whether a channel is a vital sign or a public-health
    count: any regular-grid series works. **COVID-19 daily new cases** (Our World in Data,
    JHU CSSE, CC BY 4.0) are reported on a daily grid but carry a strong **weekly
    artifact**: weekends are under-reported and the backlog lands on weekdays, so the raw
    curve has a 7-day sawtooth that is *reporting*, not epidemiology.

    This is where a fixed kernel earns its keep. A **width-7 moving average** (layer 1)
    averages over exactly one weekly cycle and erases the sawtooth, leaving the epidemic
    wave. Feeding that smoothed signal into a **difference kernel** (layer 2, so a small
    **2-layer CNN**) gives a clean trend: positive while the wave accelerates, negative as
    it recedes. Shown for the United States across the Delta-to-Omicron surge of late 2021.
    """)
    return


@app.cell
def _(Path, np, pd):
    import io
    import urllib.request

    COVID_SLICE = Path("progression_demo/covid_demo/owid_jhu_slice.csv")

    def _ensure_covid():
        """Use the committed slice; if absent, fetch the full OWID/JHU daily file once."""
        if COVID_SLICE.exists():
            return COVID_SLICE
        url = ("https://raw.githubusercontent.com/owid/covid-19-data/master/"
               "public/data/jhu/full_data.csv")
        req = urllib.request.Request(url, headers={"User-Agent": "ml4health-lecture/1.0"})
        raw = urllib.request.urlopen(req, timeout=90).read()
        full = pd.read_csv(io.BytesIO(raw), usecols=["date", "location", "new_cases"])
        keep = ["United States", "Germany", "Italy", "United Kingdom"]
        full = full[full.location.isin(keep) & full.date.between("2020-03-01", "2022-06-30")]
        COVID_SLICE.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(COVID_SLICE, index=False)
        return COVID_SLICE

    COUNTRY = "United States"
    cov = pd.read_csv(_ensure_covid(), parse_dates=["date"])
    cov = cov[cov.location == COUNTRY].sort_values("date").set_index("date")
    cov = cov.loc["2021-09-01":"2022-01-31"]
    cov_y = cov["new_cases"].to_numpy(dtype=float) / 1e3      # cases per day, thousands
    cov_t = np.arange(len(cov_y))
    print(f"{COUNTRY}: {len(cov_y)} daily points, peak {cov_y.max():.0f}k cases/day")
    return COUNTRY, cov_t, cov_y


@app.cell
def _(BLUE, COUNTRY, GREY, RED, cov_t, cov_y, nn, plt, torch):
    _K = 7                                                # one weekly cycle
    _y = torch.tensor(cov_y[None, None], dtype=torch.float32)
    _avg = nn.Conv1d(1, 1, _K, padding=_K // 2, bias=False)
    _avg.weight.data[:] = torch.ones(1, 1, _K) / _K       # layer 1: moving average
    _diff = nn.Conv1d(1, 1, _K, padding=_K // 2, bias=False)
    _diff.weight.data[:] = torch.tensor([-1., -1, -1, 0, 1, 1, 1]).view(1, 1, _K) / 4
    with torch.no_grad():
        _sm = _avg(_y)                                    # smooth, then
        _trend = _diff(_sm)[0, 0].numpy()                 # difference the smoothed signal
        _smooth = _sm[0, 0].numpy()

    figv, axv = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    axv[0].plot(cov_t, cov_y, color=GREY, lw=1.0, marker="o", ms=2.5, label="raw daily new cases")
    axv[0].plot(cov_t, _smooth, color=RED, lw=2.4, label="layer 1: width-7 average (removes weekend dip)")
    axv[0].set_title(f"{COUNTRY} COVID-19, Sep 2021 to Jan 2022 (OWID / JHU, CC BY 4.0)", fontsize=10)
    axv[0].set_ylabel("new cases / day (thousands)")
    axv[0].legend(fontsize=9, frameon=False, loc="upper left")

    axv[1].plot(cov_t, _trend, color=BLUE, lw=1.9)
    axv[1].axhline(0, color="k", lw=.5, alpha=.4)
    axv[1].fill_between(cov_t, _trend, 0, where=_trend > 0, color=RED, alpha=.18)
    axv[1].fill_between(cov_t, _trend, 0, where=_trend < 0, color=BLUE, alpha=.18)
    axv[1].set_title("layer 2: difference kernel on the smoothed signal (accelerating vs receding)", fontsize=10)
    axv[1].set_ylabel("trend (thousands/day)"); axv[1].set_xlabel("days since 1 Sep 2021")

    figv.tight_layout(h_pad=0.5)
    figv
    return


if __name__ == "__main__":
    app.run()
