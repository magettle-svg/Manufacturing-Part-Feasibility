"""
Manufacturing Part Feasibility — 50+ Pipeline Comparison
=========================================================
Run:  python pipeline_comparison.py

Outputs:
  - Live progress printed to terminal
  - pipeline_results.csv  — full results table
  - pipeline_results.html — interactive Plotly charts
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    BaggingClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, Normalizer,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these paths before running
# ══════════════════════════════════════════════════════════════════════════════
FEASIBLE_DIR   = r"C:\Users\maget\feasible\feasible"
INFEASIBLE_DIR = r"C:\Users\maget\infeasible\infeasible"
Z_CUTOFF       = -477       # strip base layer below this Z value
N_CV_SPLITS    = 5          # cross-validation folds
OUTPUT_CSV     = "pipeline_results.csv"
OUTPUT_HTML    = "pipeline_results.html"
# ══════════════════════════════════════════════════════════════════════════════


# ── Sampling utilities (no imblearn required) ──────────────────────────────────

def smote(X, y, k=5, random_state=42):
    """Synthetic Minority Oversampling — balances by generating synthetic minority points."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_cls = classes[np.argmin(counts)]
    X_min   = X[y == min_cls]
    n_gen   = counts.max() - counts.min()
    synthetic = []
    for _ in range(n_gen):
        i    = rng.integers(0, len(X_min))
        pt   = X_min[i]
        d    = np.linalg.norm(X_min - pt, axis=1)
        d[i] = np.inf
        nn   = np.argsort(d)[:k]
        nb   = X_min[rng.choice(nn)]
        synthetic.append(pt + rng.uniform(0, 1) * (nb - pt))
    X_out = np.vstack([X, np.array(synthetic)])
    y_out = np.concatenate([y, np.full(n_gen, min_cls)])
    return X_out, y_out


def adasyn(X, y, k=5, beta=1.0, random_state=42):
    """ADASYN — generates more synthetic points where the classifier struggles most."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_cls = classes[np.argmin(counts)]
    X_min   = X[y == min_cls]
    n_gen   = int((counts.max() - counts.min()) * beta)
    tree    = KDTree(X)
    _, inds = tree.query(X_min, k=k + 1)
    ratios  = np.array([
        np.sum(y[inds[i][1:]] != min_cls) / k
        for i in range(len(X_min))
    ])
    if ratios.sum() == 0:
        return smote(X, y, k, random_state)
    ratios /= ratios.sum()
    g = (ratios * n_gen).astype(int)
    synthetic = []
    for i, gi in enumerate(g):
        if gi == 0:
            continue
        pt   = X_min[i]
        d    = np.linalg.norm(X_min - pt, axis=1)
        d[i] = np.inf
        nn   = np.argsort(d)[:k]
        for _ in range(gi):
            nb = X_min[rng.choice(nn)]
            synthetic.append(pt + rng.uniform(0, 1) * (nb - pt))
    if not synthetic:
        return X, y
    X_out = np.vstack([X, np.array(synthetic)])
    y_out = np.concatenate([y, np.full(len(synthetic), min_cls)])
    return X_out, y_out


def undersample(X, y, random_state=42):
    """Random undersampling of the majority class."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        indices.append(rng.choice(cls_idx, min_count, replace=False))
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def smote_tomek(X, y, random_state=42):
    """SMOTE followed by Tomek link removal to clean the decision boundary."""
    X_s, y_s = smote(X, y, random_state=random_state)
    tree      = KDTree(X_s)
    _, inds   = tree.query(X_s, k=2)
    nn        = inds[:, 1]
    remove    = set()
    for i in range(len(X_s)):
        j = nn[i]
        if y_s[i] != y_s[j] and nn[j] == i:
            remove.add(i)
            remove.add(j)
    keep = [i for i in range(len(X_s)) if i not in remove]
    return X_s[keep], y_s[keep]


SAMPLERS = {
    "SMOTE":       smote,
    "ADASYN":      adasyn,
    "Undersample": undersample,
    "SMOTE+Tomek": smote_tomek,
    "None":        lambda X, y, **kw: (X, y),
}


# ── Feature extractor ──────────────────────────────────────────────────────────

def extract_features(pts, n_bins=20, n_slices=15,
                     n_normal_sample=500, n_density_sample=300):
    """
    439-dimensional feature vector from a raw Nx3 point cloud.
    Groups:
      A. Original   — bbox, extent, centroid, std, percentiles, histograms (154)
      B. PCA shape  — linearity, planarity, sphericity, eigenvalues         (11)
      C. Z-slices   — per-slice density + gradient                          (29)
      D. Curvature  — local surface curvature via neighborhood PCA          (5)
      E. Convex hull— volume, area, compactness, solidity                   (5)
      F. Radial     — distance from centroid axis + histogram               (24)
      G. Moments    — skewness, kurtosis per axis                           (6)
      H. Density    — mean nearest-neighbor distances                       (5)
      I. Cross-sect — XZ and YZ 2D histograms                              (200)
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # A. Original
    bbox      = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
    extent    = [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]
    centroid  = [x.mean(), y.mean(), z.mean()]
    std       = [x.std(),  y.std(),  z.std()]
    percs     = [10, 25, 50, 75, 90]
    z_hist, _    = np.histogram(z, bins=n_bins, range=(z.min(), z.max()), density=True)
    xy_hist,_,_  = np.histogram2d(x, y, bins=10, density=True)
    roughness    = np.abs(z - np.median(z)).mean()
    asp_xy = extent[0] / (extent[1] + 1e-9)
    asp_xz = extent[0] / (extent[2] + 1e-9)
    asp_yz = extent[1] / (extent[2] + 1e-9)
    feats_A = (bbox + extent + centroid + std +
               np.percentile(x, percs).tolist() +
               np.percentile(y, percs).tolist() +
               np.percentile(z, percs).tolist() +
               z_hist.tolist() + xy_hist.flatten().tolist() +
               [roughness, asp_xy, asp_xz, asp_yz])

    # B. PCA shape
    pca_s = PCA(n_components=3).fit(pts)
    ev    = pca_s.explained_variance_
    feats_B = [
        (ev[0]-ev[1]) / (ev[0]+1e-9),
        (ev[1]-ev[2]) / (ev[0]+1e-9),
         ev[2]        / (ev[0]+1e-9),
        np.cbrt(np.prod(ev)),
        (ev[0]-ev[2]) / (ev[0]+1e-9),
        *ev,
        *pca_s.explained_variance_ratio_,
    ]

    # C. Z-slice density + gradient
    edges      = np.linspace(z.min(), z.max(), n_slices + 1)
    slice_dens = np.array([
        np.sum((z >= edges[i]) & (z < edges[i+1]))
        for i in range(n_slices)
    ], dtype=np.float32) / (len(z) + 1e-9)
    feats_C = slice_dens.tolist() + np.diff(slice_dens).tolist()

    # D. Local curvature
    idx_n     = np.random.choice(len(pts), min(n_normal_sample, len(pts)), replace=False)
    pts_n     = pts[idx_n]
    tree_n    = KDTree(pts_n)
    _, inds_n = tree_n.query(pts_n, k=11)
    curvs     = []
    for i in range(len(pts_n)):
        nb   = pts_n[inds_n[i][1:]]
        cen  = nb - nb.mean(axis=0)
        ev2  = np.linalg.eigvalsh(cen.T @ cen)
        ev2  = np.sort(ev2)[::-1]
        curvs.append(ev2[2] / (ev2.sum() + 1e-9))
    curvs   = np.array(curvs)
    feats_D = [curvs.mean(), curvs.std(),
               np.percentile(curvs, 25),
               np.percentile(curvs, 75),
               np.percentile(curvs, 90)]

    # E. Convex hull
    try:
        idx_h   = np.random.choice(len(pts), min(2000, len(pts)), replace=False)
        hull    = ConvexHull(pts[idx_h])
        ext_v   = (extent[0] * extent[1] * extent[2]) + 1e-9
        feats_E = [hull.volume, hull.area, hull.volume / ext_v,
                   len(pts) / (hull.nsimplex + 1e-9), hull.nsimplex]
    except Exception:
        feats_E = [0.0] * 5

    # F. Radial distribution
    cx, cy  = x.mean(), y.mean()
    radii   = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_hist, _ = np.histogram(radii, bins=15, density=True)
    feats_F = ([radii.mean(), radii.std(), radii.min(), radii.max()] +
               np.percentile(radii, [10, 25, 50, 75, 90]).tolist() +
               r_hist.tolist())

    # G. Statistical moments
    feats_G = [skew(x), skew(y), skew(z),
               kurtosis(x), kurtosis(y), kurtosis(z)]

    # H. Local density
    idx_d      = np.random.choice(len(pts), min(n_density_sample, len(pts)), replace=False)
    tree_d     = KDTree(pts)
    dists_d, _ = tree_d.query(pts[idx_d], k=6)
    nn_d       = dists_d[:, 1:].mean(axis=1)
    feats_H    = [nn_d.mean(), nn_d.std(),
                  np.percentile(nn_d, 10),
                  np.percentile(nn_d, 50),
                  np.percentile(nn_d, 90)]

    # I. Cross-section histograms
    xz_hist, _, _ = np.histogram2d(x, z, bins=10, density=True)
    yz_hist, _, _ = np.histogram2d(y, z, bins=10, density=True)
    feats_I = xz_hist.flatten().tolist() + yz_hist.flatten().tolist()

    all_feats = (feats_A + feats_B + feats_C + feats_D +
                 feats_E + feats_F + feats_G + feats_H + feats_I)
    return np.array(all_feats, dtype=np.float32)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ply_ascii(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip() == "end_header":
            skip = i + 1
            break
    data = np.fromstring("".join(lines[skip:]), dtype=np.float64, sep=" ")
    return data.reshape(-1, 3)


def load_folder(folder, label, z_cutoff=None):
    files    = sorted(Path(folder).glob("*.ply"))
    features, labels = [], []
    for i, f in enumerate(files):
        try:
            pts = load_ply_ascii(f)
            if z_cutoff is not None:
                pts = pts[pts[:, 2] > z_cutoff]
            if len(pts) < 10:
                continue
            features.append(extract_features(pts))
            labels.append(label)
            if (i + 1) % 50 == 0:
                tag = "Feasible" if label == 1 else "Infeasible"
                print(f"    [{tag}] {i+1}/{len(files)} loaded...")
        except Exception as e:
            print(f"    Skip {f.name}: {e}")
    return features, labels


# ── Pipeline registry ──────────────────────────────────────────────────────────

def build_pipelines(n_features):
    """Returns dict of {name: (sklearn_pipeline, sampler_fn)} — 63 pipelines."""
    pipelines = {}

    SC = {
        "Std":      StandardScaler(),
        "MinMax":   MinMaxScaler(),
        "Robust":   RobustScaler(),
        "Power":    PowerTransformer(method="yeo-johnson"),
        "Quantile": QuantileTransformer(output_distribution="normal",
                                         n_quantiles=min(200, n_features)),
        "Norm":     Normalizer(),
    }

    RED = {
        "PCA50":     PCA(n_components=min(50,  n_features)),
        "PCA100":    PCA(n_components=min(100, n_features)),
        "SelectK50": SelectKBest(f_classif, k=min(50, n_features)),
        "VarThresh": VarianceThreshold(threshold=0.01),
    }

    CLF = {
        "RF200":     RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=42, n_jobs=-1),
        "RF100":     RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                             random_state=42, n_jobs=-1),
        "ET200":     ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1),
        "GB100":     GradientBoostingClassifier(n_estimators=100, random_state=42),
        "GB200":     GradientBoostingClassifier(n_estimators=200, random_state=42),
        "AdaBoost":  AdaBoostClassifier(n_estimators=100, random_state=42,
                                         algorithm="SAMME"),
        "LR_L2":     LogisticRegression(C=1.0, class_weight="balanced",
                                         max_iter=1000, random_state=42),
        "LR_L1":     LogisticRegression(C=1.0, penalty="l1", solver="saga",
                                         class_weight="balanced",
                                         max_iter=1000, random_state=42),
        "SVM_RBF":   SVC(kernel="rbf", class_weight="balanced",
                          probability=True, random_state=42),
        "SVM_Lin":   LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
        "KNN5":      KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "KNN11":     KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
        "GNB":       GaussianNB(),
        "DT":        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "LDA":       LinearDiscriminantAnalysis(),
        "QDA":       QuadraticDiscriminantAnalysis(),
        "MLP_1L":    MLPClassifier(hidden_layer_sizes=(256,),
                                    max_iter=500, random_state=42),
        "MLP_2L":    MLPClassifier(hidden_layer_sizes=(256, 128),
                                    max_iter=500, random_state=42),
        "MLP_3L":    MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                    max_iter=500, random_state=42),
        "SGD":       SGDClassifier(class_weight="balanced", random_state=42,
                                    max_iter=1000),
        "Ridge":     CalibratedClassifierCV(RidgeClassifier(class_weight="balanced")),
        "Bagging_DT":BaggingClassifier(
                         estimator=DecisionTreeClassifier(class_weight="balanced"),
                         n_estimators=50, random_state=42, n_jobs=-1),
        "Bagging_KNN":BaggingClassifier(
                         estimator=KNeighborsClassifier(n_neighbors=5),
                         n_estimators=30, random_state=42, n_jobs=-1),
    }

    def make_pipe(scaler, reducer, clf):
        steps = [("scaler", scaler)]
        if reducer is not None:
            steps.append(("reducer", reducer))
        steps.append(("clf", clf))
        return Pipeline(steps)

    # Group 01 — Baseline: vary scaler, fix RF200, no sampler
    for sc_name, sc in SC.items():
        pipelines[f"01_Baseline_{sc_name}_RF200"] = (
            make_pipe(sc, None, RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, n_jobs=-1)),
            SAMPLERS["None"],
        )

    # Group 02 — SMOTE + StandardScaler, vary classifier (14 classifiers)
    for clf_name in ["RF200","ET200","GB100","LR_L2","SVM_RBF",
                     "KNN5","GNB","DT","LDA","MLP_1L",
                     "AdaBoost","SGD","Ridge","Bagging_DT"]:
        pipelines[f"02_SMOTE_Std_{clf_name}"] = (
            make_pipe(StandardScaler(), None, CLF[clf_name]),
            SAMPLERS["SMOTE"],
        )

    # Group 03 — Std + RF200, vary sampler
    for samp_name, samp_fn in SAMPLERS.items():
        pipelines[f"03_Std_RF200_{samp_name}"] = (
            make_pipe(StandardScaler(), None,
                      RandomForestClassifier(n_estimators=200,
                                             class_weight="balanced",
                                             random_state=42, n_jobs=-1)),
            samp_fn,
        )

    # Group 04 — SMOTE + Std + PCA50, vary classifier
    for clf_name in ["RF200","ET200","LR_L2","SVM_RBF","MLP_2L",
                     "KNN5","GB100","LDA","GNB","Ridge"]:
        pipelines[f"04_SMOTE_Std_PCA50_{clf_name}"] = (
            make_pipe(StandardScaler(), RED["PCA50"], CLF[clf_name]),
            SAMPLERS["SMOTE"],
        )

    # Group 05 — SMOTE + Robust + SelectK50, vary classifier
    for clf_name in ["RF200","ET200","GB100","LR_L2","MLP_2L"]:
        pipelines[f"05_SMOTE_Robust_SelectK50_{clf_name}"] = (
            make_pipe(RobustScaler(),
                      SelectKBest(f_classif, k=min(50, n_features)),
                      CLF[clf_name]),
            SAMPLERS["SMOTE"],
        )

    # Group 06 — Power/Quantile scalers with top classifiers
    for sc_name in ["Power", "Quantile"]:
        for clf_name in ["RF200", "ET200", "MLP_2L"]:
            pipelines[f"06_SMOTE_{sc_name}_{clf_name}"] = (
                make_pipe(SC[sc_name], None, CLF[clf_name]),
                SAMPLERS["SMOTE"],
            )

    # Group 07 — ADASYN sampler, vary classifier
    for clf_name in ["RF200","ET200","GB100","LR_L2","MLP_2L"]:
        pipelines[f"07_ADASYN_Std_{clf_name}"] = (
            make_pipe(StandardScaler(), None, CLF[clf_name]),
            SAMPLERS["ADASYN"],
        )

    # Group 08 — SMOTE+Tomek, MLP variants + RF/ET
    for clf_name in ["MLP_1L","MLP_2L","MLP_3L","RF200","ET200"]:
        pipelines[f"08_SMOTETomek_Std_{clf_name}"] = (
            make_pipe(StandardScaler(), None, CLF[clf_name]),
            SAMPLERS["SMOTE+Tomek"],
        )

    # Group 09 — Ensemble meta-pipelines
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                           random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                         random_state=42)),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ],
        voting="soft", n_jobs=-1,
    )
    stacking_clf = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                           random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                         random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=3, n_jobs=-1,
    )
    pipelines["09_SMOTE_Std_VotingEnsemble"] = (
        make_pipe(StandardScaler(), None, voting_clf), SAMPLERS["SMOTE"])
    pipelines["09_SMOTE_Std_StackingEnsemble"] = (
        make_pipe(StandardScaler(), None, stacking_clf), SAMPLERS["SMOTE"])
    pipelines["09_SMOTE_Robust_VotingEnsemble"] = (
        make_pipe(RobustScaler(), None, voting_clf), SAMPLERS["SMOTE"])

    # Group 10 — Undersample + Std, vary classifier
    for clf_name in ["RF200","ET200","SVM_RBF","MLP_2L","GB100"]:
        pipelines[f"10_Undersample_Std_{clf_name}"] = (
            make_pipe(StandardScaler(), None, CLF[clf_name]),
            SAMPLERS["Undersample"],
        )

    return pipelines


# ── Evaluation engine ──────────────────────────────────────────────────────────

def evaluate_pipeline(name, pipe, sampler_fn, X, y, n_splits=5):
    """Stratified k-fold CV — sampler applied inside each fold (no leakage)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_macros, f1_feas, f1_inf, fit_times = [], [], [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        try:
            X_tr, y_tr = sampler_fn(X_tr, y_tr)
        except Exception:
            pass
        t0 = time.time()
        try:
            pipe.fit(X_tr, y_tr)
            fit_times.append(time.time() - t0)
            y_pred = pipe.predict(X_te)
            f1_macros.append(f1_score(y_te, y_pred, average="macro",  zero_division=0))
            f1_feas.append(  f1_score(y_te, y_pred, pos_label=1,      zero_division=0))
            f1_inf.append(   f1_score(y_te, y_pred, pos_label=0,      zero_division=0))
        except Exception:
            fit_times.append(0)
            f1_macros.append(0); f1_feas.append(0); f1_inf.append(0)

    return {
        "Pipeline":      name,
        "F1 Macro":      np.mean(f1_macros),
        "F1 Macro Std":  np.std(f1_macros),
        "F1 Feasible":   np.mean(f1_feas),
        "F1 Infeasible": np.mean(f1_inf),
        "Fit Time (s)":  np.mean(fit_times),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def build_plots(df):
    """Returns a list of Plotly figures."""
    figs = []

    # Plot 1 — Top 20 F1 Macro bar chart
    top20 = df.head(20)
    fig1  = go.Figure(go.Bar(
        x=top20["F1 Macro"], y=top20["Pipeline"], orientation="h",
        error_x=dict(type="data", array=top20["F1 Macro Std"], visible=True),
        marker=dict(color=top20["F1 Macro"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="F1")),
        text=top20["F1 Macro"].round(3), textposition="outside",
    ))
    fig1.update_layout(
        title=dict(text="Top 20 Pipelines — F1 Macro (5-fold CV)", x=0.5),
        xaxis=dict(title="F1 Macro", range=[0, 1.05], color="#888", gridcolor="#2a2a2a"),
        yaxis=dict(autorange="reversed", color="#888", tickfont=dict(size=10)),
        paper_bgcolor="rgb(8,10,20)", plot_bgcolor="rgb(8,10,20)",
        font_color="white", height=700,
        margin=dict(l=350, r=80, t=60, b=40),
    )
    figs.append(fig1)

    # Plot 2 — F1 Feasible vs F1 Infeasible scatter
    group_colors = {
        "01":"#888888","02":"#00c8ff","03":"#ff6b35","04":"#7fff6b",
        "05":"#ffdd00","06":"#ff00ff","07":"#00ffaa","08":"#ff8888",
        "09":"#ffffff","10":"#aaaaff",
    }
    fig2 = go.Figure()
    for grp, color in group_colors.items():
        sub = df[df["Pipeline"].str.startswith(grp)]
        if len(sub) == 0:
            continue
        fig2.add_trace(go.Scatter(
            x=sub["F1 Feasible"], y=sub["F1 Infeasible"], mode="markers",
            marker=dict(size=8, color=color, opacity=0.8,
                        line=dict(width=1, color="white")),
            name=f"Group {grp}", text=sub["Pipeline"],
            hovertemplate="<b>%{text}</b><br>F1 Feasible:%{x:.3f}<br>"
                          "F1 Infeasible:%{y:.3f}<extra></extra>",
        ))
    fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(color="#444", dash="dash", width=1))
    fig2.update_layout(
        title=dict(text="F1 Feasible vs F1 Infeasible — All Pipelines", x=0.5),
        xaxis=dict(title="F1 Feasible", range=[0,1.05], color="#888", gridcolor="#2a2a2a"),
        yaxis=dict(title="F1 Infeasible", range=[0,1.05], color="#888", gridcolor="#2a2a2a"),
        paper_bgcolor="rgb(8,10,20)", plot_bgcolor="rgb(8,10,20)",
        font_color="white", legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=600, margin=dict(l=60, r=40, t=60, b=60),
    )
    figs.append(fig2)

    # Plot 3 — F1 vs Training Time
    fig3 = go.Figure(go.Scatter(
        x=df["Fit Time (s)"], y=df["F1 Macro"], mode="markers",
        marker=dict(size=8, color=df["F1 Macro"], colorscale="Plasma",
                    showscale=True, colorbar=dict(title="F1"), opacity=0.8),
        text=df["Pipeline"],
        hovertemplate="<b>%{text}</b><br>Fit Time:%{x:.1f}s<br>"
                      "F1 Macro:%{y:.3f}<extra></extra>",
    ))
    fig3.update_layout(
        title=dict(text="F1 Macro vs Training Time — Efficiency Frontier", x=0.5),
        xaxis=dict(title="Mean Fit Time per Fold (s)", color="#888", gridcolor="#2a2a2a"),
        yaxis=dict(title="F1 Macro", range=[0,1.05], color="#888", gridcolor="#2a2a2a"),
        paper_bgcolor="rgb(8,10,20)", plot_bgcolor="rgb(8,10,20)",
        font_color="white", height=500, margin=dict(l=60, r=40, t=60, b=60),
    )
    figs.append(fig3)

    # Plot 4 — Group box plot
    group_labels = {
        "01":"01 Baseline","02":"02 SMOTE+Clf","03":"03 Vary Sampler",
        "04":"04 PCA50",   "05":"05 SelectK",  "06":"06 Power/Quant",
        "07":"07 ADASYN",  "08":"08 SMOTE+Tomek","09":"09 Ensemble",
        "10":"10 Undersample",
    }
    df2 = df.copy()
    df2["Group Label"] = df2["Pipeline"].str.split("_").str[0].map(group_labels)
    fig4  = go.Figure()
    for grp_label in sorted(df2["Group Label"].dropna().unique()):
        sub = df2[df2["Group Label"] == grp_label]
        fig4.add_trace(go.Box(
            y=sub["F1 Macro"], name=grp_label,
            boxpoints="all", jitter=0.3, pointpos=-1.5, marker_size=5,
        ))
    fig4.update_layout(
        title=dict(text="F1 Macro Distribution by Pipeline Group", x=0.5),
        yaxis=dict(title="F1 Macro", range=[0,1.05], color="#888", gridcolor="#2a2a2a"),
        xaxis=dict(color="#888"),
        paper_bgcolor="rgb(8,10,20)", plot_bgcolor="rgb(8,10,20)",
        font_color="white", showlegend=False,
        height=520, margin=dict(l=60, r=40, t=60, b=100),
    )
    figs.append(fig4)

    return figs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Manufacturing Part Feasibility — 50+ Pipeline Comparison")
    print("=" * 65)

    # Load data
    print("\n[1/4] Loading data...")
    print("  Feasible designs:")
    ff, fl = load_folder(FEASIBLE_DIR,   label=1, z_cutoff=Z_CUTOFF)
    print(f"    → {len(ff)} samples loaded")
    print("  Infeasible designs:")
    inf_f, il = load_folder(INFEASIBLE_DIR, label=0, z_cutoff=Z_CUTOFF)
    print(f"    → {len(inf_f)} samples loaded")

    X = np.nan_to_num(
        np.array(ff + inf_f, dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    y = np.array(fl + il, dtype=np.int32)
    print(f"\n  Dataset: {X.shape}  |  "
          f"Feasible: {y.sum()}  Infeasible: {(y==0).sum()}")

    # Build pipelines
    print("\n[2/4] Building pipeline registry...")
    pipelines = build_pipelines(X.shape[1])
    print(f"  → {len(pipelines)} pipelines defined")

    # Run evaluation
    print(f"\n[3/4] Evaluating all pipelines ({N_CV_SPLITS}-fold CV)...")
    print(f"  {'Pipeline':<52} F1 Macro   Time    ETA")
    print("  " + "-" * 65)
    results = []
    n_total = len(pipelines)
    t_start = time.time()

    for i, (name, (pipe, sampler_fn)) in enumerate(pipelines.items()):
        t0 = time.time()
        try:
            result = evaluate_pipeline(name, pipe, sampler_fn, X, y, N_CV_SPLITS)
            results.append(result)
            elapsed       = time.time() - t0
            total_elapsed = time.time() - t_start
            eta           = (total_elapsed / (i + 1)) * (n_total - i - 1)
            print(f"  [{i+1:02d}/{n_total}] {name:<50s}  "
                  f"F1={result['F1 Macro']:.3f}  "
                  f"({elapsed:.0f}s)  ETA {eta/60:.1f}min")
        except Exception as e:
            print(f"  [{i+1:02d}/{n_total}] {name:<50s}  ERROR: {e}")

    total_time = time.time() - t_start
    print(f"\n  Finished in {total_time/60:.1f} min")

    # Results table
    df = (pd.DataFrame(results)
            .sort_values("F1 Macro", ascending=False)
            .reset_index(drop=True))
    df.index += 1

    df.to_csv(OUTPUT_CSV, index=True)
    print(f"\n  Results saved → {OUTPUT_CSV}")

    # Summary
    print("\n[4/4] Summary")
    print("=" * 65)
    best  = df.iloc[0]
    worst = df.iloc[-1]
    df["Group Label"] = (df["Pipeline"].str.split("_").str[0]
                         .map({"01":"Baseline","02":"SMOTE+Clf",
                               "03":"Vary Sampler","04":"PCA50",
                               "05":"SelectK","06":"Power/Quant",
                               "07":"ADASYN","08":"SMOTE+Tomek",
                               "09":"Ensemble","10":"Undersample"}))
    group_means = (df.groupby("Group Label")["F1 Macro"]
                     .mean().sort_values(ascending=False))

    print(f"\n  BEST  : {best['Pipeline']}")
    print(f"          F1 Macro={best['F1 Macro']:.3f} ± {best['F1 Macro Std']:.3f}  "
          f"F1 Feasible={best['F1 Feasible']:.3f}  "
          f"F1 Infeasible={best['F1 Infeasible']:.3f}")
    print(f"\n  WORST : {worst['Pipeline']}")
    print(f"          F1 Macro={worst['F1 Macro']:.3f}")
    print("\n  Group averages:")
    for grp, mean_f1 in group_means.items():
        print(f"    {grp:<20s}  {mean_f1:.3f}")

    print(f"\n  TOP 10:")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 55)
    print(df[["Pipeline","F1 Macro","F1 Macro Std",
              "F1 Infeasible","Fit Time (s)"]].head(10).to_string())

    # Save plots
    figs = build_plots(df)
    plot_titles = [
        "top20_bar", "feasible_vs_infeasible",
        "efficiency_frontier", "group_boxplot",
    ]
    with open(OUTPUT_HTML, "w") as f:
        f.write("<html><head><title>Pipeline Comparison</title></head><body>\n")
        for fig, title in zip(figs, plot_titles):
            f.write(f"<h2>{title}</h2>\n")
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            f.write("\n")
        f.write("</body></html>")
    print(f"\n  Plots saved → {OUTPUT_HTML}")
    print("\nDone ✅")


if __name__ == "__main__":
    main()
