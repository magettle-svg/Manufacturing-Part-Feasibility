"""
app.py — Manufacturing Part Feasibility Dashboard
Run locally:  streamlit run app.py
"""

import io
import time
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, Normalizer, PowerTransformer,
    QuantileTransformer, RobustScaler, StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Part Feasibility Classifier",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #080a14;
    color: #c8d8e8;
}

h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace;
    color: #00c8ff;
    letter-spacing: 0.04em;
}

.stButton > button {
    background: linear-gradient(135deg, #00c8ff22, #00c8ff44);
    border: 1px solid #00c8ff88;
    color: #00c8ff;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.08em;
    border-radius: 4px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c8ff44, #00c8ff66);
    border-color: #00c8ff;
    color: #ffffff;
}

.metric-card {
    background: linear-gradient(135deg, #0d1525, #111a2e);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #6a8aaa;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00c8ff;
    font-weight: 600;
}
.metric-sub {
    font-size: 0.75rem;
    color: #4a6a8a;
    margin-top: 0.2rem;
}

.status-feasible {
    background: linear-gradient(135deg, #0d2e1a, #0a2015);
    border: 1px solid #00ff8844;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    color: #00ff88;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
}
.status-infeasible {
    background: linear-gradient(135deg, #2e0d0d, #200a0a);
    border: 1px solid #ff444444;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    color: #ff4444;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
}

.section-header {
    font-family: 'Share Tech Mono', monospace;
    color: #00c8ff;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

div[data-testid="stSidebar"] {
    background-color: #0a0d1a;
    border-right: 1px solid #1e3a5f;
}

.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #6a8aaa !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.uploadedFile {
    background-color: #0d1525 !important;
    border: 1px solid #1e3a5f !important;
}

.stProgress > div > div {
    background-color: #00c8ff !important;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND — all ML and feature extraction logic
# ══════════════════════════════════════════════════════════════════════════════

def load_ply_ascii(content: bytes) -> np.ndarray:
    """Parse ASCII PLY bytes → Nx3 numpy array."""
    lines = content.decode("utf-8", errors="ignore").splitlines()
    for i, l in enumerate(lines):
        if l.strip() == "end_header":
            skip = i + 1
            break
    data = np.fromstring("\n".join(lines[skip:]), dtype=np.float64, sep=" ")
    return data.reshape(-1, 3)


def extract_features(pts, n_bins=20, n_slices=15,
                     n_normal_sample=300, n_density_sample=200):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # A. Original
    bbox     = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
    extent   = [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]
    centroid = [x.mean(), y.mean(), z.mean()]
    std      = [x.std(), y.std(), z.std()]
    percs    = [10, 25, 50, 75, 90]
    z_hist, _    = np.histogram(z, bins=n_bins, range=(z.min(), z.max()), density=True)
    xy_hist, _,_ = np.histogram2d(x, y, bins=10, density=True)
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
    feats_B = [(ev[0]-ev[1])/(ev[0]+1e-9), (ev[1]-ev[2])/(ev[0]+1e-9),
               ev[2]/(ev[0]+1e-9), np.cbrt(np.prod(ev)),
               (ev[0]-ev[2])/(ev[0]+1e-9), *ev, *pca_s.explained_variance_ratio_]

    # C. Z-slices
    edges      = np.linspace(z.min(), z.max(), n_slices + 1)
    slice_dens = np.array([np.sum((z >= edges[i]) & (z < edges[i+1]))
                           for i in range(n_slices)], dtype=np.float32) / (len(z)+1e-9)
    feats_C    = slice_dens.tolist() + np.diff(slice_dens).tolist()

    # D. Curvature
    idx_n     = np.random.choice(len(pts), min(n_normal_sample, len(pts)), replace=False)
    pts_n     = pts[idx_n]
    tree_n    = KDTree(pts_n)
    _, inds_n = tree_n.query(pts_n, k=11)
    curvs     = []
    for i in range(len(pts_n)):
        nb  = pts_n[inds_n[i][1:]]
        cen = nb - nb.mean(axis=0)
        ev2 = np.sort(np.linalg.eigvalsh(cen.T @ cen))[::-1]
        curvs.append(ev2[2] / (ev2.sum() + 1e-9))
    curvs   = np.array(curvs)
    feats_D = [curvs.mean(), curvs.std(),
               np.percentile(curvs, 25), np.percentile(curvs, 75),
               np.percentile(curvs, 90)]

    # E. Convex hull
    try:
        idx_h   = np.random.choice(len(pts), min(2000, len(pts)), replace=False)
        hull    = ConvexHull(pts[idx_h])
        ext_v   = (extent[0]*extent[1]*extent[2]) + 1e-9
        feats_E = [hull.volume, hull.area, hull.volume/ext_v,
                   len(pts)/(hull.nsimplex+1e-9), hull.nsimplex]
    except Exception:
        feats_E = [0.0] * 5

    # F. Radial
    radii   = np.sqrt((x-x.mean())**2 + (y-y.mean())**2)
    r_hist, _ = np.histogram(radii, bins=15, density=True)
    feats_F = ([radii.mean(), radii.std(), radii.min(), radii.max()] +
               np.percentile(radii, [10,25,50,75,90]).tolist() + r_hist.tolist())

    # G. Moments
    feats_G = [skew(x), skew(y), skew(z), kurtosis(x), kurtosis(y), kurtosis(z)]

    # H. Local density
    idx_d      = np.random.choice(len(pts), min(n_density_sample, len(pts)), replace=False)
    dists_d, _ = KDTree(pts).query(pts[idx_d], k=6)
    nn_d       = dists_d[:, 1:].mean(axis=1)
    feats_H    = [nn_d.mean(), nn_d.std(),
                  np.percentile(nn_d, 10), np.percentile(nn_d, 50),
                  np.percentile(nn_d, 90)]

    # I. Cross-sections
    xz_hist,_,_ = np.histogram2d(x, z, bins=10, density=True)
    yz_hist,_,_ = np.histogram2d(y, z, bins=10, density=True)
    feats_I = xz_hist.flatten().tolist() + yz_hist.flatten().tolist()

    return np.nan_to_num(
        np.array(feats_A+feats_B+feats_C+feats_D+feats_E+feats_F+feats_G+feats_H+feats_I,
                 dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def smote(X, y, k=5, random_state=42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_cls = classes[np.argmin(counts)]
    X_min   = X[y == min_cls]
    n_gen   = counts.max() - counts.min()
    synthetic = []
    for _ in range(n_gen):
        i    = rng.integers(0, len(X_min))
        pt   = X_min[i]
        d    = np.linalg.norm(X_min - pt, axis=1); d[i] = np.inf
        nb   = X_min[rng.choice(np.argsort(d)[:k])]
        synthetic.append(pt + rng.uniform(0, 1) * (nb - pt))
    return (np.vstack([X, np.array(synthetic)]),
            np.concatenate([y, np.full(n_gen, min_cls)]))


def build_classifier(clf_name, n_features):
    clfs = {
        "Random Forest":        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees":          ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression":  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "SVM (RBF)":            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "KNN (k=5)":            KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "MLP (2-layer)":        MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=42),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"),
        "Voting Ensemble":      VotingClassifier(estimators=[
                                    ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
                                    ("et", ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
                                    ("lr", LogisticRegression(class_weight="balanced", max_iter=1000)),
                                ], voting="soft", n_jobs=-1),
    }
    return clfs[clf_name]


def build_scaler(scaler_name, n_features):
    scalers = {
        "StandardScaler":    StandardScaler(),
        "RobustScaler":      RobustScaler(),
        "MinMaxScaler":      MinMaxScaler(),
        "PowerTransformer":  PowerTransformer(method="yeo-johnson"),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal",
                                                    n_quantiles=min(200, n_features)),
    }
    return scalers[scaler_name]


PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,21,37,0.8)",
    font_color="#c8d8e8",
    font=dict(family="Share Tech Mono, monospace"),
)


def make_3d_scatter(pts, z_cutoff=None, title="Point Cloud"):
    if z_cutoff is not None:
        pts = pts[pts[:, 2] > z_cutoff]
    MAX = 60_000
    idx = np.random.choice(len(pts), min(MAX, len(pts)), replace=False)
    p   = pts[idx]
    fig = go.Figure(go.Scatter3d(
        x=p[:,0], y=p[:,1], z=p[:,2], mode="markers",
        marker=dict(size=1.2, color=p[:,2], colorscale="Viridis",
                    colorbar=dict(title="Z", thickness=12, len=0.5), opacity=0.85),
        hovertemplate="X:%{x:.2f} Y:%{y:.2f} Z:%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        scene=dict(
            bgcolor="rgb(8,10,20)",
            xaxis=dict(showbackground=False, color="#4a6a8a", gridcolor="#1e3a5f"),
            yaxis=dict(showbackground=False, color="#4a6a8a", gridcolor="#1e3a5f"),
            zaxis=dict(showbackground=False, color="#4a6a8a", gridcolor="#1e3a5f"),
            aspectmode="data",
        ),
        **PLOT_THEME, margin=dict(l=0,r=0,t=40,b=0), height=420,
    )
    return fig


def make_z_histogram(pts, z_cutoff=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pts[:,2], nbinsx=200,
                               marker_color="#00c8ff", opacity=0.7, name="All Z"))
    if z_cutoff is not None:
        fig.add_vline(x=z_cutoff, line_color="#ff6b35", line_dash="dash",
                      annotation_text=f"cutoff = {z_cutoff}",
                      annotation_font_color="#ff6b35")
    fig.update_layout(
        title=dict(text="Z Distribution", x=0.5, font=dict(size=13)),
        xaxis=dict(title="Z", color="#4a6a8a", gridcolor="#1e3a5f"),
        yaxis=dict(title="Count", color="#4a6a8a", gridcolor="#1e3a5f"),
        **PLOT_THEME, height=280, margin=dict(l=40,r=20,t=40,b=40),
    )
    return fig


def make_results_bar(df_res):
    top = df_res.head(15)
    fig = go.Figure(go.Bar(
        x=top["F1 Macro"], y=top["Pipeline"], orientation="h",
        error_x=dict(type="data", array=top["F1 Macro Std"], visible=True,
                     color="#ffffff44"),
        marker=dict(color=top["F1 Macro"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="F1", thickness=12)),
        text=top["F1 Macro"].round(3), textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Top 15 Pipelines — F1 Macro", x=0.5, font=dict(size=13)),
        xaxis=dict(title="F1 Macro", range=[0,1.1], color="#4a6a8a", gridcolor="#1e3a5f"),
        yaxis=dict(autorange="reversed", color="#c8d8e8", tickfont=dict(size=9)),
        **PLOT_THEME, height=520, margin=dict(l=280,r=60,t=50,b=40),
    )
    return fig


def make_scatter_f1(df_res):
    group_colors = {
        "01":"#888888","02":"#00c8ff","03":"#ff6b35","04":"#7fff6b",
        "05":"#ffdd00","06":"#ff00ff","07":"#00ffaa","08":"#ff8888",
        "09":"#ffffff","10":"#aaaaff",
    }
    fig = go.Figure()
    for grp, color in group_colors.items():
        sub = df_res[df_res["Pipeline"].str.startswith(grp)]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["F1 Feasible"], y=sub["F1 Infeasible"], mode="markers",
            marker=dict(size=7, color=color, opacity=0.85,
                        line=dict(width=1, color="white")),
            name=f"Grp {grp}", text=sub["Pipeline"],
            hovertemplate="<b>%{text}</b><br>Feasible:%{x:.3f} Infeasible:%{y:.3f}<extra></extra>",
        ))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#333", dash="dash"))
    fig.update_layout(
        title=dict(text="F1 Feasible vs F1 Infeasible", x=0.5, font=dict(size=13)),
        xaxis=dict(title="F1 Feasible", range=[0,1.05], color="#4a6a8a", gridcolor="#1e3a5f"),
        yaxis=dict(title="F1 Infeasible", range=[0,1.05], color="#4a6a8a", gridcolor="#1e3a5f"),
        **PLOT_THEME, height=420, margin=dict(l=60,r=20,t=50,b=50),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(size=9)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key, default in [
    ("feasible_pts",   []),
    ("infeasible_pts", []),
    ("X", None), ("y", None),
    ("pipeline_results", None),
    ("trained_pipe", None),
    ("trained_scaler", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 PART FEASIBILITY")
    st.markdown("<div class='section-header'>Configuration</div>", unsafe_allow_html=True)

    z_cutoff = st.slider(
        "Z Cutoff (strip base layer)",
        min_value=-600.0, max_value=0.0,
        value=-477.0, step=1.0,
        help="Points below this Z value are removed (base/fixture layer).",
    )

    st.markdown("<div class='section-header'>Classifier</div>", unsafe_allow_html=True)
    clf_name = st.selectbox("Algorithm", [
        "Random Forest", "Extra Trees", "Gradient Boosting",
        "Logistic Regression", "SVM (RBF)", "KNN (k=5)",
        "MLP (2-layer)", "AdaBoost", "Voting Ensemble",
    ])
    scaler_name = st.selectbox("Scaler", [
        "StandardScaler", "RobustScaler", "MinMaxScaler",
        "PowerTransformer", "QuantileTransformer",
    ])
    use_smote = st.toggle("Apply SMOTE", value=True,
                          help="Oversample minority class during training.")
    n_cv      = st.slider("CV Folds", 3, 10, 5)

    st.markdown("<div class='section-header'>Pipeline Comparison</div>",
                unsafe_allow_html=True)
    run_all = st.toggle("Run all 63 pipelines", value=False,
                        help="Evaluates all pipeline combinations. Takes 15–40 min.")

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Share Tech Mono;font-size:0.65rem;color:#2a4a6a;"
        "line-height:1.8'>"
        "v1.0 · Manufacturing QC<br>"
        "Point Cloud Classifier<br>"
        "439-dim feature vectors<br>"
        "63 pipeline combinations"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — 4 tabs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏭 MANUFACTURING PART FEASIBILITY CLASSIFIER")
st.markdown(
    "<p style='color:#4a6a8a;font-size:0.85rem;margin-top:-0.5rem'>"
    "Upload point cloud scans · Extract geometric features · "
    "Classify feasible vs infeasible designs</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📂  Data Upload",
    "🔬  Feature Analysis",
    "⚙️  Train & Evaluate",
    "🔍  Predict New Part",
])


# ── TAB 1: Data Upload ─────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-header'>Feasible Designs</div>",
                    unsafe_allow_html=True)
        feasible_files = st.file_uploader(
            "Upload feasible .ply files",
            type=["ply"], accept_multiple_files=True, key="feas_upload",
        )
        if feasible_files:
            progress = st.progress(0, text="Loading feasible files...")
            pts_list = []
            for i, f in enumerate(feasible_files):
                try:
                    pts = load_ply_ascii(f.read())
                    pts = pts[pts[:, 2] > z_cutoff]
                    pts_list.append(pts)
                except Exception as e:
                    st.warning(f"Skipped {f.name}: {e}")
                progress.progress((i+1)/len(feasible_files),
                                   text=f"Loaded {i+1}/{len(feasible_files)}")
            st.session_state.feasible_pts = pts_list
            progress.empty()
            st.success(f"✅ {len(pts_list)} feasible files loaded")

    with col2:
        st.markdown("<div class='section-header'>Infeasible Designs</div>",
                    unsafe_allow_html=True)
        infeasible_files = st.file_uploader(
            "Upload infeasible .ply files",
            type=["ply"], accept_multiple_files=True, key="infeas_upload",
        )
        if infeasible_files:
            progress2 = st.progress(0, text="Loading infeasible files...")
            pts_list2 = []
            for i, f in enumerate(infeasible_files):
                try:
                    pts = load_ply_ascii(f.read())
                    pts = pts[pts[:, 2] > z_cutoff]
                    pts_list2.append(pts)
                except Exception as e:
                    st.warning(f"Skipped {f.name}: {e}")
                progress2.progress((i+1)/len(infeasible_files),
                                    text=f"Loaded {i+1}/{len(infeasible_files)}")
            st.session_state.infeasible_pts = pts_list2
            progress2.empty()
            st.success(f"✅ {len(pts_list2)} infeasible files loaded")

    # Dataset summary
    n_feas = len(st.session_state.feasible_pts)
    n_inf  = len(st.session_state.infeasible_pts)
    n_tot  = n_feas + n_inf

    if n_tot > 0:
        st.markdown("---")
        st.markdown("<div class='section-header'>Dataset Summary</div>",
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, sub in [
            (m1, "TOTAL SAMPLES",    str(n_tot),          "files loaded"),
            (m2, "FEASIBLE",         str(n_feas),          f"{n_feas/n_tot*100:.1f}%"),
            (m3, "INFEASIBLE",       str(n_inf),           f"{n_inf/n_tot*100:.1f}%"),
            (m4, "IMBALANCE RATIO",  f"{max(n_feas,n_inf)/max(min(n_feas,n_inf),1):.1f}:1",
             "majority:minority"),
        ]:
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{val}</div>"
                f"<div class='metric-sub'>{sub}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        # Preview first cloud
        st.markdown("---")
        st.markdown("<div class='section-header'>Point Cloud Preview</div>",
                    unsafe_allow_html=True)
        preview_col = "Feasible" if n_feas > 0 else "Infeasible"
        preview_pts = (st.session_state.feasible_pts[0] if n_feas > 0
                       else st.session_state.infeasible_pts[0])
        st.plotly_chart(
            make_3d_scatter(preview_pts, title=f"Preview — {preview_col} Sample #1"),
            use_container_width=True,
        )


# ── TAB 2: Feature Analysis ────────────────────────────────────────────────────
with tab2:
    n_feas = len(st.session_state.feasible_pts)
    n_inf  = len(st.session_state.infeasible_pts)

    if n_feas == 0 and n_inf == 0:
        st.info("Upload point cloud files in the **Data Upload** tab first.")
    else:
        st.markdown("<div class='section-header'>Z Distribution Analysis</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "The Z cutoff separates the **base/fixture layer** (dense spike) "
            "from the **part geometry** (floating points above). "
            "Adjust the slider in the sidebar until the base spike is excluded.",
        )

        # Show Z histogram for first feasible sample
        if n_feas > 0:
            pts_preview = st.session_state.feasible_pts[0]
            # Show raw (before cutoff) by reloading isn't possible here,
            # so show the trimmed cloud distribution
            st.plotly_chart(make_z_histogram(pts_preview, z_cutoff),
                            use_container_width=True)

        # Extract features button
        st.markdown("---")
        st.markdown("<div class='section-header'>Feature Extraction</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "Converts each point cloud into a **439-dimensional** feature vector "
            "spanning geometry, shape, curvature, density, and cross-section profiles."
        )

        if st.button("⚙️  Extract Features from All Files"):
            all_pts  = (st.session_state.feasible_pts +
                        st.session_state.infeasible_pts)
            all_lbls = ([1] * len(st.session_state.feasible_pts) +
                        [0] * len(st.session_state.infeasible_pts))

            progress = st.progress(0, text="Extracting features...")
            feats = []
            for i, pts in enumerate(all_pts):
                feats.append(extract_features(pts))
                progress.progress((i+1)/len(all_pts),
                                   text=f"Extracting {i+1}/{len(all_pts)}...")
            progress.empty()

            st.session_state.X = np.array(feats, dtype=np.float32)
            st.session_state.y = np.array(all_lbls, dtype=np.int32)
            st.success(f"✅ Feature matrix: {st.session_state.X.shape}")

        if st.session_state.X is not None:
            X, y = st.session_state.X, st.session_state.y
            st.markdown("---")
            st.markdown("<div class='section-header'>Feature Statistics</div>",
                        unsafe_allow_html=True)

            # PCA 2D separation plot
            pca2 = PCA(n_components=2)
            X_2d = pca2.fit_transform(
                StandardScaler().fit_transform(X)
            )
            fig_pca = go.Figure()
            for lbl, name, color in [(1,"Feasible","#00c8ff"),(0,"Infeasible","#ff6b35")]:
                mask = y == lbl
                fig_pca.add_trace(go.Scatter(
                    x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers",
                    marker=dict(size=6, color=color, opacity=0.7),
                    name=name,
                ))
            fig_pca.update_layout(
                title=dict(text="PCA 2D — Feature Space Separation", x=0.5,
                           font=dict(size=13)),
                xaxis=dict(title=f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
                           color="#4a6a8a", gridcolor="#1e3a5f"),
                yaxis=dict(title=f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)",
                           color="#4a6a8a", gridcolor="#1e3a5f"),
                **PLOT_THEME, height=400,
                legend=dict(bgcolor="rgba(0,0,0,0.4)"),
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            # Feature group variance
            group_names = {
                "A: Original (0-153)":    (0,   154),
                "B: PCA Shape (154-164)": (154, 165),
                "C: Z-Slices (165-193)":  (165, 194),
                "D: Curvature (194-198)": (194, 199),
                "E: Hull (199-203)":      (199, 204),
                "F: Radial (204-227)":    (204, 228),
                "G: Moments (228-233)":   (228, 234),
                "H: Density (234-238)":   (234, 239),
                "I: Cross-sect (239+)":   (239, X.shape[1]),
            }
            group_var = {k: X[:, s:e].var(axis=0).mean()
                         for k, (s, e) in group_names.items()
                         if e <= X.shape[1]}
            fig_var = go.Figure(go.Bar(
                x=list(group_var.values()),
                y=list(group_var.keys()),
                orientation="h",
                marker=dict(color=list(group_var.values()),
                            colorscale="Plasma", showscale=False),
            ))
            fig_var.update_layout(
                title=dict(text="Mean Feature Variance by Group", x=0.5,
                           font=dict(size=13)),
                xaxis=dict(title="Mean Variance", color="#4a6a8a", gridcolor="#1e3a5f"),
                yaxis=dict(color="#c8d8e8", tickfont=dict(size=10)),
                **PLOT_THEME, height=380, margin=dict(l=200,r=20,t=50,b=40),
            )
            st.plotly_chart(fig_var, use_container_width=True)


# ── TAB 3: Train & Evaluate ────────────────────────────────────────────────────
with tab3:
    if st.session_state.X is None:
        st.info("Extract features in the **Feature Analysis** tab first.")
    else:
        X, y = st.session_state.X, st.session_state.y
        st.markdown("<div class='section-header'>Model Training</div>",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2], gap="large")

        with col_a:
            st.markdown(f"**Classifier:** {clf_name}")
            st.markdown(f"**Scaler:** {scaler_name}")
            st.markdown(f"**SMOTE:** {'On' if use_smote else 'Off'}")
            st.markdown(f"**CV Folds:** {n_cv}")
            st.markdown(f"**Samples:** {len(X):,}  |  "
                        f"Feasible: {y.sum()}  Infeasible: {(y==0).sum()}")

        with col_b:
            train_btn = st.button("🚀  Train & Evaluate Selected Pipeline",
                                  use_container_width=True)

        if train_btn:
            with st.spinner("Running cross-validation..."):
                skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
                f1_macros, f1_feas, f1_inf = [], [], []
                scaler = build_scaler(scaler_name, X.shape[1])
                clf    = build_classifier(clf_name, X.shape[1])
                pipe   = Pipeline([("scaler", scaler), ("clf", clf)])

                prog = st.progress(0)
                for fold, (tr, te) in enumerate(skf.split(X, y)):
                    X_tr, X_te = X[tr], X[te]
                    y_tr, y_te = y[tr], y[te]
                    if use_smote:
                        try:
                            X_tr, y_tr = smote(X_tr, y_tr)
                        except Exception:
                            pass
                    pipe.fit(X_tr, y_tr)
                    y_pred = pipe.predict(X_te)
                    f1_macros.append(f1_score(y_te, y_pred, average="macro",  zero_division=0))
                    f1_feas.append(  f1_score(y_te, y_pred, pos_label=1,      zero_division=0))
                    f1_inf.append(   f1_score(y_te, y_pred, pos_label=0,      zero_division=0))
                    prog.progress((fold+1)/n_cv)

                prog.empty()
                # Retrain on all data for prediction tab
                X_all, y_all = X.copy(), y.copy()
                if use_smote:
                    try:
                        X_all, y_all = smote(X_all, y_all)
                    except Exception:
                        pass
                pipe.fit(X_all, y_all)
                st.session_state.trained_pipe = pipe

            st.markdown("---")
            st.markdown("<div class='section-header'>Results</div>",
                        unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            for col, label, vals, color in [
                (r1, "F1 MACRO",      f1_macros, "#00c8ff"),
                (r2, "F1 FEASIBLE",   f1_feas,   "#7fff6b"),
                (r3, "F1 INFEASIBLE", f1_inf,    "#ff6b35"),
            ]:
                col.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value' style='color:{color}'>"
                    f"{np.mean(vals):.3f}</div>"
                    f"<div class='metric-sub'>± {np.std(vals):.3f} std</div>"
                    f"</div>", unsafe_allow_html=True,
                )

            # F1 per fold chart
            fig_fold = go.Figure()
            folds = list(range(1, n_cv+1))
            for vals, name, color in [
                (f1_macros, "F1 Macro",      "#00c8ff"),
                (f1_feas,   "F1 Feasible",   "#7fff6b"),
                (f1_inf,    "F1 Infeasible", "#ff6b35"),
            ]:
                fig_fold.add_trace(go.Scatter(
                    x=folds, y=vals, mode="lines+markers",
                    name=name, line=dict(color=color, width=2),
                    marker=dict(size=8),
                ))
            fig_fold.update_layout(
                title=dict(text="F1 Score per CV Fold", x=0.5, font=dict(size=13)),
                xaxis=dict(title="Fold", color="#4a6a8a", gridcolor="#1e3a5f",
                           tickvals=folds),
                yaxis=dict(title="F1", range=[0,1.05], color="#4a6a8a",
                           gridcolor="#1e3a5f"),
                **PLOT_THEME, height=320, margin=dict(l=50,r=20,t=50,b=40),
                legend=dict(bgcolor="rgba(0,0,0,0.4)"),
            )
            st.plotly_chart(fig_fold, use_container_width=True)

        # ── Full pipeline comparison ───────────────────────────────────────────
        if run_all:
            st.markdown("---")
            st.markdown("<div class='section-header'>Full Pipeline Comparison (63 pipelines)</div>",
                        unsafe_allow_html=True)

            if st.button("▶  Run All 63 Pipelines", use_container_width=True):
                from pipeline_comparison import build_pipelines, evaluate_pipeline
                pipelines = build_pipelines(X.shape[1])
                results   = []
                n_total   = len(pipelines)
                prog      = st.progress(0)
                status    = st.empty()

                for i, (name, (pipe_i, samp_fn)) in enumerate(pipelines.items()):
                    status.markdown(f"`[{i+1}/{n_total}]` Running **{name}**...")
                    try:
                        res = evaluate_pipeline(name, pipe_i, samp_fn, X, y, n_cv)
                        results.append(res)
                    except Exception as e:
                        results.append({"Pipeline": name, "F1 Macro": 0,
                                        "F1 Macro Std": 0, "F1 Feasible": 0,
                                        "F1 Infeasible": 0, "Fit Time (s)": 0})
                    prog.progress((i+1)/n_total)

                status.empty(); prog.empty()
                df_res = (pd.DataFrame(results)
                            .sort_values("F1 Macro", ascending=False)
                            .reset_index(drop=True))
                df_res.index += 1
                st.session_state.pipeline_results = df_res
                st.success(f"✅ {len(df_res)} pipelines evaluated")

        if st.session_state.pipeline_results is not None:
            df_res = st.session_state.pipeline_results
            pc1, pc2 = st.columns(2, gap="large")
            with pc1:
                st.plotly_chart(make_results_bar(df_res), use_container_width=True)
            with pc2:
                st.plotly_chart(make_scatter_f1(df_res), use_container_width=True)

            st.markdown("**Full Results Table**")
            st.dataframe(
                df_res[["Pipeline","F1 Macro","F1 Macro Std",
                         "F1 Feasible","F1 Infeasible","Fit Time (s)"]
                       ].style.background_gradient(subset=["F1 Macro"],
                                                    cmap="Blues"),
                use_container_width=True, height=400,
            )
            csv = df_res.to_csv(index=True).encode("utf-8")
            st.download_button("⬇  Download Results CSV", csv,
                               "pipeline_results.csv", "text/csv")


# ── TAB 4: Predict New Part ────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Predict a New Part</div>",
                unsafe_allow_html=True)

    if st.session_state.trained_pipe is None:
        st.info("Train a model in the **Train & Evaluate** tab first.")
    else:
        st.markdown(
            "Upload a single `.ply` file to classify it as feasible or infeasible "
            "using the trained model."
        )
        new_file = st.file_uploader("Upload new .ply file for prediction",
                                    type=["ply"], key="predict_upload")

        if new_file is not None:
            with st.spinner("Processing..."):
                pts = load_ply_ascii(new_file.read())
                pts_trimmed = pts[pts[:, 2] > z_cutoff]

                if len(pts_trimmed) < 10:
                    st.error("Too few points after applying Z cutoff. "
                             "Try lowering the cutoff value in the sidebar.")
                else:
                    feats = extract_features(pts_trimmed).reshape(1, -1)
                    pipe  = st.session_state.trained_pipe
                    pred  = pipe.predict(feats)[0]

                    try:
                        proba = pipe.predict_proba(feats)[0]
                        conf_feasible   = proba[1] * 100
                        conf_infeasible = proba[0] * 100
                        has_proba = True
                    except Exception:
                        has_proba = False

                    st.markdown("---")
                    pred_col, viz_col = st.columns([1, 2], gap="large")

                    with pred_col:
                        if pred == 1:
                            st.markdown(
                                "<div class='status-feasible'>"
                                "✅ FEASIBLE<br>"
                                "<span style='font-size:0.75rem;opacity:0.7'>"
                                "Part meets design requirements</span>"
                                "</div>", unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='status-infeasible'>"
                                "❌ INFEASIBLE<br>"
                                "<span style='font-size:0.75rem;opacity:0.7'>"
                                "Part does not meet requirements</span>"
                                "</div>", unsafe_allow_html=True,
                            )

                        st.markdown("<br>", unsafe_allow_html=True)

                        if has_proba:
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<div class='metric-label'>Confidence</div>"
                                f"<div class='metric-value' style='color:#7fff6b'>"
                                f"{conf_feasible:.1f}%</div>"
                                f"<div class='metric-sub'>feasible probability</div>"
                                f"</div>", unsafe_allow_html=True,
                            )
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<div class='metric-label'>Infeasible Prob</div>"
                                f"<div class='metric-value' style='color:#ff6b35'>"
                                f"{conf_infeasible:.1f}%</div>"
                                f"<div class='metric-sub'>infeasible probability</div>"
                                f"</div>", unsafe_allow_html=True,
                            )

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div style='font-family:Share Tech Mono;font-size:0.7rem;"
                            f"color:#4a6a8a;line-height:2'>"
                            f"FILE: {new_file.name}<br>"
                            f"POINTS (raw): {len(pts):,}<br>"
                            f"POINTS (trimmed): {len(pts_trimmed):,}<br>"
                            f"MODEL: {clf_name}<br>"
                            f"SCALER: {scaler_name}"
                            f"</div>", unsafe_allow_html=True,
                        )

                    with viz_col:
                        st.plotly_chart(
                            make_3d_scatter(pts_trimmed,
                                            title=f"'{new_file.name}' — "
                                                  f"{'FEASIBLE ✅' if pred==1 else 'INFEASIBLE ❌'}"),
                            use_container_width=True,
                        )

                    # Confidence gauge
                    if has_proba:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=conf_feasible,
                            title=dict(text="Feasibility Confidence (%)",
                                       font=dict(color="#c8d8e8", size=13)),
                            gauge=dict(
                                axis=dict(range=[0, 100], tickcolor="#4a6a8a"),
                                bar=dict(color="#00c8ff"),
                                bgcolor="rgb(13,21,37)",
                                bordercolor="#1e3a5f",
                                steps=[
                                    dict(range=[0, 40],  color="#2e0d0d"),
                                    dict(range=[40, 60], color="#1a1a0d"),
                                    dict(range=[60, 100], color="#0d2e1a"),
                                ],
                                threshold=dict(
                                    line=dict(color="#ff6b35", width=2),
                                    thickness=0.75, value=50,
                                ),
                            ),
                            number=dict(suffix="%", font=dict(color="#00c8ff", size=36)),
                        ))
                        fig_gauge.update_layout(
                            **PLOT_THEME, height=280,
                            margin=dict(l=30,r=30,t=30,b=10),
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
