import streamlit as st
import numpy as np
import tempfile

# import your existing functions
from pipeline_comparison import extract_features, evaluate_pipeline, build_pipelines, load_ply_ascii

st.title("Point Cloud Feasibility Pipeline Comparison")

# ── Upload section ─────────────────────────────────────────
st.header("Upload Data")

feasible_files = st.file_uploader(
    "Upload FEASIBLE .ply files",
    type=["ply"],
    accept_multiple_files=True
)

infeasible_files = st.file_uploader(
    "Upload INFEASIBLE .ply files",
    type=["ply"],
    accept_multiple_files=True
)

run_button = st.button("Run Pipeline Comparison")

# ── Helper to read uploaded files ──────────────────────────
def process_uploaded(files, label, z_cutoff=None):
    features, labels = [], []

    for uploaded_file in files:
        try:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            pts = load_ply_ascii(path)

            if z_cutoff is not None:
                pts = pts[pts[:, 2] > z_cutoff]

            if len(pts) < 10:
                continue

            features.append(extract_features(pts))
            labels.append(label)

        except Exception as e:
            st.warning(f"Skipped file: {uploaded_file.name} ({e})")

    return features, labels


# ── Run pipelines ──────────────────────────────────────────
if run_button:

    if not feasible_files or not infeasible_files:
        st.error("Please upload both feasible and infeasible files.")
        st.stop()

    st.write("Processing files...")

    ff, fl = process_uploaded(feasible_files, label=1)
    inf_f, il = process_uploaded(infeasible_files, label=0)

    X = np.nan_to_num(
        np.array(ff + inf_f, dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    y = np.array(fl + il, dtype=np.int32)

    st.write(f"Dataset shape: {X.shape}")

    # Build pipelines
    pipelines = build_pipelines(X.shape[1])

    results = []

    progress = st.progress(0)
    total = len(pipelines)

    for i, (name, (pipe, sampler_fn)) in enumerate(pipelines.items()):
        result = evaluate_pipeline(name, pipe, sampler_fn, X, y)
        results.append(result)

        progress.progress((i + 1) / total)

    import pandas as pd
    df = pd.DataFrame(results).sort_values("F1 Macro", ascending=False)

    st.success("Done!")

    st.dataframe(df)

    # Optional: plot top 10
    st.bar_chart(df.head(10).set_index("Pipeline")["F1 Macro"])