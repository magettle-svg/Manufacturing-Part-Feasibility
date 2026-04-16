# Manufacturing Part Feasibility — Point Cloud Classifier

A machine learning pipeline that classifies 3D-scanned manufacturing parts as **feasible** or **infeasible** using point cloud data (`.ply` files). Includes a full pipeline comparison system (63 pipelines) and an interactive Streamlit dashboard.

---

## Project Structure

```
├── pipeline_comparison.py   # Standalone script — runs all 63 pipelines and saves results
├── app.py                   # Streamlit dashboard (coming soon)
├── requirements.txt         # Python dependencies
├── pipeline_results.csv     # Output — generated after running pipeline_comparison.py
├── pipeline_results.html    # Output — interactive Plotly charts
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your data paths

Open `pipeline_comparison.py` and update the paths at the top of the file:

```python
FEASIBLE_DIR   = r"C:\path\to\your\feasible\designs"
INFEASIBLE_DIR = r"C:\path\to\your\infeasible\designs"
Z_CUTOFF       = -477    # strip base/fixture layer below this Z value
```

### 5. Run the pipeline comparison

```bash
python pipeline_comparison.py
```

This will:
- Load and extract features from all `.ply` files
- Run 63 classification pipelines with 5-fold cross-validation
- Print live progress and ETA to the terminal
- Save `pipeline_results.csv` and `pipeline_results.html`

Expected runtime: **15–40 minutes** depending on your CPU.

### 6. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Data Format

The pipeline expects ASCII `.ply` files with `x y z` double-precision coordinates:

```
ply
format ascii 1.0
element vertex 270524
property double x
property double y
property double z
end_header
80.943 -79.681 -509.998
...
```

Folder structure expected:

```
feasible/
  ├── design_001.ply
  ├── design_002.ply
  └── ...          (300 files)

infeasible/
  ├── design_001.ply
  ├── design_002.ply
  └── ...          (200 files)
```

---

## Feature Engineering

Each point cloud is converted into a **439-dimensional feature vector** across 9 groups:

| Group | Features | Description |
|---|---|---|
| A | 154 | Bounding box, extent, centroid, std, percentiles, XY/Z histograms |
| B | 11 | PCA shape descriptors — linearity, planarity, sphericity, eigenvalues |
| C | 29 | Z-slice density profile + gradient |
| D | 5 | Local surface curvature (neighborhood PCA) |
| E | 5 | Convex hull — volume, area, compactness, solidity |
| F | 24 | Radial distribution from centroid axis |
| G | 6 | Statistical moments — skewness and kurtosis per axis |
| H | 5 | Local point density (mean nearest-neighbor distances) |
| I | 200 | Cross-section histograms — XZ and YZ projections |

---

## Pipeline System

63 pipelines are evaluated across 10 groups, combining:

- **6 scalers** — StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
- **5 samplers** — SMOTE, ADASYN, Undersample, SMOTE+Tomek, None
- **4 reducers** — PCA(50), PCA(100), SelectKBest(50), None
- **23 classifiers** — RandomForest, ExtraTrees, GradientBoosting, LogisticRegression, SVM, KNN, GaussianNB, DecisionTree, LDA, QDA, MLP (1/2/3 layer), AdaBoost, SGD, Ridge, Bagging, VotingEnsemble, StackingEnsemble

| Group | # Pipelines | What varies |
|---|---|---|
| 01 Baseline | 6 | Scaler only, no sampling |
| 02 SMOTE + Std | 14 | Classifier |
| 03 Vary Sampler | 5 | Sampler strategy |
| 04 PCA50 | 10 | Classifier after PCA |
| 05 SelectK | 5 | Classifier after SelectKBest |
| 06 Power/Quantile | 6 | Non-linear scalers |
| 07 ADASYN | 5 | Classifier under ADASYN |
| 08 SMOTE+Tomek | 5 | MLP depths + RF/ET |
| 09 Ensembles | 3 | Voting and Stacking |
| 10 Undersample | 5 | Classifier under undersampling |

All samplers (SMOTE, ADASYN, SMOTE+Tomek, Undersample) are implemented from scratch using only `numpy` and `sklearn` — no `imbalanced-learn` required.

---

## Class Imbalance

The dataset has a 60/40 imbalance (300 feasible / 200 infeasible). Key mitigations applied:

- `class_weight='balanced'` in all classifiers that support it
- SMOTE oversampling of the minority class inside each CV fold
- F1 Macro and F1 Infeasible as primary metrics (not accuracy)
- Sampling applied **inside** each cross-validation fold to prevent data leakage

---

## Outputs

| File | Description |
|---|---|
| `pipeline_results.csv` | Full ranked results — F1 Macro, F1 Feasible, F1 Infeasible, fit time |
| `pipeline_results.html` | 4 interactive Plotly charts — top 20 bar, scatter, efficiency frontier, box plot |

---

## Requirements

- Python 3.9+
- See `requirements.txt`

No GPU required. No `open3d` or `imbalanced-learn` required.

---

## Streamlit Deployment

To deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and set **Main file** to `app.py`
5. Click **Deploy**

Streamlit Cloud will automatically install dependencies from `requirements.txt`.

> **Note:** Point cloud files are not included in this repository due to size. When deploying to Streamlit Cloud, upload your `.ply` files via the app's file uploader or connect to cloud storage (S3, Google Drive, etc.).

---

## License

MIT
