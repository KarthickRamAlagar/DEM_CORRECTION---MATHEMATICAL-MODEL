# DEM-CORRECTION — Mathematical Modeling for Terrain-Aware Satellite DEM Correction

Mathematics for Data Science — Assignment cum Term Project. A single convex
least-squares objective, solved five independent mathematical ways (Normal
Equations, SVD, PCA Regression, Gradient Descent, Newton's Method),
benchmarked honestly against classical spatial interpolation baselines
(IDW, k-NN averaging), on **real** satellite-derived elevation data — no
synthetic terrain, no toy examples at deployment scale.

**Framing note:** the mathematical foundation (residual-learning
formulation, finite-difference terrain derivatives, IDW/Bilinear baseline
derivations) follows the companion document *Mathematical Modeling and
Residual Learning for Terrain-Aware Correction of Satellite-Derived DEMs*.
That document works a single 25×25 hand-derived example; this project
implements and validates the same mathematics at full dataset scale
(1.2M+ real pixels) and reports what actually happens when it's run on
real data — including a result that contradicts the small hand-worked
example, discussed honestly rather than smoothed over (see *Results*
below).

**Live app:** https://demcorrection---mathematical-model.streamlit.app/

---

## What's actually built

- **One objective function, five solvers.** DEM elevation error
  $e_i = z_i - z_i^{*}$ is modeled as $\hat e_i = \boldsymbol\theta^\top
  \mathbf{x}_i$ and fit by minimizing MSE — solved via Normal Equations,
  SVD-based least squares, PCA regression (dimensionality-reduced), batch
  Gradient Descent, and Newton's Method. All five converge to the same
  optimum on this data (condition number of $X^\top X$ = 110, well
  conditioned) — implemented in raw NumPy, **no ML libraries** in the core
  assignment (no scikit-learn, no statsmodels).
- **Real data, not synthetic.** Sentinel-2 optical bands (NDVI, NDWI, EVI,
  BSI, raw B02/B03/B04/B08/B11/B12), Sentinel-1 SAR (VV/VH, partially
  available), and terrain-derivative features (slope, aspect, TRI, TPI,
  planform/profile/general curvature, hillshade) against ICESat/LiDAR
  reference elevation. **1,199,340** valid pixels after removing
  satellite DEM void/no-data sentinels, split by **year** —
  **1,147,514** training pixels (2018–2022) and **51,826** held-out test
  pixels (2023) — not a random shuffle, a genuine temporal holdout.
- **Classical spatial baselines, implemented for real.** Inverse Distance
  Weighting ($k{=}8$, geographic distance via a local equirectangular
  projection, `scipy.spatial.cKDTree`) and an unweighted $k{=}4$
  nearest-neighbor mean — the latter reproducing the companion document's
  own worked "Bilinear" example exactly (which is itself a simple
  4-neighbor average, not true bilinear weights, on inspection).
- **An honestly reported, non-obvious finding.** The purely spatial IDW
  baseline beats every feature-based regression method by **29.2%** test
  RMSE — traced to genuine spatial autocorrelation, not a modeling error
  (see *Results* below). Reporting this plainly, instead of only showing
  favorable comparisons, is treated as part of doing the mathematics
  correctly.
- **LE90 and full statistical reporting.** RMSE, MAE, $R^2$, and LE90
  (Linear Error at 90% confidence, $= 1.6449 \times \text{RMSE}$, the
  USGS/NGA DEM-accuracy standard used in the companion document) reported
  for every method, both formula-based and empirical (90th-percentile of
  $|{\text{residual}}|$).
- **A presentation-layer Streamlit dashboard.** Interactive dark-themed
  dashboard (Plotly, not static images) covering the mathematical
  derivations (live LaTeX), method comparison, convergence analysis
  (Gradient Descent vs. Newton's, live), PCA scree plot, an interactive
  correction predictor (slide terrain inputs, see the predicted DEM error
  update live using the trained $\boldsymbol\theta^*$), the spatial
  baseline comparison, and the SDG 11 alignment. This is the one place
  ML-adjacent packages (Plotly) are used — purely for visualization, no
  modeling happens here; every number displayed is precomputed by the raw
  NumPy pipeline.

---

## Mathematical foundation (assignment component)

| # | Method | Category | Update / Solution |
|---|---|---|---|
| 1 | Normal Equations | Linear Algebra | $\boldsymbol\theta^* = (X^\top X)^{-1}X^\top e$ |
| 2 | SVD Least Squares | Matrix Factorization | $\boldsymbol\theta^* = V\Sigma^{-1}U^\top e$ |
| 3 | PCA Regression ($k{=}6$) | Dimensionality Reduction | $\boldsymbol\theta = P_k\,\boldsymbol\theta_{\text{reduced}}$ |
| 4 | Gradient Descent | First-Order Optimization | $\boldsymbol\theta \leftarrow \boldsymbol\theta - \alpha\nabla J(\boldsymbol\theta)$, $\alpha = 1/\lambda_{\max}(H)$ |
| 5 | Newton's Method | Second-Order Optimization | $\boldsymbol\theta \leftarrow \boldsymbol\theta - H^{-1}\nabla J(\boldsymbol\theta)$ |

Since $J(\boldsymbol\theta) = \frac{1}{N}\lVert X\boldsymbol\theta -
\mathbf{e}\rVert_2^2$ is an exact convex quadratic, its Hessian $H =
\frac{2}{N}X^\top X$ is **constant** — Newton's Method reaches the exact
Normal-Equations optimum in **one iteration**, while Gradient Descent
takes on the order of hundreds of iterations at the maximal stable step
size $\alpha = 1/\lambda_{\max}(H)$. Full derivations, iteration logs, and
convergence plots are in
[`DEM_Correction_Mathematical_Models.pdf`](./DEM_Correction_Mathematical_Models.pdf).

---

## Real, validated results

### Feature-based regression (5 methods) vs. classical spatial baselines

| Method | Test RMSE (m) | Test MAE (m) | LE90 (m) | Uses features? | Iterations |
|---|---|---|---|---|---|
| Normal Equations | 2.0679 | 1.4145 | 3.4015 | Yes | 1 (closed-form) |
| SVD Least Squares | 2.0679 | 1.4145 | 3.4015 | Yes | 1 (closed-form) |
| PCA Regression ($k{=}6$) | 2.0824 | 1.4275 | 3.4253 | Yes (reduced, 87.9% variance) | 1 (closed-form) |
| Gradient Descent | 2.0679 | 1.4145 | 3.4015 | Yes | 300 |
| Newton's Method | 2.0679 | 1.4145 | 3.4015 | Yes | **1** |
| **IDW ($k{=}8$, spatial)** | **1.4644** | **0.7233** | **2.4088** | No | 1 |
| k-NN Mean ($k{=}4$, "Bilinear-style") | 1.4849 | 0.7594 | 2.4426 | No | 1 |

All five feature-based methods converge to the **same optimum**
(confirming numerical correctness across every solution route) — but the
purely spatial IDW baseline outperforms all of them.

**An honest, empirically investigated finding, not glossed over:** train
(2018–2022) and test (2023) data cover the *same geographic region*,
split only by year. Since terrain-driven DEM bias is largely
time-invariant, a nearby training pixel from an earlier year is often an
almost-exact preview of a 2023 test pixel's error:

- Median distance from a test pixel to its nearest training pixel:
  **78.3 m**
- **53.2%** of test pixels have a training-set neighbor within **one DEM
  pixel width (93 m)**
- **10.6%** are within **10 m** — effectively the same physical location,
  observed in a different year

IDW exploits this directly; the linear regression models must instead
compress all location-specific bias through a single global 10-parameter
function of terrain/optical features, and cannot memorize per-pixel
history the way a nearest-neighbor spatial method can. This is treated as
a genuine diagnosis, not a failure: it identifies **regression-kriging**
(trend regression + spatial residual correction via IDW) as the concrete,
well-motivated next step —
$$\hat e(\mathbf{s}) = \underbrace{\boldsymbol\theta^\top\mathbf{x}(\mathbf{s})}_{\text{trend}} + \underbrace{\text{IDW}\big(e(\mathbf{s}') - \boldsymbol\theta^\top\mathbf{x}(\mathbf{s}')\big)}_{\text{spatial residual correction}}$$
— unifying every method in this project into one hybrid pipeline, rather
than treating regression and spatial interpolation as competitors.

Full derivation, the spatial-proximity histogram, and this discussion are
in Section 7 of
[`DEM_Correction_Mathematical_Models.pdf`](./DEM_Correction_Mathematical_Models.pdf).

---

## SDG 11 — Sustainable Cities and Communities (term project component)

Corrected elevation data maps directly to SDG 11 targets **11.3**
(inclusive settlement planning — free satellite DEMs are often the only
elevation source available to local authorities), **11.5** (reducing
disaster deaths/losses — flood-extent and landslide-susceptibility maps
depend on elevation accuracy), and **11.b** (integrated disaster-risk
policy, Sendai Framework).

Study region: **25°N–28°N, 80°E–84°E** — the Himalayan foothill/Terai
belt (India–Nepal border zone), a real, recurring monsoon flood/landslide
risk zone with sparse LiDAR coverage and heavy reliance on free satellite
DEMs for planning. Full alignment write-up, including honestly-stated
limitations (linear $R^2 \approx 0.10$, SAR features excluded due to
~90% missing coverage) and future work, is in
[`SDG11_Term_Project_Section.md`](./SDG11_Term_Project_Section.md).

---

## Folder layout (current)

```
DEM-CORRECTION-Maths-Modelling/
├── dem_correction_math_models.py      # full pipeline: load, clean, 5 solvers,
│                                        # IDW/k-NN baselines, LE90, all plots
├── integrated_dataset.parquet          # real Sentinel-1/2 + terrain + ICESat/LiDAR
│                                        # data (gitignored — see Data below)
├── results_summary.csv                 # metrics table, regenerated on each run
├── results.json                        # full numeric results (theta, iteration
│                                        # histories, spatial-leakage diagnostics)
├── plots/                              # 10 PNGs: convergence, PCA scree, curve
│                                        # fit, spatial-leakage histogram, etc.
├── DEM_Correction_Mathematical_Models.pdf   # assignment deliverable — full
│                                              # derivations + iteration logs
├── SDG11_Term_Project_Section.md       # term project SDG 11 alignment write-up
└── streamlit_app/                      # presentation layer (deploy target)
    ├── app.py                          # dark-themed interactive dashboard
    ├── requirements.txt
    ├── results.json                    # dashboard's own copy (see Setup)
    ├── results_summary.csv
    └── .streamlit/
        └── config.toml                 # dark theme config
```

**On `integrated_dataset.parquet`:** not committed to this repo (179 MB —
over GitHub's file-size limits, and raw data doesn't belong in git
regardless). Regenerate the pipeline outputs (`results.json`,
`results_summary.csv`, `plots/`) by running the script against your own
copy of the dataset — see *Setup* below.

---

## Setup

```bash
pip install -r streamlit_app/requirements.txt
pip install scipy pyarrow   # required by the core pipeline script
```

### Run the mathematical pipeline

```bash
python dem_correction_math_models.py
```

Edit the `DATA_PATH` variable near the top of `main()` to point to your
local copy of `integrated_dataset.parquet` first. This regenerates
`results.json`, `results_summary.csv`, and every plot in `plots/`.

### Run the Streamlit dashboard

```bash
cd streamlit_app
streamlit run app.py
```

After regenerating results with a fresh dataset, copy the new
`results.json` and `results_summary.csv` into `streamlit_app/` so the
dashboard reflects them — it reads its own local copy, not the root
folder's, so it can be deployed (e.g. to Streamlit Community Cloud)
without shipping the large source dataset alongside it.

---

## Honest scope notes

- **No ML libraries in the core assignment** — every solver (Normal
  Equations, SVD, PCA, Gradient Descent, Newton's Method) is implemented
  in raw NumPy/SciPy linear algebra, not scikit-learn. Plotly is used
  **only** in the Streamlit presentation layer for interactive charts —
  it does not participate in any modeling.
- **SAR (VV/VH) features excluded** from the regression feature set —
  ~90% missing coverage in the source data made them unusable without a
  separate imputation strategy, which was out of scope here. Left as
  explicit future work rather than silently dropped.
- **Linear $R^2 \approx 0.10$** — the linear feature-based model captures
  only part of the systematic DEM error; a meaningful share is likely
  nonlinear (slope–vegetation interaction effects) or driven by
  sensor-specific artifacts (radar layover/shadow) not present in this
  feature set.
- **The IDW-beats-regression finding is dataset-structure-specific** —
  it follows directly from the year-only train/test split over a fixed
  geographic region, and is explained, not hidden, in both the PDF and
  the dashboard's dedicated "Spatial Baselines" page.
