"""
DEM Correction via Mathematical Optimization Models
=====================================================
Term Project: DEM Correction (Sentinel-1/2 + ICESat/LiDAR reference elevation)
Assignment  : Mathematics for Data Science

Same objective function, five independent mathematical routes to the solution:
    1. Normal Equations        (Linear Algebra)
    2. SVD-based Least Squares (Linear Algebra / numerical stability)
    3. PCA Regression          (Dimensionality reduction)
    4. Gradient Descent        (First-order iterative optimization)
    5. Newton's Method         (Second-order iterative optimization)

No ML libraries used (no sklearn / statsmodels). Pure NumPy linear algebra.

Author : KARTHIKEYAN R 
Data   : integrated_dataset.parquet  (Sentinel-2 optical + terrain-derived features,
         ICESat/LiDAR elevation_ref as ground truth, raw_z as raw DEM elevation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import gc
from scipy.spatial import cKDTree

np.random.seed(42)
OUT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# =====================================================================
# 1. DATA LOADING AND CLEANING
# =====================================================================

FEATURES = [
    "slope_deg", "aspect_sin", "aspect_cos", "tri", "tpi",
    "planform_curvature", "profile_curvature", "hillshade", "ndvi", "bsi",
]
TARGET = "elevation_error_signed"   # e_i = raw_z - elevation_ref (bias-corrected DEM error)

# Only these columns are ever used anywhere in this script. The source
# parquet has 56 columns; reading all of them for 1.2M rows can exhaust
# memory on machines with limited RAM. Restricting the read to just what's
# needed cuts memory usage roughly 3x (56 cols -> 17 cols).
NEEDED_COLUMNS = list(dict.fromkeys(
    ["latitude", "longitude", "raw_z", "elevation_ref", "is_void_flag", "year_matched"]
    + FEATURES + [TARGET]
))


def load_and_clean(path):
    """
    Load integrated DEM correction dataset and remove satellite DEM
    'void' / no-data sentinel pixels (raw_z == -32768, SRTM/ASTER convention)
    which otherwise corrupt the least-squares fit with values many orders
    of magnitude larger than real elevation error.

    Only reads NEEDED_COLUMNS from the parquet file (not all 56 columns)
    to keep memory usage low on machines with limited RAM.
    """
    try:
        df = pd.read_parquet(path, columns=NEEDED_COLUMNS)
    except MemoryError as e:
        raise MemoryError(
            "Ran out of memory reading the parquet file even with a reduced "
            "column set. Try: (1) close other applications, (2) upgrade "
            "pyarrow (`pip install --upgrade pyarrow`), (3) use the "
            "chunked-loading fallback -- see load_and_clean_chunked() below."
        ) from e

    n_before = len(df)

    sentinel_mask = df["raw_z"] <= -1000          # nodata sentinel values
    void_mask = df["is_void_flag"] == 1           # explicit void flag

    df = df[~sentinel_mask & ~void_mask].copy()
    n_after = len(df)
    print(f"[clean] rows before={n_before:,}  after={n_after:,}  "
          f"removed={n_before - n_after:,} void/sentinel pixels")
    return df


def load_and_clean_chunked(path, row_group_batch=1):
    """
    Fallback loader for very memory-constrained machines: reads the parquet
    file one row-group at a time via pyarrow's low-level API instead of
    loading the whole file into memory at once, cleaning and discarding
    unneeded columns/rows as it goes. Slower, but has a much lower peak
    memory footprint. Use this if load_and_clean() still raises MemoryError.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    chunks = []
    n_before_total = 0
    for batch in pf.iter_batches(columns=NEEDED_COLUMNS, batch_size=200_000):
        chunk = batch.to_pandas()
        n_before_total += len(chunk)
        sentinel_mask = chunk["raw_z"] <= -1000
        void_mask = chunk["is_void_flag"] == 1
        chunk = chunk[~sentinel_mask & ~void_mask]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    n_after = len(df)
    print(f"[clean, chunked] rows before={n_before_total:,}  after={n_after:,}  "
          f"removed={n_before_total - n_after:,} void/sentinel pixels")
    return df


def build_design_matrix(df, feature_means=None, feature_stds=None):
    """
    Build standardized design matrix X = [1, x1_std, x2_std, ..., xp_std]
    Standardization (z-score) is required so that:
      - gradient descent step sizes behave sensibly across features of very
        different natural scales (degrees, NDVI [-1,1], curvature, etc.)
      - condition number of X^T X is not dominated by scale differences
    """
    # float32 (not float64) to roughly halve memory use on large datasets --
    # more than sufficient precision for this regression (values are O(1-10)).
    X_raw = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)

    if feature_means is None:
        feature_means = X_raw.mean(axis=0)
        feature_stds = X_raw.std(axis=0)
        feature_stds[feature_stds == 0] = 1.0

    X_std = (X_raw - feature_means) / feature_stds
    X = np.column_stack([np.ones(len(df), dtype=np.float32), X_std])   # bias column
    return X, y, feature_means, feature_stds


# =====================================================================
# 2. OBJECTIVE FUNCTION AND DERIVATIVES
# =====================================================================
# Model:      e_hat_i = theta^T x_i
# Loss:       J(theta) = (1/N) * ||X theta - y||^2               (Mean Squared Error)
# Gradient:   grad J   = (2/N) * X^T (X theta - y)
# Hessian:    H        = (2/N) * X^T X                            (constant, PSD -> convex)

def loss(X, y, theta):
    r = X @ theta - y
    return float(np.mean(r ** 2))


def grad(X, y, theta):
    N = X.shape[0]
    return (2.0 / N) * (X.T @ (X @ theta - y))


def hessian(X):
    N = X.shape[0]
    return (2.0 / N) * (X.T @ X)


def rmse(X, y, theta):
    return float(np.sqrt(np.mean((X @ theta - y) ** 2)))


def mae(X, y, theta):
    return float(np.mean(np.abs(X @ theta - y)))


def r2_score(X, y, theta):
    pred = X @ theta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


# =====================================================================
# 3. METHOD 1 -- NORMAL EQUATIONS  (theta* = (X^T X)^-1 X^T y)
# =====================================================================

def solve_normal_equations(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    theta = np.linalg.solve(XtX, Xty)     # solve, not explicit inverse (more stable)
    cond_number = np.linalg.cond(XtX)
    return theta, cond_number


# =====================================================================
# 4. METHOD 2 -- SVD-BASED LEAST SQUARES  (theta* = V Sigma^-1 U^T y)
# =====================================================================

def solve_svd_least_squares(X, y):
    # Uses SciPy's 'gesvd' LAPACK driver rather than NumPy's default
    # ('gesdd'). gesdd (divide-and-conquer) occasionally fails its internal
    # workspace initialization on some Windows/LAPACK builds -- it prints
    # "init_gesdd failed init" and silently returns an incorrect result
    # instead of raising an exception. gesvd (classic QR-based SVD) is
    # slower but far more numerically robust; for a matrix this shape
    # (N rows x ~11 columns) the extra cost is negligible.
    from scipy.linalg import svd as scipy_svd
    U, S, Vt = scipy_svd(X, full_matrices=False, lapack_driver="gesvd")
    S_inv = np.diag(1.0 / S)
    theta = Vt.T @ S_inv @ (U.T @ y)
    singular_value_ratio = S.max() / S.min()   # sqrt of cond(X^T X)
    return theta, singular_value_ratio, S


# =====================================================================
# 5. METHOD 3 -- PCA REGRESSION
#    Reduce correlated terrain/optical features to k principal components,
#    regress error on the PC scores, then map coefficients back to the
#    original feature space:  theta_original = P_k @ theta_pca
# =====================================================================

def solve_pca_regression(X, y, k):
    X_features = X[:, 1:]               # drop bias column, PCA on features only
    N = X_features.shape[0]

    cov = (X_features.T @ X_features) / N
    eigvals, eigvecs = np.linalg.eigh(cov)         # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    explained_variance_ratio = eigvals / eigvals.sum()

    P_k = eigvecs[:, :k]                 # top-k principal directions (p x k)
    Z = X_features @ P_k                 # projected scores (N x k)
    Z_design = np.column_stack([np.ones(N), Z])

    theta_pca, _ = solve_normal_equations(Z_design, y)   # bias + k PCA coeffs

    # map back: e_hat = b0 + (X_features @ P_k) @ theta_pca[1:]
    #                 = b0 + X_features @ (P_k @ theta_pca[1:])
    theta_original = np.zeros(X.shape[1])
    theta_original[0] = theta_pca[0]
    theta_original[1:] = P_k @ theta_pca[1:]

    return theta_original, explained_variance_ratio, eigvals


# =====================================================================
# 6. METHOD 4 -- GRADIENT DESCENT (first-order)
#    theta_{t+1} = theta_t - alpha * grad J(theta_t)
#    Step size alpha = 1 / L where L = largest eigenvalue of Hessian
#    (guarantees monotonic decrease for this convex quadratic loss)
# =====================================================================

def solve_gradient_descent(X, y, max_iter=300, tol=1e-10):
    H = hessian(X)
    L = np.linalg.eigvalsh(H).max()      # Lipschitz constant of the gradient
    alpha = 1.0 / L

    theta = np.zeros(X.shape[1])
    history = []
    for t in range(max_iter):
        J = loss(X, y, theta)
        history.append(J)
        g = grad(X, y, theta)
        theta_new = theta - alpha * g
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            history.append(loss(X, y, theta))
            break
        theta = theta_new
    return theta, history, alpha


# =====================================================================
# 7. METHOD 5 -- NEWTON'S METHOD (second-order)
#    theta_{t+1} = theta_t - H^-1 grad J(theta_t)
#    Because J is an exact quadratic, H is CONSTANT -> Newton's method
#    reaches the exact minimum in a single step (quadratic convergence
#    degenerates to one-shot convergence here). We run several iterations
#    to demonstrate this numerically.
# =====================================================================

def solve_newton(X, y, max_iter=5):
    H = hessian(X)
    H_inv = np.linalg.inv(H)

    theta = np.zeros(X.shape[1])
    history = [loss(X, y, theta)]
    for t in range(max_iter):
        g = grad(X, y, theta)
        theta = theta - H_inv @ g
        history.append(loss(X, y, theta))
    return theta, history


# =====================================================================
# 8. CURVE FITTING DEMO -- error vs slope (all other features held at mean)
# =====================================================================

def error_vs_slope_curve(theta, feature_means, feature_stds, n_points=200):
    slope_raw = np.linspace(0, 10, n_points)     # degrees
    slope_std = (slope_raw - feature_means[0]) / feature_stds[0]

    X_curve = np.tile(np.zeros(len(FEATURES)), (n_points, 1))  # all other feats at mean (=0 after std)
    X_curve[:, 0] = slope_std
    X_design = np.column_stack([np.ones(n_points), X_curve])
    e_hat = X_design @ theta
    return slope_raw, e_hat


# =====================================================================
# 9. CLASSICAL SPATIAL INTERPOLATION BASELINES -- IDW and k-NN Averaging
# =====================================================================
# These reproduce, at full dataset scale, the classical DEM-correction
# baselines derived by hand in the reference document "Mathematical
# Modeling and Residual Learning for Terrain-Aware Correction of
# Satellite-Derived DEMs": Inverse Distance Weighting and (unweighted)
# neighbor averaging -- note the reference document's own "Bilinear"
# worked example is itself a simple unweighted 4-neighbor mean
# (Z_Bilinear = (4656+4593+4692+4676)/4), not a true bilinear-weight
# interpolation -- so a k=4 unweighted k-NN mean reproduces that
# baseline faithfully.
#
# Both baselines are SPATIAL ONLY: they interpolate the known elevation
# error at nearby training pixels using geographic distance, with no
# access to terrain/optical features. This is exactly the limitation
# the reference document identifies mathematically: spatial-smoothness
# assumptions cannot capture sharp, terrain-driven bias (ridges, slope
# breaks), which is precisely why the feature-based regression methods
# (Methods 1-5) are expected to outperform them.

def _latlon_to_local_meters(lat, lon, lat0):
    """
    Equirectangular projection to local planar meters. Valid for the
    small (a few-degree) study-area extent used here; matches the
    reference document's approach of converting degrees to meters via
    a fixed latitude scale factor (h = 0.000833 deg x 111320 m/deg).
    """
    R = 6371000.0
    x = np.radians(lon) * R * np.cos(np.radians(lat0))
    y = np.radians(lat) * R
    return x, y


def build_spatial_index(train_df):
    lat0 = float(train_df["latitude"].mean())
    x_train, y_train = _latlon_to_local_meters(
        train_df["latitude"].to_numpy(), train_df["longitude"].to_numpy(), lat0
    )
    tree = cKDTree(np.column_stack([x_train, y_train]))
    train_target = train_df[TARGET].to_numpy()
    return tree, train_target, lat0


def solve_idw_baseline(tree, train_target, test_df, lat0, k=8, power=2):
    """Inverse Distance Weighting: Z_IDW(x) = [sum(Z_i / d_i^p)] / [sum(1/d_i^p)]"""
    x_test, y_test = _latlon_to_local_meters(
        test_df["latitude"].to_numpy(), test_df["longitude"].to_numpy(), lat0
    )
    dist, idx = tree.query(np.column_stack([x_test, y_test]), k=k)
    dist = np.maximum(dist, 1e-3)   # guard against zero distance (coincident point)
    neighbor_vals = train_target[idx]
    weights = 1.0 / (dist ** power)
    e_hat = np.sum(neighbor_vals * weights, axis=1) / np.sum(weights, axis=1)
    return e_hat


def solve_knn_mean_baseline(tree, train_target, test_df, lat0, k=4):
    """Unweighted k-neighbor mean -- reproduces the reference document's
    'Bilinear' worked example exactly (simple average of nearest neighbors)."""
    x_test, y_test = _latlon_to_local_meters(
        test_df["latitude"].to_numpy(), test_df["longitude"].to_numpy(), lat0
    )
    _, idx = tree.query(np.column_stack([x_test, y_test]), k=k)
    neighbor_vals = train_target[idx]
    return neighbor_vals.mean(axis=1)


def rmse_arr(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_arr(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_arr(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def le90_formula(residuals):
    """LE90 = 1.6449 x RMSE, the standard linear-error-at-90%-confidence
    formula (assumes approximately normal error distribution) -- used in
    the reference document and standard in USGS/NGA DEM accuracy reporting."""
    return float(1.6449 * np.sqrt(np.mean(residuals ** 2)))


def le90_empirical(residuals):
    """Empirical LE90: the 90th percentile of |residual|, with no
    normality assumption -- reported alongside the formula-based value."""
    return float(np.percentile(np.abs(residuals), 90))


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    DATA_PATH = r"D:\DEM-CORRECTION-Maths-Modelling\integrated_dataset.parquet"
    try:
        df = load_and_clean(DATA_PATH)
    except MemoryError:
        print("[warning] standard load failed on memory; retrying with "
              "chunked low-memory loader (slower, lower peak RAM)...")
        df = load_and_clean_chunked(DATA_PATH)

    train_df = df[df["year_matched"] == 2022]
    test_df = df[df["year_matched"] == 2023]
    print(f"[split] train={len(train_df):,}  test={len(test_df):,}")

    X_train, y_train, mu, sigma = build_design_matrix(train_df)
    X_test, y_test, _, _ = build_design_matrix(test_df, mu, sigma)

    results = {}

    # ---- Method 1: Normal Equations ----
    theta_ne, cond_XtX = solve_normal_equations(X_train, y_train)
    results["Normal Equations"] = dict(
        theta=theta_ne,
        train_rmse=rmse(X_train, y_train, theta_ne),
        test_rmse=rmse(X_test, y_test, theta_ne),
        train_r2=r2_score(X_train, y_train, theta_ne),
        cond_number=cond_XtX,
    )

    # ---- Method 2: SVD Least Squares ----
    theta_svd, sv_ratio, singular_values = solve_svd_least_squares(X_train, y_train)
    results["SVD Least Squares"] = dict(
        theta=theta_svd,
        train_rmse=rmse(X_train, y_train, theta_svd),
        test_rmse=rmse(X_test, y_test, theta_svd),
        train_r2=r2_score(X_train, y_train, theta_svd),
        singular_value_ratio=sv_ratio,
    )

    # ---- Method 3: PCA Regression (k=6 of 10 features) ----
    k = 6
    theta_pca, explained_var, eigvals = solve_pca_regression(X_train, y_train, k)
    results["PCA Regression (k=6)"] = dict(
        theta=theta_pca,
        train_rmse=rmse(X_train, y_train, theta_pca),
        test_rmse=rmse(X_test, y_test, theta_pca),
        train_r2=r2_score(X_train, y_train, theta_pca),
        explained_variance=float(explained_var[:k].sum()),
    )

    # ---- Method 4: Gradient Descent ----
    theta_gd, gd_history, alpha_gd = solve_gradient_descent(X_train, y_train, max_iter=300)
    results["Gradient Descent"] = dict(
        theta=theta_gd,
        train_rmse=rmse(X_train, y_train, theta_gd),
        test_rmse=rmse(X_test, y_test, theta_gd),
        train_r2=r2_score(X_train, y_train, theta_gd),
        iterations=len(gd_history),
        step_size=alpha_gd,
        final_loss=gd_history[-1],
    )

    # ---- Method 5: Newton's Method ----
    theta_nt, nt_history = solve_newton(X_train, y_train, max_iter=5)
    results["Newton's Method"] = dict(
        theta=theta_nt,
        train_rmse=rmse(X_train, y_train, theta_nt),
        test_rmse=rmse(X_test, y_test, theta_nt),
        train_r2=r2_score(X_train, y_train, theta_nt),
        iterations=len(nt_history) - 1,
        final_loss=nt_history[-1],
    )

    # ---- LE90 for every regression-based method (Methods 1-5) ----
    for name, r in results.items():
        test_residuals = X_test @ r["theta"] - y_test
        r["le90_formula"] = le90_formula(test_residuals)
        r["le90_empirical"] = le90_empirical(test_residuals)
        r["test_mae"] = mae_arr(y_test, X_test @ r["theta"])

    # ---- Classical spatial interpolation baselines (IDW, k-NN mean) ----
    # Reproduces, at full dataset scale, the hand-derived baselines from the
    # reference document "Mathematical Modeling and Residual Learning for
    # Terrain-Aware Correction of Satellite-Derived DEMs": these interpolate
    # the elevation error spatially (lat/lon only) with NO terrain/optical
    # features, unlike Methods 1-5.
    print("\n[baselines] building spatial index for IDW / k-NN baselines...")
    tree, train_target, lat0 = build_spatial_index(train_df)

    # Diagnostic: how close is each 2023 test pixel to its nearest 2018-2022
    # training pixel? This quantifies the spatial-leakage explanation for why
    # spatial baselines may outperform feature-based regression (see write-up).
    x_test_diag, y_test_diag = _latlon_to_local_meters(
        test_df["latitude"].to_numpy(), test_df["longitude"].to_numpy(), lat0
    )
    nn_dist, _ = tree.query(np.column_stack([x_test_diag, y_test_diag]), k=1)
    frac_within_pixel = float((nn_dist < 93).mean())
    frac_within_10m = float((nn_dist < 10).mean())
    print(f"[diagnostic] nearest-neighbor distance (test->train): "
          f"median={np.median(nn_dist):.1f} m, "
          f"{frac_within_10m*100:.1f}% within 10 m, "
          f"{frac_within_pixel*100:.1f}% within 1 DEM pixel (93 m)")

    e_hat_idw = solve_idw_baseline(tree, train_target, test_df, lat0, k=8, power=2)
    idw_residuals = e_hat_idw - y_test
    results["IDW (k=8, spatial)"] = dict(
        theta=None,
        train_rmse=None,
        test_rmse=rmse_arr(y_test, e_hat_idw),
        test_mae=mae_arr(y_test, e_hat_idw),
        train_r2=None,
        test_r2=r2_arr(y_test, e_hat_idw),
        iterations=1,
        le90_formula=le90_formula(idw_residuals),
        le90_empirical=le90_empirical(idw_residuals),
    )

    e_hat_knn = solve_knn_mean_baseline(tree, train_target, test_df, lat0, k=4)
    knn_residuals = e_hat_knn - y_test
    results["k-NN Mean, k=4 (Bilinear-style)"] = dict(
        theta=None,
        train_rmse=None,
        test_rmse=rmse_arr(y_test, e_hat_knn),
        test_mae=mae_arr(y_test, e_hat_knn),
        train_r2=None,
        test_r2=r2_arr(y_test, e_hat_knn),
        iterations=1,
        le90_formula=le90_formula(knn_residuals),
        le90_empirical=le90_empirical(knn_residuals),
    )
    print(f"[baselines] IDW test RMSE={results['IDW (k=8, spatial)']['test_rmse']:.4f} m   "
          f"k-NN mean test RMSE={results['k-NN Mean, k=4 (Bilinear-style)']['test_rmse']:.4f} m")

    # =========== PRINT SUMMARY TABLE ===========
    print("\n" + "=" * 100)
    print(f"{'Method':<30}{'Test RMSE':>11}{'Test MAE':>10}{'LE90':>9}{'Iterations':>12}")
    print("-" * 100)
    for name, r in results.items():
        it = r.get("iterations", 1)
        mae_v = r.get("test_mae", float("nan"))
        le90_v = r.get("le90_formula", float("nan"))
        print(f"{name:<30}{r['test_rmse']:>11.4f}{mae_v:>10.4f}{le90_v:>9.4f}{it:>12}")
    print("=" * 100)
    print(f"\ncondition number of X^T X (Normal Equations): {cond_XtX:.2f}")
    print(f"singular value ratio sigma_max/sigma_min (SVD): {sv_ratio:.2f}")
    print(f"(cond(X^T X) should approx equal (sv_ratio)^2 = {sv_ratio**2:.2f})")

    best_math = min(
        (n for n in results if "spatial" not in n and "Bilinear" not in n),
        key=lambda n: results[n]["test_rmse"],
    )
    best_baseline = min(
        ("IDW (k=8, spatial)", "k-NN Mean, k=4 (Bilinear-style)"),
        key=lambda n: results[n]["test_rmse"],
    )
    math_rmse = results[best_math]["test_rmse"]
    baseline_rmse = results[best_baseline]["test_rmse"]
    print(f"\nBest feature-based method  : {best_math}  (RMSE={math_rmse:.4f} m)")
    print(f"Best classical baseline    : {best_baseline}  (RMSE={baseline_rmse:.4f} m)")
    if baseline_rmse < math_rmse:
        pct = (math_rmse - baseline_rmse) / math_rmse * 100
        print(f"NOTE: the spatial baseline is {pct:.2f}% LOWER RMSE than the best feature-based "
              f"regression -- see write-up for the spatial-autocorrelation / spatial-leakage "
              f"explanation (train/test overlap in geography, differ only by year).")
    else:
        pct = (baseline_rmse - math_rmse) / baseline_rmse * 100
        print(f"Feature-based regression improves on the best spatial baseline by {pct:.2f}%.")

    # =========== SAVE RESULTS TABLE ===========
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "method": name,
            "train_rmse": r["train_rmse"] if r["train_rmse"] is not None else float("nan"),
            "test_rmse": r["test_rmse"],
            "test_mae": r.get("test_mae", float("nan")),
            "train_r2": r["train_r2"] if r["train_r2"] is not None else float("nan"),
            "le90_formula": r.get("le90_formula", float("nan")),
            "le90_empirical": r.get("le90_empirical", float("nan")),
            "iterations": r.get("iterations", 1),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "results_summary.csv"), index=False)

    # =========== PLOTS ===========
    # Free large intermediate arrays before entering the plotting stage --
    # this is where the memory pressure showed up before (matplotlib's
    # Agg renderer needs a contiguous pixel-buffer allocation, which fails
    # first if RAM is already tight from the regression/spatial-baseline
    # computations above).
    gc.collect()

    def safe_savefig(path, dpi=150):
        """
        Save the current matplotlib figure, retrying at progressively lower
        DPI if memory is tight, and skipping (with a warning) rather than
        crashing the whole pipeline if every attempt fails. This ensures a
        single low-memory plot failure doesn't cost you the CSV/JSON
        results that were already computed successfully.
        """
        for attempt_dpi in (dpi, 100, 72, 50):
            try:
                plt.savefig(path, dpi=attempt_dpi)
                if attempt_dpi != dpi:
                    print(f"[plots] saved {os.path.basename(path)} at reduced "
                          f"dpi={attempt_dpi} (memory-constrained)")
                return
            except MemoryError:
                gc.collect()
                continue
        print(f"[plots] WARNING: could not save {os.path.basename(path)} "
              f"(out of memory at all DPI levels) -- skipping this plot only.")

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Gradient Descent convergence
    plt.figure(figsize=(7, 5))
    plt.plot(gd_history, color="#2563eb", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss  J(theta)")
    plt.title("Gradient Descent Convergence")
    plt.yscale("log")
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "gd_convergence.png"))
    plt.close()
    gc.collect()

    # 2. Newton's Method convergence
    plt.figure(figsize=(7, 5))
    plt.plot(nt_history, marker="o", color="#dc2626", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss  J(theta)")
    plt.title("Newton's Method Convergence (one-step for quadratic loss)")
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "newton_convergence.png"))
    plt.close()
    gc.collect()

    # 3. GD vs Newton overlay (first 10 iterations)
    plt.figure(figsize=(7, 5))
    plt.plot(gd_history[:15], marker="o", label="Gradient Descent", color="#2563eb")
    plt.plot(nt_history, marker="s", label="Newton's Method", color="#dc2626")
    plt.xlabel("Iteration")
    plt.ylabel("Loss  J(theta)")
    plt.title("Convergence Speed: Gradient Descent vs Newton's Method")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "gd_vs_newton.png"))
    plt.close()
    gc.collect()

    # 4. Method comparison bar chart (test RMSE) -- 5 regression methods + 2 spatial baselines
    plt.figure(figsize=(10, 5.5))
    names = list(results.keys())
    test_rmses = [results[n]["test_rmse"] for n in names]
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#be185d"]
    bar_colors = colors[:len(names)]
    bars = plt.bar(names, test_rmses, color=bar_colors)
    # highlight the spatial baselines with a hatch pattern to visually separate
    # them from the feature-based regression methods
    for i, n in enumerate(names):
        if "spatial" in n or "Bilinear" in n:
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("white")
    plt.ylabel("Test RMSE (m)")
    plt.title("Feature-Based Regression vs. Classical Spatial Baselines (Test RMSE)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "method_comparison_rmse.png"))
    plt.close()
    gc.collect()

    # 4b. LE90 comparison across all methods
    plt.figure(figsize=(10, 5.5))
    le90_vals = [results[n]["le90_formula"] for n in names]
    bars2 = plt.bar(names, le90_vals, color=bar_colors)
    for i, n in enumerate(names):
        if "spatial" in n or "Bilinear" in n:
            bars2[i].set_hatch("//")
            bars2[i].set_edgecolor("white")
    plt.ylabel("LE90 (m)")
    plt.title("Linear Error at 90% Confidence (LE90) Across All Methods")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "le90_comparison.png"))
    plt.close()
    gc.collect()

    # 5. PCA scree plot
    plt.figure(figsize=(7, 5))
    cum_var = np.cumsum(explained_var)
    plt.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.6, label="Individual", color="#2563eb")
    plt.plot(range(1, len(explained_var) + 1), cum_var, marker="o", color="#dc2626", label="Cumulative")
    plt.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot (Terrain + Optical Features)")
    plt.legend()
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "pca_scree.png"))
    plt.close()
    gc.collect()

    # 6. Curve fit: error vs slope
    slope_raw, e_hat_curve = error_vs_slope_curve(theta_ne, mu, sigma)
    sample_idx = np.random.choice(len(train_df), size=5000, replace=False)
    plt.figure(figsize=(7, 5))
    plt.scatter(train_df["slope_deg"].to_numpy()[sample_idx],
                train_df[TARGET].to_numpy()[sample_idx],
                s=4, alpha=0.25, color="#94a3b8", label="Observed error (sample)")
    plt.plot(slope_raw, e_hat_curve, color="#dc2626", linewidth=2.5,
              label="Fitted model (Normal Equations)")
    plt.xlabel("Slope (degrees)")
    plt.ylabel("DEM Elevation Error (m)")
    plt.title("Curve Fit: Elevation Error vs Terrain Slope")
    plt.legend()
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "curve_fit_slope.png"))
    plt.close()
    gc.collect()

    # 7. Actual vs Predicted (best model = Normal Equations, on test set)
    pred_test = X_test @ theta_ne
    sample_idx_test = np.random.choice(len(test_df), size=min(5000, len(test_df)), replace=False)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test[sample_idx_test], pred_test[sample_idx_test], s=4, alpha=0.3, color="#2563eb")
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.plot(lims, lims, color="#dc2626", linewidth=1.5, label="y = x (perfect fit)")
    plt.xlabel("Actual DEM Error (m)")
    plt.ylabel("Predicted DEM Error (m)")
    plt.title("Actual vs Predicted Elevation Error (Test Set, 2023)")
    plt.legend()
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "actual_vs_predicted.png"))
    plt.close()
    gc.collect()

    # 8. Residual histogram
    residuals = y_test - pred_test
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=80, color="#16a34a", alpha=0.75)
    plt.xlabel("Residual (Actual - Predicted), m")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution After Correction (Test Set)")
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "residual_histogram.png"))
    plt.close()
    gc.collect()

    # 9. Spatial-leakage diagnostic: nearest-neighbor distance histogram
    # (test 2023 pixels -> nearest 2018-2022 training pixel)
    plt.figure(figsize=(7, 5))
    plt.hist(np.clip(nn_dist, 0, 1000), bins=80, color="#0891b2", alpha=0.8)
    plt.axvline(93, color="#dc2626", linestyle="--", linewidth=1.5,
                label="1 DEM pixel (93 m)")
    plt.xlabel("Distance to nearest training-set pixel (m, clipped at 1000 m)")
    plt.ylabel("Frequency")
    plt.title(f"Spatial Proximity: Test (2023) to Training (2018-2022) Pixels\n"
              f"{frac_within_pixel*100:.1f}% of test pixels within 1 DEM pixel of a training pixel")
    plt.legend()
    plt.tight_layout()
    safe_savefig(os.path.join(PLOTS, "spatial_leakage_histogram.png"))
    plt.close()
    gc.collect()

    print(f"\n[done] plots saved to: {PLOTS}")
    print(f"[done] results summary saved to: {os.path.join(OUT, 'results_summary.csv')}")

    # save raw numeric results as json for report generation
    json_results = {}
    for name, r in results.items():
        json_results[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in r.items()}
    json_results["gd_history"] = gd_history
    json_results["newton_history"] = nt_history
    json_results["explained_variance_ratio"] = explained_var.tolist()
    json_results["feature_names"] = FEATURES
    json_results["train_size"] = len(train_df)
    json_results["test_size"] = len(test_df)
    json_results["feature_means"] = mu.tolist()
    json_results["feature_stds"] = sigma.tolist()
    json_results["condition_number"] = cond_XtX
    json_results["singular_value_ratio"] = sv_ratio
    json_results["singular_values"] = singular_values.tolist()
    json_results["spatial_leakage"] = {
        "median_nn_distance_m": float(np.median(nn_dist)),
        "frac_within_10m": frac_within_10m,
        "frac_within_1_pixel_93m": frac_within_pixel,
    }
    json_results["best_math_method"] = best_math
    json_results["best_baseline_method"] = best_baseline
    # a light sample of raw (slope, error) pairs for scatter plots in the dashboard
    # (avoids shipping the full multi-GB parquet alongside the presentation layer)
    sample_n = min(8000, len(train_df))
    sample_idx = np.random.choice(len(train_df), size=sample_n, replace=False)
    json_results["scatter_sample"] = {
        "slope_deg": train_df["slope_deg"].to_numpy()[sample_idx].tolist(),
        "elevation_error_signed": train_df[TARGET].to_numpy()[sample_idx].tolist(),
    }
    test_sample_n = min(8000, len(test_df))
    test_sample_idx = np.random.choice(len(test_df), size=test_sample_n, replace=False)
    pred_test_full = X_test @ theta_ne
    json_results["test_scatter_sample"] = {
        "actual": y_test[test_sample_idx].tolist(),
        "predicted": pred_test_full[test_sample_idx].tolist(),
    }
    json_results["residuals_sample"] = (y_test[test_sample_idx] - pred_test_full[test_sample_idx]).tolist()

    def _json_default(o):
        """
        Handles numpy scalar types that Python's json module can't serialize
        natively. np.float64 happens to subclass Python's built-in float (so
        it was serializing fine before), but np.float32 -- used now to halve
        memory footprint -- does NOT subclass float, so it needs explicit
        conversion via .item(). Also covers any numpy array that slipped
        through without an explicit .tolist() call above.
        """
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(json_results, f, indent=2, default=_json_default)

    return results


if __name__ == "__main__":
    main()