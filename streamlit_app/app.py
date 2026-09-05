import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG + DARK THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DEM Correction — Mathematical Framework",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1420;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background: linear-gradient(145deg, #131a2b, #0f1420);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #131a2b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#38bdf8"
ACCENT2 = "#f87171"
ACCENT3 = "#34d399"
ACCENT4 = "#fbbf24"
ACCENT5 = "#a78bfa"
METHOD_COLORS = {
    "Normal Equations": ACCENT,
    "SVD Least Squares": ACCENT3,
    "PCA Regression (k=6)": ACCENT4,
    "Gradient Descent": ACCENT2,
    "Newton's Method": ACCENT5,
    "IDW (k=8, spatial)": "#0891b2",
    "k-NN Mean, k=4 (Bilinear-style)": "#be185d",
}
BASELINE_NAMES = {"IDW (k=8, spatial)", "k-NN Mean, k=4 (Bilinear-style)"}

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
@st.cache_data
def load_results():
    with open("results.json") as f:
        r = json.load(f)
    summary = pd.read_csv("results_summary.csv")
    return r, summary


try:
    results, summary_df = load_results()
except FileNotFoundError:
    st.error(
        "Could not find `results.json` / `results_summary.csv` in this folder. "
        "Run `python dem_correction_math_models.py` first to generate them."
    )
    st.stop()

FEATURES = results["feature_names"]
mu = np.array(results["feature_means"])
sigma = np.array(results["feature_stds"])

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🛰️ DEM Correction")
st.sidebar.markdown("*Mathematical Framework Dashboard*")
st.sidebar.markdown("---")

# Real glassmorphism hover on native st.button elements. Streamlit renders
# each button with data-testid="stBaseButton-primary" (active page, set via
# type="primary" below) or "stBaseButton-secondary" (inactive pages) — these
# testids live in the main app DOM (not an iframe), so backdrop-filter blur
# actually applies, unlike component-based nav menus.
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button[data-testid^="stBaseButton-"] {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        margin-bottom: 3px;
    }
    /* Inactive nav items */
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        background-color: rgba(255,255,255,0.02) !important;
        border: 1px solid transparent !important;
        color: #94a3b8 !important;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(14px) saturate(160%);
        -webkit-backdrop-filter: blur(14px) saturate(160%);
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        color: #f1f5f9 !important;
        transform: translateX(3px);
        box-shadow: 0 4px 18px rgba(56, 189, 248, 0.15);
    }
    /* Active nav item */
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(167,139,250,0.16)) !important;
        border: 1px solid rgba(56, 189, 248, 0.55) !important;
        backdrop-filter: blur(14px) saturate(160%);
        -webkit-backdrop-filter: blur(14px) saturate(160%);
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.20);
        color: #e0f2fe !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
        transform: translateX(3px);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.30);
    }
    section[data-testid="stSidebar"] button p {
        text-align: left !important;
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NAV_ITEMS = [
    ("Overview", "∑"),                     # aggregate view
    ("Mathematical Model", "∇"),           # gradient / objective function
    ("Method Comparison", "≈"),            # methods converge to same optimum
    ("Spatial Baselines", "📍"),           # IDW / k-NN geographic baselines
    ("Convergence Analysis", "→0"),        # loss shrinking to zero
    ("PCA Analysis", "λ"),                 # eigenvalues
    ("Curve Fit & Predictor", "ƒ(x)"),     # fitted function
    ("SDG 11 Impact", "🌍"),
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

for name, symbol in NAV_ITEMS:
    is_active = st.session_state.page == name
    if st.sidebar.button(
        f"{symbol}   {name}",
        key=f"nav_{name}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.page = name
        st.rerun()

page = st.session_state.page

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <span class="badge">Train: {results['train_size']:,}</span>
    <span class="badge">Test: {results['test_size']:,}</span>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Region: 25°N–28°N, 80°E–84°E · Himalayan foothill belt")
st.sidebar.caption("Sentinel-2 optical + terrain features · LiDAR/ICESat reference")

# ---------------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("DEM Correction via Mathematical Optimization")
    st.markdown(
        "Five independent mathematical routes solving **the same convex objective "
        "function** to correct systematic, terrain-correlated satellite DEM error."
    )

    cols = st.columns(4)
    best = summary_df.loc[summary_df["test_rmse"].idxmin()]
    metrics = [
        ("Best Test RMSE", f"{best['test_rmse']:.3f} m", best["method"]),
        ("Training Pixels", f"{results['train_size']:,}", "2018–2022"),
        ("Test Pixels", f"{results['test_size']:,}", "2023 (held out)"),
        ("Condition Number", f"{results['condition_number']:.1f}", "X\u1d40X — well-conditioned"),
    ]
    for c, (label, value, sub) in zip(cols, metrics):
        c.markdown(
            f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div style="color:#64748b; font-size:0.75rem;">{sub}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### The Five Methods")
    method_cards = st.columns(5)
    descs = {
        "Normal Equations": ("Linear Algebra", "θ* = (XᵀX)⁻¹Xᵀe", "Closed-form, exact"),
        "SVD Least Squares": ("Matrix Factorization", "θ* = VΣ⁻¹Uᵀe", "Numerically robust"),
        "PCA Regression (k=6)": ("Dimensionality Reduction", "θ = Pₖ θ_reduced", "87.9% variance, 40% fewer dims"),
        "Gradient Descent": ("First-Order Optimization", "θ ← θ − α∇J(θ)", "~300 iterations"),
        "Newton's Method": ("Second-Order Optimization", "θ ← θ − H⁻¹∇J(θ)", "1 iteration (quadratic loss)"),
    }
    for c, (name, (cat, formula, note)) in zip(method_cards, descs.items()):
        color = METHOD_COLORS[name]
        c.markdown(
            f"""<div class="metric-card" style="border-top:3px solid {color};">
                    <div style="color:{color}; font-weight:700; font-size:0.85rem;">{name}</div>
                    <div style="color:#64748b; font-size:0.7rem; margin:4px 0;">{cat}</div>
                    <div style="font-family:monospace; color:#e2e8f0; font-size:0.8rem; margin:8px 0;">{formula}</div>
                    <div style="color:#94a3b8; font-size:0.7rem;">{note}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### Pipeline")
    st.markdown(
        """
        ```
        Raw satellite DEM (SRTM/Copernicus)
                │  systematic terrain-correlated bias
                ▼
        Feature engineering (slope, aspect, curvature, NDVI, BSI...)
                │
                ▼
        Correction model  ê = θᵀx   ← 5 mathematical solvers (this dashboard)
                │
                ▼
        Corrected DEM  z_corr = z_raw − ê
                │
                ▼
        Flood / landslide hazard mapping  →  SDG 11 disaster-risk planning
        ```
        """
    )

# ---------------------------------------------------------------------------
# PAGE: MATHEMATICAL MODEL
# ---------------------------------------------------------------------------
elif page == "Mathematical Model":
    st.title("Mathematical Formulation")

    st.markdown("#### Elevation Error")
    st.latex(r"e_i = z_i - z_i^{*}")
    st.caption("z_i = raw DEM elevation · z_i* = LiDAR/ICESat reference elevation")

    st.markdown("#### Correction Model")
    st.latex(r"\hat e_i = \boldsymbol\theta^{\top}\mathbf{x}_i, \qquad \mathbf{x}_i = [1, x_{i1}, \ldots, x_{ip}]^{\top}")

    st.markdown("#### Objective Function (Mean Squared Error)")
    st.latex(r"J(\boldsymbol\theta) = \frac{1}{N}\left\lVert X\boldsymbol\theta - \mathbf{e}\right\rVert_2^2")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Gradient")
        st.latex(r"\nabla J(\boldsymbol\theta) = \frac{2}{N}X^{\top}(X\boldsymbol\theta - \mathbf{e})")
    with col2:
        st.markdown("#### Hessian (constant)")
        st.latex(r"H = \nabla^2 J(\boldsymbol\theta) = \frac{2}{N}X^{\top}X")

    st.info(
        "Because H does not depend on θ and Xᵀ X is positive semi-definite, "
        "J(θ) is a **convex quadratic** — guaranteeing a unique global minimum, "
        "reached at ∇J(θ)=0:"
    )
    st.latex(r"X^{\top}X\,\boldsymbol\theta^{*} = X^{\top}\mathbf{e}")

    st.markdown("### Feature Set")
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Train Mean": mu.round(4),
        "Train Std": sigma.round(4),
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: METHOD COMPARISON
# ---------------------------------------------------------------------------
elif page == "Method Comparison":
    st.title("Method Comparison")
    st.caption("Five mathematical routes to the same objective function — see the "
               "**Spatial Baselines** page for how these compare against classical "
               "spatial interpolation.")

    math_df = summary_df[~summary_df["method"].isin(BASELINE_NAMES)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=math_df["method"], y=math_df["test_rmse"],
        marker_color=[METHOD_COLORS[m] for m in math_df["method"]],
        text=math_df["test_rmse"].round(4), textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title="Test RMSE Across the Five Mathematical Methods",
        yaxis_title="Test RMSE (m)", height=450,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full Results Table")
    st.dataframe(
        math_df.style.format({
            "train_rmse": "{:.4f}", "test_rmse": "{:.4f}", "test_mae": "{:.4f}",
            "train_r2": "{:.4f}", "le90_formula": "{:.4f}", "le90_empirical": "{:.4f}",
            "iterations": "{:.0f}",
        }).background_gradient(subset=["test_rmse"], cmap="Blues_r"),
        use_container_width=True, hide_index=True,
    )

    st.success(
        f"All closed-form methods (Normal Equations, SVD) and iterative methods "
        f"(Gradient Descent, Newton's) converge to the **same optimum** "
        f"(test RMSE ≈ {math_df['test_rmse'].min():.3f} m), confirming numerical "
        f"correctness across every solution route."
    )

# ---------------------------------------------------------------------------
# PAGE: SPATIAL BASELINES
# ---------------------------------------------------------------------------
elif page == "Spatial Baselines":
    st.title("📍 Comparison with Classical Spatial Interpolation")
    st.markdown(
        "Benchmarking the feature-based regression methods against **Inverse Distance "
        "Weighting (IDW)** and **k-NN averaging** — reproducing the classical baselines "
        "from the reference document *Mathematical Modeling and Residual Learning for "
        "Terrain-Aware Correction of Satellite-Derived DEMs*, at full dataset scale."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Inverse Distance Weighting")
        st.latex(r"\hat e(\mathbf{s}) = \frac{\sum_{i=1}^{k} e_i / d_i^{\,p}}{\sum_{i=1}^{k} 1/d_i^{\,p}}")
        st.caption("k=8 nearest training pixels, p=2, geographic distance in meters")
    with col2:
        st.markdown("#### k-NN Unweighted Mean (\"Bilinear-style\")")
        st.latex(r"\hat e(\mathbf{s}) = \frac{1}{k}\sum_{i=1}^{k} e_i, \quad k=4")
        st.caption("Reproduces the reference document's own worked Bilinear example — "
                   "which is itself a simple 4-neighbor average, not true bilinear weights")

    st.markdown("### Result: An Honest Surprise")
    display_df = summary_df.copy()
    display_df["type"] = display_df["method"].apply(
        lambda m: "Spatial (no features)" if m in BASELINE_NAMES else "Feature-based regression"
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display_df["method"], y=display_df["test_rmse"],
        marker_color=[METHOD_COLORS[m] for m in display_df["method"]],
        marker_pattern_shape=["/" if m in BASELINE_NAMES else "" for m in display_df["method"]],
        text=display_df["test_rmse"].round(4), textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title="Test RMSE — Feature-Based Regression vs. Spatial Baselines",
        yaxis_title="Test RMSE (m)", height=460, xaxis_tickangle=-25,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    best_math = results.get("best_math_method", "SVD Least Squares")
    best_baseline = results.get("best_baseline_method", "IDW (k=8, spatial)")
    math_rmse = summary_df.loc[summary_df["method"] == best_math, "test_rmse"].values[0]
    base_rmse = summary_df.loc[summary_df["method"] == best_baseline, "test_rmse"].values[0]
    pct = (math_rmse - base_rmse) / math_rmse * 100

    st.warning(
        f"The purely spatial **{best_baseline}** achieves **{pct:.1f}% lower** test RMSE "
        f"({base_rmse:.4f} m) than the best feature-based regression method "
        f"**{best_math}** ({math_rmse:.4f} m) — the opposite of what a naive expectation "
        f"would suggest, and worth reporting honestly rather than hiding."
    )

    st.markdown("### Why: Spatial Autocorrelation and Train/Test Geographic Overlap")
    leak = results.get("spatial_leakage", {})
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-card"><div class="metric-label">Median NN Distance</div>
        <div class="metric-value">{leak.get('median_nn_distance_m', 0):.1f} m</div></div>""",
        unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="metric-label">Within 1 DEM Pixel (93m)</div>
        <div class="metric-value">{leak.get('frac_within_1_pixel_93m', 0)*100:.1f}%</div></div>""",
        unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="metric-label">Within 10 m</div>
        <div class="metric-value">{leak.get('frac_within_10m', 0)*100:.1f}%</div></div>""",
        unsafe_allow_html=True)

    st.markdown(
        """
        Training data (2018–2022) and test data (2023) cover the **same geographic
        region**, split only by year. Since terrain-driven DEM bias is largely
        **time-invariant**, a nearby training pixel from an earlier year is often an
        almost-exact preview of a 2023 test pixel's error — which IDW exploits directly
        by retrieving that historical measurement. The linear regression models must
        instead compress all location-specific bias through a single global 10-parameter
        function of terrain/optical features, and cannot memorize per-pixel history the
        way a nearest-neighbor spatial method can.
        """
    )

    st.markdown("### What This Means: Regression-Kriging as the Natural Next Step")
    st.latex(
        r"\hat e(\mathbf{s}) = \underbrace{\boldsymbol\theta^{\top}\mathbf{x}(\mathbf{s})}_{\text{trend (Methods 1-5)}} "
        r"+ \underbrace{\text{IDW}\big(e(\mathbf{s}') - \boldsymbol\theta^{\top}\mathbf{x}(\mathbf{s}')\big)}_{\text{spatial residual correction}}"
    )
    st.info(
        "This is not a failure of the mathematical framework — it's a correct diagnosis "
        "that unifies every method in this project into a single hybrid pipeline: the "
        "regression captures the systematic terrain-driven trend, and spatial "
        "interpolation cleans up the remaining spatially autocorrelated residual."
    )

# ---------------------------------------------------------------------------
# PAGE: CONVERGENCE ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Convergence Analysis":
    st.title("Convergence Analysis")

    gd_hist = results["gd_history"]
    nt_hist = results["newton_history"]

    tab1, tab2 = st.tabs(["Gradient Descent vs Newton", "Individual Curves"])

    with tab1:
        n_show = st.slider("Iterations to display", 5, len(gd_hist), 20)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=gd_hist[:n_show], mode="lines+markers", name="Gradient Descent",
            line=dict(color=ACCENT, width=3), marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            y=nt_hist[:min(n_show, len(nt_hist))], mode="lines+markers", name="Newton's Method",
            line=dict(color=ACCENT2, width=3), marker=dict(size=8, symbol="square"),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, title="Loss J(θ) vs Iteration (log scale)",
            yaxis_type="log", xaxis_title="Iteration", yaxis_title="J(θ)",
            height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
            - **Newton's Method** reaches the optimum (`J = {nt_hist[1]:.6f}`) in **1 iteration**
              — because the Hessian is *constant* for this exactly-quadratic loss.
            - **Gradient Descent** takes **~{len(gd_hist)} iterations** at step size
              α = 1/λ_max(H) to reach the same loss.
            - Newton pays O(p³) once (Hessian inversion, p={len(FEATURES)+1}); Gradient
              Descent pays O(Np) every iteration.
            """
        )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = go.Figure(go.Scatter(y=gd_hist, mode="lines", line=dict(color=ACCENT, width=2)))
            fig1.update_layout(template=PLOTLY_TEMPLATE, title="Gradient Descent (full run)",
                                yaxis_type="log", height=400,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Scatter(y=nt_hist, mode="lines+markers",
                                          line=dict(color=ACCENT2, width=2), marker=dict(size=10)))
            fig2.update_layout(template=PLOTLY_TEMPLATE, title="Newton's Method (full run)",
                                height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: PCA ANALYSIS
# ---------------------------------------------------------------------------
elif page == "PCA Analysis":
    st.title("PCA: Dimensionality Reduction")

    var_ratio = np.array(results["explained_variance_ratio"])
    cum_var = np.cumsum(var_ratio)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(var_ratio) + 1)), y=var_ratio,
                          name="Individual", marker_color=ACCENT, opacity=0.7))
    fig.add_trace(go.Scatter(x=list(range(1, len(var_ratio) + 1)), y=cum_var,
                              name="Cumulative", mode="lines+markers",
                              line=dict(color=ACCENT2, width=3)))
    fig.add_hline(y=0.95, line_dash="dash", line_color="#64748b",
                  annotation_text="95% threshold")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title="Scree Plot — Terrain + Optical Features",
        xaxis_title="Principal Component", yaxis_title="Explained Variance Ratio",
        height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    k = st.slider("Number of components (k)", 1, len(var_ratio), 6)
    st.metric(f"Variance retained with k={k}", f"{cum_var[k-1]*100:.1f}%")
    st.caption(
        f"With k=6 (used in the assignment), 87.9% of feature variance is retained "
        f"at 40% lower dimensionality, at a cost of only "
        f"{summary_df.loc[summary_df['method']=='PCA Regression (k=6)','test_rmse'].values[0] - summary_df['test_rmse'].min():.4f} m "
        f"additional test RMSE versus the full-rank solution."
    )

# ---------------------------------------------------------------------------
# PAGE: CURVE FIT & PREDICTOR
# ---------------------------------------------------------------------------
elif page == "Curve Fit & Predictor":
    st.title("Curve Fit & Interactive Predictor")

    sample = results["scatter_sample"]
    theta = np.array(results["Normal Equations"]["theta"])

    st.markdown("### Elevation Error vs Slope (fitted model)")
    slope_range = np.linspace(0, 10, 100)
    slope_std = (slope_range - mu[0]) / sigma[0]
    X_curve = np.zeros((100, len(FEATURES)))
    X_curve[:, 0] = slope_std
    X_design = np.column_stack([np.ones(100), X_curve])
    e_curve = X_design @ theta

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=sample["slope_deg"], y=sample["elevation_error_signed"],
        mode="markers", marker=dict(size=3, color="#475569", opacity=0.4),
        name="Observed (sample)",
    ))
    fig.add_trace(go.Scatter(
        x=slope_range, y=e_curve, mode="lines",
        line=dict(color=ACCENT2, width=4), name="Fitted (Normal Equations)",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Slope (degrees)",
        yaxis_title="Elevation Error (m)", height=460,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Interactive Correction Predictor")
    st.caption("Adjust terrain/optical inputs and see the model's predicted DEM error in real time.")

    cols = st.columns(5)
    raw_inputs = {}
    default_ranges = {
        "slope_deg": (0.0, 45.0, float(mu[0])),
        "aspect_sin": (-1.0, 1.0, 0.0),
        "aspect_cos": (-1.0, 1.0, 0.0),
        "tri": (0.0, 3.0, float(mu[3])),
        "tpi": (-2.0, 2.0, 0.0),
        "planform_curvature": (-0.01, 0.01, 0.0),
        "profile_curvature": (-0.01, 0.01, 0.0),
        "hillshade": (0.0, 1.0, float(mu[7])),
        "ndvi": (-1.0, 1.0, float(mu[8])),
        "bsi": (-1.0, 1.0, 0.0),
    }
    for i, feat in enumerate(FEATURES):
        lo, hi, default = default_ranges[feat]
        with cols[i % 5]:
            raw_inputs[feat] = st.slider(feat, lo, hi, default, key=f"slider_{feat}")

    x_raw = np.array([raw_inputs[f] for f in FEATURES])
    x_std = (x_raw - mu) / sigma
    x_design = np.concatenate([[1.0], x_std])
    predicted_error = float(x_design @ theta)

    st.markdown(
        f"""<div class="metric-card" style="text-align:center; margin-top:16px;">
                <div class="metric-label">Predicted DEM Elevation Error  ê</div>
                <div class="metric-value" style="font-size:2.5rem;">{predicted_error:+.3f} m</div>
                <div style="color:#64748b; font-size:0.8rem;">
                    Corrected elevation: z_corr = z_raw − ê &nbsp;·&nbsp; using Normal Equations θ*
                </div>
            </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# PAGE: SDG 11 IMPACT
# ---------------------------------------------------------------------------
elif page == "SDG 11 Impact":
    st.title("🏙️ SDG 11 — Sustainable Cities and Communities")

    st.markdown(
        """
        Corrected elevation data is a direct input to two SDG 11 targets:

        | Target | Requirement | Link to this project |
        |---|---|---|
        | **11.5** | Reduce disaster deaths/losses, incl. water-related disasters | Flood-extent & landslide-susceptibility maps depend on elevation accuracy |
        | **11.3** | Inclusive, participatory settlement planning | Free satellite DEMs are the only elevation source for many local authorities |
        | **11.b** | Integrated disaster-risk-reduction policy (Sendai Framework) | Neighborhood-scale accuracy is exactly where terrain-correlated bias is largest |
        """
    )

    st.markdown("### Study Region")
    st.markdown(
        "**25°N–28°N, 80°E–84°E** — Himalayan foothill / Terai belt (India–Nepal border zone). "
        "Recurring monsoon flash floods and landslides, sparse LiDAR coverage, "
        "heavy reliance on free satellite DEMs for planning."
    )

    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f"""<div class="metric-card"><div class="metric-label">Corrected RMSE</div>
        <div class="metric-value">{summary_df['test_rmse'].min():.2f} m</div></div>""",
        unsafe_allow_html=True)
    m2.markdown(
        f"""<div class="metric-card"><div class="metric-label">Validation Pixels</div>
        <div class="metric-value">{results['test_size']:,}</div></div>""",
        unsafe_allow_html=True)
    m3.markdown(
        f"""<div class="metric-card"><div class="metric-label">Held-out Year</div>
        <div class="metric-value">2023</div></div>""",
        unsafe_allow_html=True)

    st.markdown("### Pipeline to Impact")
    st.markdown(
        """
        ```
        Raw satellite DEM (free, biased)
                │
                ▼
        Mathematical correction (this project — 5 solver routes)
                │
                ▼
        Corrected DEM
                │
                ▼
        Flood-extent / landslide-susceptibility mapping   ──▶  SDG 11.5
                │
                ▼
        Municipal disaster-risk-informed planning          ──▶  SDG 11.3 / 11.b
        ```
        """
    )

    st.warning(
        "**Honest limitation:** linear R² ≈ 0.10 — the correction captures only part "
        "of the systematic error. SAR (`vv`/`vh`) features were excluded here due to "
        "~90% missing coverage; nonlinear extensions (e.g. Huber loss, interaction "
        "terms) are natural next steps within the same mathematical framework."
    )