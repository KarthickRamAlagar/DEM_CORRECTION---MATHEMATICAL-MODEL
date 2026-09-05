# SDG 11 Alignment: DEM Correction for Sustainable Cities and Communities

## 1. Why This Project Maps to SDG 11

Sustainable Development Goal 11 — *Sustainable Cities and Communities* — commits to
making human settlements safe, resilient, and sustainable. Two of its targets depend
directly on accurate elevation data, which is the exact output this project produces:

| SDG 11 Target | Requirement | Dependency on Corrected DEM |
|---|---|---|
| **11.5** | Significantly reduce deaths, affected populations, and economic losses from disasters, including water-related disasters, by 2030 | Flood-extent models, landslide-susceptibility maps, and drainage design all take elevation as their primary input. A DEM with terrain-correlated bias produces systematically wrong flood boundaries and slope-failure risk zones. |
| **11.3** | Enhance inclusive and sustainable urbanization and capacity for participatory, integrated human settlement planning | Local planning authorities in data-scarce regions rely on freely available satellite DEMs (SRTM/ASTER/Copernicus) rather than expensive LiDAR surveys. Bias-corrected DEMs make that free data usable for real planning decisions. |
| **11.b** | Substantially increase the number of cities/settlements implementing integrated policies for disaster risk reduction, in line with the Sendai Framework | Disaster-risk-informed planning requires elevation accuracy at the neighborhood scale — the resolution at which slope- and vegetation-correlated DEM error is largest. |

## 2. Why the Study Region Makes This Concrete, Not Generic

The dataset underlying this project spans **25°N–28°N, 80°E–84°E** — the Himalayan
foothill / Terai belt straddling northern India and Nepal. This is not an arbitrary
choice of coordinates; it is one of the regions where DEM accuracy has the highest
real-world stakes:

- **Steep-to-moderate terrain gradients** (the same `slope_deg`, `tri`, and curvature
  features used as regression predictors in this project) are exactly the terrain
  conditions under which optical/radar satellite DEMs are known to have their largest
  systematic vertical bias — SAR layover/shadow and optical parallax errors both scale
  with slope.
- The region experiences recurring **monsoon-driven flash floods and landslides**,
  where hazard maps are built on freely available DEMs (SRTM/Copernicus) because
  LiDAR coverage is sparse or unavailable at the municipal level.
- Vegetation-correlated bias (captured here through `ndvi`, `evi`, `bsi`) is
  significant in the outer Terai belt's mixed forest-agriculture cover, where canopy
  return height inflates apparent ground elevation in the raw DEM.

This means the same feature set built for the mathematics assignment — slope, aspect,
curvature, NDVI, BSI — is not an abstract regression exercise; each feature was
selected *because* it is a documented physical driver of DEM error in exactly this
kind of terrain.

## 3. From Mathematical Model to SDG Outcome

The project's contribution sits at a specific point in the disaster-risk pipeline:

```
Raw satellite DEM (SRTM/Copernicus, freely available, terrain-biased)
        │
        ▼
Bias correction model  ĥeᵢ = θᵀxᵢ   (this project — 5 mathematical solution methods)
        │
        ▼
Corrected DEM   z_corr = z_raw − ê
        │
        ▼
Flood-extent / landslide-susceptibility mapping   (downstream use, SDG 11.5)
        │
        ▼
Municipal disaster-risk-informed land-use planning   (downstream use, SDG 11.3 / 11.b)
```

The mathematical framework developed for the assignment component is therefore not a
side exercise — it is the correction stage that determines how trustworthy every
downstream hazard product is. An uncorrected DEM with several meters of
slope-correlated bias can shift a modeled flood boundary or landslide-susceptibility
contour by an amount that is operationally significant at the scale local planning
authorities work at.

## 4. Quantified Contribution (from this project's results)

- Test-set RMSE after correction: **2.07 m**, evaluated on 51,826 pixels withheld from
  a different year (2023) than the training data (2018–2022) — i.e. the model was
  validated on genuine out-of-sample terrain, not just refitting the training set.
- All five mathematical solution routes (Normal Equations, SVD, PCA Regression,
  Gradient Descent, Newton's Method) converge to the same correction, which matters
  operationally: whichever computational environment a downstream disaster-management
  agency uses, the correction is mathematically reproducible, not solver-dependent.
- PCA regression shows that **6 of 10 terrain/optical features retain 87.9% of the
  explanatory signal** — meaning a lighter-weight version of this correction model
  could realistically run on lower-resource local-government hardware, which matters
  for target 11.b's emphasis on capacity in under-resourced settlements.

## 5. Limitations and Future Work (honest framing for the term project)

- Linear R² (~0.10) indicates the correction captures only part of the systematic
  error; a meaningful share of DEM error is likely nonlinear in these features
  (interaction effects between slope and vegetation cover, for instance) or driven by
  sensor-specific artifacts not present in this feature set (radar layover/shadow —
  the `vv`/`vh` SAR features were excluded here due to ~90% missing coverage).
- Future iterations of this pipeline could extend the same mathematical framework
  (same objective function, same five solvers) to a **non-quadratic robust loss**
  (e.g. Huber loss) to reduce sensitivity to GCP/reference outliers, or incorporate
  the SAR bands once coverage gaps are resolved — at which point Newton's Method
  would no longer converge in one step, and the iterative comparison becomes even
  more informative.
- Operational deployment for actual municipal disaster planning would require
  validation against locally surveyed benchmarks in addition to the ICESat/LiDAR
  reference used here, and formal hydrological/geotechnical hazard modeling
  downstream of the corrected DEM — both are outside the mathematical scope of this
  project but are the natural next stage of the same pipeline.

## 6. One-Sentence Summary (for abstract/conclusion use)

> This project demonstrates that terrain- and vegetation-correlated satellite DEM
> error can be corrected using a mathematically rigorous, reproducible regression
> framework, directly supporting SDG 11 targets 11.3, 11.5, and 11.b in a
> disaster-prone Himalayan foothill region where free satellite elevation data is the
> only elevation source available to local planning authorities.
