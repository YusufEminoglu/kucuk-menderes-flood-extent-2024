# Rapid Flood Extent Mapping and Exposure Assessment — Küçük Menderes Basin (Sep 2024)

**Repository:** `kucuk-menderes-flood-extent-2024`  
**Scope:** Cloud-native workflow for flood-extent mapping and exposure assessment using Sentinel-1/2, Google Earth Engine (GEE), and classical ML (RF/SVM/CART) for the Küçük Menderes Basin, Türkiye (10–14 Sep 2024).

> Code runs in Google Earth Engine (JavaScript) and Google Colab (Python).  
> Data products and figures are released under **CC BY 4.0**; code is released under **MIT**.

---

## What's here

- `gee/` — GEE JavaScript scripts (LULC mapping; SAR log-ratio threshold; MNDWI & Otsu; RF flood classifier).  
- `colab/` — Python script/notebook for hyper-parameter tuning & ROC (reproduces Fig. 6-like panels).  
- `docs/` — Reproducibility notes and figure guide.  
- `data/` — Pointers to public datasets & how to request derived rasters (no large files tracked).  
- `fig/` — (Optional) export folder for figures (not versioned by default).  
- `other/` — Bibliography and ancillary files.

## Quick start

### 1) LULC in GEE (RF/SVM/CART)
1. Open **GEE Code Editor** and create four scripts; copy the contents of:
   - `gee/lulc_classification.js`
   - `gee/flood_extent_sar.js`
   - `gee/otsu_mndwi.js`
   - `gee/rf_flood_classifier.js`
2. Set **AOI** and **asset IDs** where marked `// TODO: UPDATE_ME`.
3. Run `lulc_classification.js` to produce the **2024 Aug–Sep** LULC (RF retained).
4. Export results to your GEE Assets (GeoTIFF recommended).

### 2) Flood-extent (four detectors)
- **SAR log-ratio** with **~1.40** linear threshold (event-informed; see histogram panel).  
- **MNDWI > 0** after permanent-water masking (map + histogram panel).  
- **Otsu** on SAR-derived edges (final Otsu cut ≈ **−0.1243** for this event).  
- **RF flood classifier** trained on **910** points (80/20 split; 50 trees).

### 3) Hyper-parameter tuning & ROC (Colab)
- Upload `colab/fig6_hyperparam_roc.py` **or** adapt into a notebook.  
- Provide CSVs (RF grid, SVM grid, CART sweep, sampled normalized parameters) as in the script header.  
- Outputs: heatmap/curves (top row) + 10-fold ROC panels (bottom row).

## Reproducibility keys (event specifics)

- **Dates:** LULC composite **Aug–Sep 2024**; flood window **10–14 Sep 2024**.  
- **Masks:** Permanent water (JRC GSW); slope **> 5%** removed (HydroSHEDS).  
- **Thresholds:** SAR log-ratio ≈ **1.40** (linear); **MNDWI > 0**; Otsu cut **−0.1243**.  
- **ML flood:** **RF (50 trees)**, 910 labels, **80/20 split**; metrics reported with 10-fold ROC.  
- **Exposure:** HRSL population overlay + RF LULC.

> See `docs/REPRODUCIBILITY.md` and `docs/FIGURE_GUIDE.md` for step-by-step notes.

## Data availability

- **Public sources:** Sentinel-1/2 (Copernicus), JRC Global Surface Water, HydroSHEDS, HRSL.  
- **Derived rasters** (LULC, flood masks per detector, CFEVI) will be shared via DOI archive upon publication; until then, request via Issues or email (see below).

## How to cite

```
@article{YourLastName2025_KMB_Flood,
  title   = {Rapid Flood Extent Mapping and Exposure Assessment using SAR and Machine Learning in the Küçük Menderes Basin, Türkiye},
  author  = {Your Name and Co-authors},
  year    = {2025},
  journal = {Natural Hazards},
  note    = {Preprint / In review. Repository: kucuk-menderes-flood-extent-2024}
}
```

A `CITATION.cff` is provided for GitHub citation support.

## License

- **Code:** MIT (see `LICENSE-MIT`)  
- **Data & figures:** CC BY 4.0 (see `LICENSE-CC-BY-4.0`)

## Contact

- Maintainer: **Your Name** — <your.email@domain>  
- Please open Issues for bugs, questions, or data requests.
