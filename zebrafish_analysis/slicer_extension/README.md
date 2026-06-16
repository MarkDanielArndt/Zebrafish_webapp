# ZebrafishAnalysis — 3D Slicer Extension

Offline zebrafish morphometry from 2-D microscopy images.

## Features

- Body length (µm) via medial-axis centerline
- Curvature classification (classes 1–4)
- Length / straight-line ratio
- Eye segmentation (optional)
- Scalebar auto-detection + manual entry
- Batch processing with per-image error isolation
- Gallery, Detail, Results, and Exclude tabs
- Excel + CSV export

## First-time setup

Dependencies are pip-installed into Slicer's Python on first load.
Slicer restart required after installation.

## Adding a model

Edit `zebrafish_analysis/core/models/registry.py` — add one entry to
`MODEL_REGISTRY` and optionally `MODEL_PRESETS`.
