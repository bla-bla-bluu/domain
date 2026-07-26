# Thermal Pedestrian Detection in Wilderness Environments

Code release for the paper **"Thermal Pedestrian Detection in Wilderness Environments:
Self-Training on Pseudo-Labels Outperforms Scale Augmentation and Generative Domain
Adaptation."**

The study compares three *label-free* strategies for adapting an urban-trained thermal
(LWIR) pedestrian detector to a forest/wilderness target domain, evaluated on a common
hand-annotated forest test set:

1. **SAHI** — Sliced Aided Hyper Inference (tiled inference at test time, no retraining).
2. **CycleGAN UDA** — unpaired urban→forest image translation, then train on the synthetic set.
3. **Self-training** — pseudo-labels from the SAHI-augmented baseline on real unlabeled
   forest frames, then train a fresh detector on them.

A supervised **real-label reference model** ("oracle") and an **LLVIP-trained baseline**
bracket the comparison. All detectors are YOLOv8-Nano with identical hyperparameters.

## Repository layout

```
code/
  train/        train_selftrain.py            – detector training (self-train condition)
  eval/         eval_sahi.py                  – TP/FP/FN under SAHI / standard inference
                eval_sahi_map.py              – SAHI-mode mAP50 (bespoke tile-merged PR sweep)
                eval_ultralytics_native.py    – standard-mode counts via ultralytics' own validator
                validate_custom_ap.py/_ap2.py – custom-vs-ultralytics AP cross-checks
  figures/      make_figures.py               – results_comparison + tile_size_sweep figures
  experiments/  deleak_setup.py, deleak_run.py    – de-leaked validation split (paper §5.4)
                map_pipeline_analysis.py, map_pipeline_probe.py – SAHI-mAP50 pipeline-discrepancy analysis (§5.1)
configs/        *.yaml                         – Ultralytics data configs (per condition)
```

## Status

This is a **partial release**. The scripts above reproduce the evaluation, the figures, and
the two robustness analyses reported in the paper. Some data-preparation and figure scripts
(`generate_pseudo_labels.py`, `prepare_yolo.py`, `prepare_ir_oracle.py`, `eval_bootstrap.py`,
`eval_threshold_sweep.py`, `make_confusion_figures.py`, `make_oracle_figures.py`,
`make_selftrain_figures.py`, `make_framework_diagram.py`) are being added.

**Paths** inside the scripts are hardcoded to the original workstation layout
(`/home/deepak/domain/...`); adjust them to your environment before running.

## Requirements

- Python 3, [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8), and
  [SAHI](https://github.com/obss/sahi) for the tiled-inference conditions.
- The CycleGAN component builds on
  [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Data

The forest test set and training pools are thermal human-subject recordings self-generated
by the authors. They are **not** included here; availability is described in the paper's Data
Availability statement. The [LLVIP](https://github.com/bupt-ai-cz/LLVIP) source dataset is
public under its own license.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

No license is set yet — see `CITATION.cff`. Until a license is added, no permissions are
granted beyond viewing. (Add e.g. MIT/BSD-3-Clause per your institution's policy.)
