#!/usr/bin/env python3
"""Standard-mode TP/FP/FN via ultralytics' own validator (its dataloader, inference, NMS,
and ConfusionMatrix matching), with the matching IoU forced to 0.5. Runs with batch=1 and a
per-batch callback so per-image counts are available for bootstrap resampling.

Usage:
  python eval_ultralytics_native.py --model <best.pt> --conf 0.15 --bootstrap"""
import argparse
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

DATA_YAML = "/home/deepak/domain/code/forest_config_v2.yaml"

_orig_process_batch = ConfusionMatrix.process_batch
def _patched_process_batch(self, detections, batch, conf=0.25, iou_thres=0.45):
    return _orig_process_batch(self, detections, batch, conf=conf, iou_thres=0.5)
ConfusionMatrix.process_batch = _patched_process_batch


def per_image_counts(model_path, conf, iou=0.5):
    """Returns a list of (tp, fp, fn) tuples, one per test image, in dataset order."""
    model = YOLO(model_path)
    per_image = []
    prev = {"matrix": None}

    def on_batch_end(validator):
        cm = validator.confusion_matrix.matrix
        delta = cm.copy() if prev["matrix"] is None else cm - prev["matrix"]
        prev["matrix"] = cm.copy()
        per_image.append((int(delta[0, 0]), int(delta[0, 1]), int(delta[1, 0])))

    model.add_callback("on_val_batch_end", on_batch_end)
    model.val(data=DATA_YAML, imgsz=640, conf=conf, iou=iou, batch=1, workers=0,
              device="cuda", verbose=False, plots=True)
    return per_image


def summarize(per_image):
    tp = sum(x[0] for x in per_image)
    fp = sum(x[1] for x in per_image)
    fn = sum(x[2] for x in per_image)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f1


def bootstrap_ci(per_image, n_resamples=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(per_image)
    arr = np.array(per_image)  # (n, 3): tp, fp, fn
    ps, rs, f1s = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        tp, fp, fn = sample[:, 0].sum(), sample[:, 1].sum(), sample[:, 2].sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        ps.append(p); rs.append(r); f1s.append(f1)
    def ci(vals):
        return np.percentile(vals, 2.5) * 100, np.percentile(vals, 97.5) * 100
    return {"P_CI": ci(ps), "R_CI": ci(rs), "F1_CI": ci(f1s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()

    per_image = per_image_counts(args.model, args.conf)
    tp, fp, fn, p, r, f1 = summarize(per_image)
    print(f"\n{'='*55}")
    print(f"  Ultralytics-native (IoU=0.5 matching), conf={args.conf}")
    print(f"  N images: {len(per_image)}  Total GT-matched TP+FN: {tp+fn}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision: {p*100:.1f}%   Recall: {r*100:.1f}%   F1: {f1*100:.1f}%")
    if args.bootstrap:
        cis = bootstrap_ci(per_image)
        print(f"  95% CI  P: [{cis['P_CI'][0]:.1f}, {cis['P_CI'][1]:.1f}]"
              f"   R: [{cis['R_CI'][0]:.1f}, {cis['R_CI'][1]:.1f}]"
              f"   F1: [{cis['F1_CI'][0]:.1f}, {cis['F1_CI'][1]:.1f}]")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
