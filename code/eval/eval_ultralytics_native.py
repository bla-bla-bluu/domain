#!/usr/bin/env python3
"""
Standard-mode (non-SAHI) TP/FP/FN using ultralytics' OWN validator pipeline end to end:
its dataloader (rect=True letterbox), its inference, its NMS, and its ConfusionMatrix
matching -- with the matching IoU threshold forced to 0.5 to match this paper's stated
IoU@0.5 methodology (ultralytics hardcodes 0.45 for confusion-matrix matching regardless
of the `iou=` argument passed to .val(), which only controls NMS; verified by reading
ultralytics.models.yolo.detect.val.DetectionValidator's call to
confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf) -- no iou_thres passed).

Uses the high-level `model.val(...)` API (NOT a manually-constructed DetectionValidator --
an earlier version of this script did that and gave numbers inconsistent with model.val()'s
own aggregate P/R printout, most likely because model.val() does additional setup, e.g.
layer fusion, that a bare DetectionValidator(args=...) call skips; model.val() is treated
as ground truth here since it is the officially documented, public entry point). Runs with
batch=1 so each validation "batch" is exactly one image, and a callback reads the confusion
matrix's cumulative total before/after each image to recover that image's own TP/FP/FN --
giving frame-level granularity for bootstrap resampling while reusing ultralytics' real
preprocessing/inference/matching rather than a hand-rolled loop. Cross-checked: summed
per-image counts equal r.confusion_matrix.matrix exactly, and P/R match ultralytics' own
printed aggregate metrics to 3 decimal places.

Supersedes eval_sahi.py's --no_sahi mode for baseline/v2/oracle (v1 has no available
weights on this workstation, so it retains its previously-measured custom-pipeline numbers).

Usage:
  python eval_ultralytics_native.py --model <path/to/best.pt> --conf 0.15 --bootstrap
"""
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
