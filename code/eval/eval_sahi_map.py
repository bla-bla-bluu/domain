#!/usr/bin/env python3
"""
mAP50 for SAHI sliced-inference conditions on the 150-frame Golden Test Set.

eval_sahi.py only reports TP/FP/FN at a single confidence threshold (0.15).
Standard-mode mAP50 is available via ultralytics' own .val() (PR-sweep built in),
but ultralytics has no notion of SAHI tiling, so SAHI mAP50 needs a bespoke sweep:
run SAHI at a near-zero confidence floor to keep the full score range, pool all
detections across all 150 images, then compute single-class AP@IoU=0.5 with the
same 101-point-interpolation method ultralytics uses (metrics.py: compute_ap),
so the number is on the same footing as the standard-mode mAP50 already reported.

Usage:
  python eval_sahi_map.py --model <path/to/best.pt> --slice 320 --overlap 0.2
"""
import os, argparse, glob
import numpy as np
from PIL import Image


def parse_gt(label_dir):
    gt = {}
    for lf in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lf))[0]
        boxes = []
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(x) for x in parts[1:]])
        gt[stem] = boxes
    return gt


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def norm_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    return [(cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h]


def compute_ap(recall, precision):
    """101-point interpolation, exact port of ultralytics.utils.metrics.compute_ap
    (verified against ultralytics source; the extra recall[-1]/0.0 sentinel pair
    is required so precision drops to 0 immediately past the last real data point
    instead of interpolating linearly down to 0 across [recall[-1], 1.0])."""
    recall = np.asarray(recall)
    precision = np.asarray(precision)
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ap = trapz(np.interp(x, mrec, mpre), x)
    return ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--model", required=True)
    ap_.add_argument("--slice", type=int, default=320)
    ap_.add_argument("--overlap", type=float, default=0.2)
    ap_.add_argument("--iou_thresh", type=float, default=0.5)
    ap_.add_argument("--score_floor", type=float, default=0.001)
    args = ap_.parse_args()

    VAL_IMG = "/home/deepak/domain/data/yolo_val/images"
    VAL_LBL = "/home/deepak/domain/data/yolo_val/labels"

    gt_all = parse_gt(VAL_LBL)
    images = sorted(glob.glob(os.path.join(VAL_IMG, "*.jpg")) + glob.glob(os.path.join(VAL_IMG, "*.png")))

    print(f"Model: {args.model}  SAHI {args.slice}x{args.slice}  overlap={args.overlap}")

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.score_floor,
        device="cuda",
    )

    all_dets = []  # (score, image_idx, box)
    gt_boxes_per_image = []
    total_gt = 0

    for img_idx, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_norm = gt_all.get(stem, [])
        img = Image.open(img_path)
        img_w, img_h = img.size
        gt_boxes = [norm_to_xyxy(g, img_w, img_h) for g in gt_norm]
        gt_boxes_per_image.append(gt_boxes)
        total_gt += len(gt_boxes)

        result = get_sliced_prediction(
            img_path, detection_model,
            slice_height=args.slice, slice_width=args.slice,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            verbose=0,
        )
        for obj in result.object_prediction_list:
            b = obj.bbox
            all_dets.append((obj.score.value, img_idx, [b.minx, b.miny, b.maxx, b.maxy]))

        if (img_idx + 1) % 25 == 0:
            print(f"  ... {img_idx + 1}/{len(images)} images processed")

    # Sort all detections globally by confidence, descending
    all_dets.sort(key=lambda d: d[0], reverse=True)

    claimed = [np.zeros(len(g), dtype=bool) for g in gt_boxes_per_image]
    tp = np.zeros(len(all_dets))
    fp = np.zeros(len(all_dets))

    for di, (score, img_idx, box) in enumerate(all_dets):
        gts = gt_boxes_per_image[img_idx]
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if claimed[img_idx][gi]:
                continue
            iou = iou_xyxy(box, g)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= args.iou_thresh:
            tp[di] = 1
            claimed[img_idx][best_gi] = True
        else:
            fp[di] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    ap50 = compute_ap(recall, precision) if len(all_dets) > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  Total GT: {total_gt}  Total detections (conf>={args.score_floor}): {len(all_dets)}")
    print(f"  mAP50: {ap50*100:.1f}%")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
