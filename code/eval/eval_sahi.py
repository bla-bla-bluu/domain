#!/usr/bin/env python3
"""
SAHI sliced inference evaluation on the 150-frame Golden Test Set.
Usage:
  python eval_sahi.py --model <path/to/best.pt> [--conf 0.15] [--slice 320]
  python eval_sahi.py --model <path/to/best.pt> --no_sahi --conf 0.25

Prints TP, FP, FN, Precision, Recall, F1 across all 150 frames.
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
                    boxes.append([float(x) for x in parts[1:]])  # cx, cy, w, h normalized
        gt[stem] = boxes
    return gt

def iou_xyxy(a, b):
    """IoU between two [x1,y1,x2,y2] pixel boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def norm_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    return [(cx - w/2)*img_w, (cy - h/2)*img_h,
            (cx + w/2)*img_w, (cy + h/2)*img_h]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      required=True)
    ap.add_argument("--conf",       type=float, default=0.15)
    ap.add_argument("--slice",      type=int,   default=320)
    ap.add_argument("--overlap",    type=float, default=0.2)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--no_sahi",    action="store_true")
    args = ap.parse_args()

    VAL_IMG = "/home/deepak/domain/data/yolo_val/images"
    VAL_LBL = "/home/deepak/domain/data/yolo_val/labels"

    gt_all = parse_gt(VAL_LBL)
    images = sorted(glob.glob(os.path.join(VAL_IMG, "*.jpg")) +
                    glob.glob(os.path.join(VAL_IMG, "*.png")))

    print(f"Model:   {args.model}")
    print(f"Images:  {len(images)}")
    print(f"Mode:    {'Standard 640×640' if args.no_sahi else f'SAHI {args.slice}×{args.slice} tiles'}")
    print(f"Conf:    {args.conf}")

    if not args.no_sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=args.model,
            confidence_threshold=args.conf,
            device="cuda",
        )
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)

    total_tp = total_fp = total_fn = 0

    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_norm = gt_all.get(stem, [])

        # Get actual image dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size  # (width, height)

        # Convert GT to pixel xyxy
        gt_boxes = [norm_to_xyxy(g, img_w, img_h) for g in gt_norm]

        if args.no_sahi:
            res = model(img_path, conf=args.conf, imgsz=640, device="cuda", verbose=False)[0]
            # res.boxes.xyxy is already in original pixel space
            pred_boxes = [box.xyxy[0].tolist() for box in res.boxes]
        else:
            result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=args.slice,
                slice_width=args.slice,
                overlap_height_ratio=args.overlap,
                overlap_width_ratio=args.overlap,
                verbose=0,
            )
            pred_boxes = []
            for obj in result.object_prediction_list:
                b = obj.bbox
                pred_boxes.append([b.minx, b.miny, b.maxx, b.maxy])

        # Greedy IoU matching
        matched_gt   = set()
        matched_pred = set()
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_mat = np.zeros((len(pred_boxes), len(gt_boxes)))
            for pi, p in enumerate(pred_boxes):
                for gi, g in enumerate(gt_boxes):
                    iou_mat[pi, gi] = iou_xyxy(p, g)
            for _ in range(min(len(pred_boxes), len(gt_boxes))):
                max_idx = np.argmax(iou_mat)
                pi, gi = divmod(int(max_idx), len(gt_boxes))
                if iou_mat[pi, gi] >= args.iou_thresh:
                    matched_pred.add(pi)
                    matched_gt.add(gi)
                    iou_mat[pi, :] = -1
                    iou_mat[:, gi] = -1
                else:
                    break

        tp = len(matched_gt)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    P  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    R  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

    print(f"\n{'='*45}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {P*100:.1f}%")
    print(f"  Recall:    {R*100:.1f}%")
    print(f"  F1:        {F1*100:.1f}%")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()
