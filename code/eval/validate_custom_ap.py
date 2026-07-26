"""Cross-check: compute standard-inference mAP50 with both the custom AP pipeline and
ultralytics' .val() on the same model, and report the difference."""
import glob, os
import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_sahi_map import parse_gt, norm_to_xyxy, iou_xyxy, compute_ap

VAL_IMG = "/home/deepak/domain/data/yolo_val/images"
VAL_LBL = "/home/deepak/domain/data/yolo_val/labels"
MODEL = "/home/deepak/domain/weights/baseline_best.pt"

gt_all = parse_gt(VAL_LBL)
images = sorted(glob.glob(os.path.join(VAL_IMG, "*.jpg")) + glob.glob(os.path.join(VAL_IMG, "*.png")))

model = YOLO(MODEL)

all_dets = []
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

    res = model(img_path, conf=0.001, iou=0.6, imgsz=640, device="cuda", verbose=False)[0]
    for box in res.boxes:
        score = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        all_dets.append((score, img_idx, xyxy))

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
    if best_iou >= 0.5:
        tp[di] = 1
        claimed[img_idx][best_gi] = True
    else:
        fp[di] = 1

cum_tp = np.cumsum(tp)
cum_fp = np.cumsum(fp)
recall = cum_tp / max(total_gt, 1)
precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
custom_ap50 = compute_ap(recall, precision)

print(f"Custom AP implementation, standard-mode baseline: mAP50 = {custom_ap50*100:.2f}%")

r = model.val(data="/home/deepak/domain/code/forest_config_v2.yaml", imgsz=640, conf=0.001, iou=0.6, device="cuda:0", verbose=False, workers=0)
print(f"Ultralytics .val(), standard-mode baseline:       mAP50 = {r.box.map50*100:.2f}%")
print(f"Absolute difference: {abs(custom_ap50*100 - r.box.map50*100):.2f} percentage points")
