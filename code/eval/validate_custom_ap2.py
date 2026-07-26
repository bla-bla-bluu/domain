"""
Second validation pass: instead of reimplementing inference (which may not exactly
match ultralytics' own preprocessing/letterboxing/NMS pipeline), hook ultralytics'
own validator to capture the exact per-image predictions it uses internally, then
run only the custom AP math on those. This isolates "is the AP formula correct"
from "does the custom inference pipeline match ultralytics'".
"""
import numpy as np
from ultralytics import YOLO
from eval_sahi_map import compute_ap

MODEL = "/home/deepak/domain/weights/baseline_best.pt"

captured = []  # (conf, correct@iou0.5) pairs pulled from ultralytics' own matching

model = YOLO(MODEL)
r = model.val(
    data="/home/deepak/domain/code/forest_config_v2.yaml",
    imgsz=640, conf=0.001, iou=0.6, device="cuda:0", verbose=False, workers=0,
)

# ultralytics stores per-detection (conf, tp-at-each-iou-threshold) in r.box
# r.box.all_ap / confusion etc. don't expose raw per-det arrays directly, but
# stats are on the validator object.
print("Available Metric fields:", [a for a in dir(r.box) if not a.startswith("_")])
print()
print(f"ultralytics mAP50 = {r.box.map50*100:.2f}%  (precision={r.box.mp*100:.2f}%, recall={r.box.mr*100:.2f}%)")
