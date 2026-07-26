#!/usr/bin/env python3
"""Decompose the standard-mode custom-pipeline vs. ultralytics mAP50 gap into (1) rect
letterbox, (2) detection generation, and (3) TP/FP matcher, and compute SAHI-mode mAP50 under
both the custom and ultralytics matching rules. The AP integration is held fixed; only the
matcher is swapped, applied to the same detections."""
import os, sys, glob, json, time
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/deepak/domain/code")
from eval_sahi_map import parse_gt, norm_to_xyxy, iou_xyxy, compute_ap

ROOT = "/home/deepak/domain"
VAL_IMG = f"{ROOT}/data/yolo_val/images"
VAL_LBL = f"{ROOT}/data/yolo_val/labels"
OUT = f"{ROOT}/deleaked/map_pipeline_result.json"
MODELS = {
    "baseline": f"{ROOT}/weights/baseline_best.pt",
    "selftrain": f"{ROOT}/runs/SelfTrain/weights/best.pt",
}

def load_gt(images):
    gt_all = parse_gt(VAL_LBL)
    gtpi, total = [], 0
    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        w, h = Image.open(p).size
        boxes = [norm_to_xyxy(g, w, h) for g in gt_all.get(stem, [])]
        gtpi.append(boxes); total += len(boxes)
    return gtpi, total

def ap_from_tp(scores, tp, total):
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp)[order]; fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / max(total, 1); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    return compute_ap(rec, prec) * 100

def matcher_custom(dets, gtpi, total, t=0.5):
    """confidence-order greedy matcher."""
    d = sorted(dets, key=lambda x: x[0], reverse=True)
    claimed = [np.zeros(len(g), bool) for g in gtpi]
    scores, tp = [], []
    for s, ii, box in d:
        best, bg = 0.0, -1
        for gi, g in enumerate(gtpi[ii]):
            if claimed[ii][gi]:
                continue
            v = iou_xyxy(box, g)
            if v > best:
                best, bg = v, gi
        hit = best >= t
        if hit:
            claimed[ii][bg] = True
        scores.append(s); tp.append(1.0 if hit else 0.0)
    return ap_from_tp(scores, tp, total)

def matcher_ultralytics(dets, gtpi, total, t=0.5):
    """IoU-sorted greedy with uniqueness -- port of ultralytics match_predictions."""
    from collections import defaultdict
    byimg = defaultdict(list)
    for s, ii, box in dets:
        byimg[ii].append((s, box))
    scores, tp = [], []
    for ii, ds in byimg.items():
        gts = gtpi[ii]
        if not gts:
            for s, _ in ds:
                scores.append(s); tp.append(0.0)
            continue
        M = np.zeros((len(ds), len(gts)))
        for i, (s, box) in enumerate(ds):
            for j, g in enumerate(gts):
                M[i, j] = iou_xyxy(box, g)
        correct = np.zeros(len(ds), bool)
        pr = np.argwhere(M >= t)  # (det_i, gt_j)
        if len(pr):
            ious = M[pr[:, 0], pr[:, 1]]
            pr = pr[ious.argsort()[::-1]]
            pr = pr[np.unique(pr[:, 0], return_index=True)[1]]  # unique det
            pr = pr[np.unique(pr[:, 1], return_index=True)[1]]  # unique gt
            correct[pr[:, 0].astype(int)] = True
        for (s, _), c in zip(ds, correct):
            scores.append(s); tp.append(1.0 if c else 0.0)
    return ap_from_tp(scores, tp, total)

def main():
    from ultralytics import YOLO
    images = sorted(glob.glob(f"{VAL_IMG}/*.jpg") + glob.glob(f"{VAL_IMG}/*.png"))
    gtpi, total = load_gt(images)
    print(f"images={len(images)} total_gt={total}", flush=True)
    result = {}
    for mname, mpath in MODELS.items():
        print(f"\n########## {mname} ##########", flush=True)
        r = {}
        model = YOLO(mpath)
        # anchors
        r["val_rectTrue"]  = round(float(model.val(data=f"{ROOT}/code/forest_config_v2.yaml", imgsz=640, conf=0.001, iou=0.6, rect=True,  device="cuda:0", workers=0, verbose=False, plots=False).box.map50)*100, 2)
        r["val_rectFalse"] = round(float(model.val(data=f"{ROOT}/code/forest_config_v2.yaml", imgsz=640, conf=0.001, iou=0.6, rect=False, device="cuda:0", workers=0, verbose=False, plots=False).box.map50)*100, 2)
        # standard predict detections (square letterbox), conf floor 0.001, NMS iou 0.6
        std = []
        for ii, p in enumerate(images):
            res = model(p, conf=0.001, iou=0.6, imgsz=640, device="cuda:0", verbose=False)[0]
            for b in res.boxes:
                std.append((float(b.conf[0]), ii, b.xyxy[0].tolist()))
        r["std_predict_customMatch"] = round(matcher_custom(std, gtpi, total), 2)
        r["std_predict_ultMatch"]    = round(matcher_ultralytics(std, gtpi, total), 2)
        # SAHI detections (320/0.2), conf floor 0.001
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=mpath, confidence_threshold=0.001, device="cuda")
        sahi = []
        t0 = time.time()
        for ii, p in enumerate(images):
            pr = get_sliced_prediction(p, dm, slice_height=320, slice_width=320, overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
            for o in pr.object_prediction_list:
                bb = o.bbox
                sahi.append((o.score.value, ii, [bb.minx, bb.miny, bb.maxx, bb.maxy]))
        r["sahi_customMatch"] = round(matcher_custom(sahi, gtpi, total), 2)
        r["sahi_ultMatch"]    = round(matcher_ultralytics(sahi, gtpi, total), 2)
        r["_sahi_secs"] = round(time.time()-t0, 1)
        for k, v in r.items():
            print(f"  {k:26s} = {v}", flush=True)
        result[mname] = r
    json.dump(result, open(OUT, "w"), indent=2)
    print("\nWROTE", OUT, flush=True)
    print("ANALYSIS_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
