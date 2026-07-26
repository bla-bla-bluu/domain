#!/usr/bin/env python3
"""Pin down the residual 3.5pp (val rect=False 27.3 vs predict 23.8) for the baseline.
Test half-precision and the predict input shape, holding matcher fixed (custom==ult, proven)."""
import os, sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/deepak/domain/code")
from eval_sahi_map import parse_gt, norm_to_xyxy, iou_xyxy, compute_ap

ROOT = "/home/deepak/domain"
VAL_IMG = f"{ROOT}/data/yolo_val/images"; VAL_LBL = f"{ROOT}/data/yolo_val/labels"
MODEL = f"{ROOT}/weights/baseline_best.pt"

def load_gt(images):
    gt_all = parse_gt(VAL_LBL); gtpi=[]; total=0
    for p in images:
        stem=os.path.splitext(os.path.basename(p))[0]; w,h=Image.open(p).size
        b=[norm_to_xyxy(g,w,h) for g in gt_all.get(stem,[])]; gtpi.append(b); total+=len(b)
    return gtpi,total

def ap_custom(dets,gtpi,total,t=0.5):
    d=sorted(dets,key=lambda x:x[0],reverse=True)
    claimed=[np.zeros(len(g),bool) for g in gtpi]; sc=[];tp=[]
    for s,ii,box in d:
        best,bg=0.0,-1
        for gi,g in enumerate(gtpi[ii]):
            if claimed[ii][gi]: continue
            v=iou_xyxy(box,g)
            if v>best: best,bg=v,gi
        hit=best>=t
        if hit: claimed[ii][bg]=True
        sc.append(s); tp.append(1.0 if hit else 0.0)
    order=np.argsort(-np.asarray(sc)); tp=np.asarray(tp)[order]; fp=1.0-tp
    ctp,cfp=np.cumsum(tp),np.cumsum(fp)
    rec=ctp/max(total,1); prec=ctp/np.maximum(ctp+cfp,1e-9)
    return compute_ap(rec,prec)*100

def predict_dets(model, images, half, imgsz=640, rect_native=False):
    dets=[]
    for ii,p in enumerate(images):
        res=model(p, conf=0.001, iou=0.6, imgsz=imgsz, half=half, device="cuda:0", verbose=False)[0]
        for b in res.boxes:
            dets.append((float(b.conf[0]), ii, b.xyxy[0].tolist()))
    return dets

def main():
    from ultralytics import YOLO
    images=sorted(glob.glob(f"{VAL_IMG}/*.jpg")+glob.glob(f"{VAL_IMG}/*.png"))
    gtpi,total=load_gt(images)
    m=YOLO(MODEL)
    # predict input shape probe
    r0=m(images[0], imgsz=640, verbose=False)[0]
    print("predict orig_shape:", r0.orig_shape, flush=True)
    # half on/off for predict
    for half in (False, True):
        ap=ap_custom(predict_dets(m,images,half), gtpi, total)
        print(f"predict half={half!s:5}  mAP50={ap:.2f}", flush=True)
    # half on/off for val(rect=False)
    for half in (False, True):
        r=m.val(data=f"{ROOT}/code/forest_config_v2.yaml", imgsz=640, conf=0.001, iou=0.6,
                rect=False, half=half, device="cuda:0", workers=0, verbose=False, plots=False)
        print(f"val rect=False half={half!s:5}  mAP50={float(r.box.map50)*100:.2f}", flush=True)
    # val rect=True half on/off
    for half in (False, True):
        r=m.val(data=f"{ROOT}/code/forest_config_v2.yaml", imgsz=640, conf=0.001, iou=0.6,
                rect=True, half=half, device="cuda:0", workers=0, verbose=False, plots=False)
        print(f"val rect=True  half={half!s:5}  mAP50={float(r.box.map50)*100:.2f}", flush=True)
    print("PROBE_DONE", flush=True)

if __name__ == "__main__":
    main()
