#!/usr/bin/env python3
"""De-leaked retrain and evaluation. Retrains the self-train, oracle, and CycleGAN v2 models
selecting the checkpoint on a validation split, then evaluates each model's original vs.
de-leaked checkpoint on a held-out test split; an anchor pass on the full 150 frames
reproduces the reported mAP50. Hyperparameters match the main runs."""
import os, json, time, traceback

ROOT = "/home/deepak/domain"
DL = f"{ROOT}/deleaked"
BASE = f"{ROOT}/yolov8n.pt"
RUNS = f"{ROOT}/runs_deleaked"

TRAIN = [
    ("SelfTrain_dl",   f"{DL}/st_train.yaml"),
    ("Oracle_dl",      f"{DL}/oracle_train.yaml"),
    ("Proposed_v2_dl", f"{DL}/v2_train.yaml"),
]
CKPTS = {
    "baseline_LLVIP":  f"{ROOT}/weights/baseline_best.pt",
    "selftrain_ORIG":  f"{ROOT}/runs/SelfTrain/weights/best.pt",
    "selftrain_DL":    f"{RUNS}/SelfTrain_dl/weights/best.pt",
    "oracle_ORIG":     f"{ROOT}/runs/Oracle_ir/weights/best.pt",
    "oracle_DL":       f"{RUNS}/Oracle_dl/weights/best.pt",
    "v2_ORIG":         f"{ROOT}/runs/Proposed_v2/weights/best.pt",
    "v2_DL":           f"{RUNS}/Proposed_v2_dl/weights/best.pt",
}


def main():
    from ultralytics import YOLO
    os.makedirs(RUNS, exist_ok=True)
    with open(f"{DL}/full150.yaml", "w") as fh:
        fh.write(f"train: {ROOT}/data/yolo_val/images\nval: {ROOT}/data/yolo_val/images\nnames:\n  0: person\n")
    log = open(f"{DL}/run.log", "a", buffering=1)

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True); log.write(msg + "\n")

    say("\n===== STAGE 1: DE-LEAKED TRAINING (best.pt selected on VAL-50) =====", time.ctime())
    for name, data in TRAIN:
        t0 = time.time()
        try:
            say(f"[train] {name} <- {data}")
            YOLO(BASE).train(data=data, epochs=50, imgsz=640, batch=0.75, device="cuda:0",
                             workers=16, project=RUNS, name=name, patience=20,
                             verbose=False, plots=False, exist_ok=True)
            say(f"[train] {name} DONE in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            say(f"[train] {name} FAILED: {e}"); say(traceback.format_exc())

    def evaluate(ckpt, data_yaml, tag):
        res = YOLO(ckpt).val(data=data_yaml, imgsz=640, conf=0.001, iou=0.6, device="cuda:0",
                             workers=8, verbose=False, plots=False, project=f"{RUNS}/eval",
                             name=tag, exist_ok=True)
        b = res.box
        return dict(mAP50=round(float(b.map50)*100, 2), mAP50_95=round(float(b.map)*100, 2),
                    precision=round(float(b.mp)*100, 2), recall=round(float(b.mr)*100, 2))

    summary = {"split": {"val": 50, "test": 100}, "TEST100": {}, "FULL150_anchor": {}}

    say("\n===== STAGE 2: EVAL on held-out TEST-100 (orig vs de-leaked, same set) =====")
    for tag, ck in CKPTS.items():
        if not os.path.exists(ck):
            say(f"[eval TEST100] {tag}: MISSING {ck}"); continue
        try:
            r = evaluate(ck, f"{DL}/test.yaml", f"test100_{tag}")
            summary["TEST100"][tag] = r
            say(f"[eval TEST100] {tag:16s} mAP50={r['mAP50']:5.1f}  mAP50-95={r['mAP50_95']:5.1f}  P={r['precision']:5.1f}  R={r['recall']:5.1f}")
        except Exception as e:
            say(f"[eval TEST100] {tag} FAILED: {e}")

    say("\n===== STAGE 3: ANCHOR eval on FULL-150 (reproduce paper mAP50) =====")
    for tag in ["baseline_LLVIP", "selftrain_ORIG", "oracle_ORIG", "v2_ORIG"]:
        try:
            r = evaluate(CKPTS[tag], f"{DL}/full150.yaml", f"full150_{tag}")
            summary["FULL150_anchor"][tag] = r
            say(f"[anchor FULL150] {tag:16s} mAP50={r['mAP50']:5.1f}  (paper: baseline 35.3 / selftrain 86.5 / oracle 96.9 / v2 33.7)")
        except Exception as e:
            say(f"[anchor FULL150] {tag} FAILED: {e}")

    json.dump(summary, open(f"{DL}/summary.json", "w"), indent=2)
    say("\n===== ALL DONE =====", time.ctime())
    say("SUMMARY_JSON " + json.dumps(summary))
    say("SENTINEL_COMPLETE")


if __name__ == "__main__":
    main()
