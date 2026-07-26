#!/usr/bin/env python3
"""Split the 150-frame test set into a temporally disjoint validation set (earliest 50) and
test set (latest 100) and write the ultralytics data configs. The validation set is used for
checkpoint selection; the test set is held out for final evaluation."""
import os, shutil

ROOT = "/home/deepak/domain"
SRC_IMG = f"{ROOT}/data/yolo_val/images"
SRC_LBL = f"{ROOT}/data/yolo_val/labels"
DL = f"{ROOT}/deleaked"

frames = sorted(f for f in os.listdir(SRC_IMG) if f.lower().endswith((".jpg", ".png")))
assert len(frames) == 150, f"expected 150 frames, got {len(frames)}"
val_frames = frames[:50]     # earliest ~7 s
test_frames = frames[50:]    # latest ~14 s  (100 frames)

def stem(f):
    return os.path.splitext(f)[0]

def build(split, names):
    idir = f"{DL}/{split}/images"; ldir = f"{DL}/{split}/labels"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    inst = 0
    for f in names:
        shutil.copy2(f"{SRC_IMG}/{f}", f"{idir}/{f}")
        lf = f"{SRC_LBL}/{stem(f)}.txt"
        dst = f"{ldir}/{stem(f)}.txt"
        if os.path.exists(lf):
            shutil.copy2(lf, dst)
            inst += sum(1 for _ in open(lf) if _.strip())
        else:
            open(dst, "w").close()  # background frame
    return inst

if os.path.exists(DL):
    shutil.rmtree(DL)
vi = build("val", val_frames)
ti = build("test", test_frames)
print(f"VAL : {len(val_frames)} frames, {vi} GT instances  ({val_frames[0]}..{val_frames[-1]})")
print(f"TEST: {len(test_frames)} frames, {ti} GT instances  ({test_frames[0]}..{test_frames[-1]})")

# ---- data yamls ----
def yaml(path, train_images, val_images):
    with open(path, "w") as fh:
        fh.write(f"train: {train_images}\nval: {val_images}\nnames:\n  0: person\n")

# training yamls: train = each model's existing training pool, val = VAL split
yaml(f"{DL}/st_train.yaml",     f"{ROOT}/yolo_selftrain/train/images",    f"{DL}/val/images")
yaml(f"{DL}/oracle_train.yaml", f"{ROOT}/yolo_ir_oracle/train/images",    f"{DL}/val/images")
yaml(f"{DL}/v2_train.yaml",     f"{ROOT}/yolo_synthetic_v2/train/images", f"{DL}/val/images")
# eval yaml: val = held-out TEST split
yaml(f"{DL}/test.yaml",         f"{DL}/test/images",                      f"{DL}/test/images")
print("wrote yamls:", ", ".join(sorted(os.listdir(DL))))
