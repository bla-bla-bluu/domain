#!/usr/bin/env python3
"""Train YOLOv8-Nano on the self-training pseudo-labeled dataset (see generate_pseudo_labels.py).
Identical hyperparameters to the CycleGAN v2 and oracle runs (same architecture and detector
hyperparameters; only the training-data source differs)."""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="/home/deepak/domain/code/selftrain_config.yaml",
        epochs=50, imgsz=640, batch=0.75,
        device="cuda:0", workers=16,
        project="/home/deepak/domain/runs", name="SelfTrain",
        patience=20,
    )
