from ultralytics import YOLO
import torch

model = YOLO("model/dongguya.pt")
model.export(format="torchscript")  # best.torchscript 생성됨

