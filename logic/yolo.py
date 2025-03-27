import os
import torch
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dongguya.torchscript")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

def preprocess_frame(frame, img_size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)

def box_iou(box1, box2):
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union

def simple_nms(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        current = idxs[0].item()
        keep.append(current)
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def postprocess_yolo(preds, orig_shape, conf_threshold=0.5, iou_threshold=0.5, img_size=640):
    preds = preds.permute(0, 2, 1)[0]
    cls_confidences = preds[:, 4:7]
    cls_conf, cls_ids = cls_confidences.max(dim=1)
    mask = cls_conf > conf_threshold
    if not mask.any():
        return []
    boxes = preds[mask, :4]
    scores = cls_conf[mask]
    classes = cls_ids[mask]
    keypoints = preds[mask, 7:].reshape(-1, 24, 3)
    xy = boxes[:, :2] - boxes[:, 2:] / 2
    wh = boxes[:, :2] + boxes[:, 2:]
    boxes_xyxy = torch.cat((xy, wh), dim=1)
    keep = simple_nms(boxes_xyxy, scores, iou_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]
    keypoints = keypoints[keep]
    orig_h, orig_w = orig_shape
    scale = torch.tensor([orig_w / img_size, orig_h / img_size,
                          orig_w / img_size, orig_h / img_size], device=boxes_xyxy.device)
    boxes_xyxy *= scale
    keypoints[..., 0] *= orig_w / img_size
    keypoints[..., 1] *= orig_h / img_size
    results = []
    for b, s, c, kp in zip(boxes_xyxy, scores, classes, keypoints):
        results.append({
            "box": b.int().tolist(),
            "score": float(s),
            "class": int(c),
            "keypoints": kp.cpu().numpy()
        })
    return results