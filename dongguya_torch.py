import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("model/dongguya.torchscript", map_location=device)
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


cap = cv2.VideoCapture(0)
class_labels = ["default", "sitting", "lying"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    input_tensor = preprocess_frame(frame)

    with torch.no_grad():
        preds = model(input_tensor)

    detections = postprocess_yolo(preds, (orig_h, orig_w))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]
        cid = det["class"]
        kp = det["keypoints"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_labels[cid]}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for x, y, pc in kp:
            if pc > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow("Dog Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
