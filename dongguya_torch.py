import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 (TorchScript 형식)
model = torch.jit.load("model/dongguya.torchscript", map_location=device)
model.eval()

def preprocess_frame(frame, img_size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

# 간단한 자체 NMS 구현 (torchvision 제거)
def simple_nms(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort(descending=True)
    keep_boxes = []

    while idxs.numel() > 0:
        current = idxs[0]
        keep_boxes.append(current.item())

        if idxs.numel() == 1:
            break

        ious = box_iou(boxes[current].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_threshold]

    return keep_boxes

# 바운딩 박스 IOU 계산 함수
def box_iou(box1, box2):
    # 교집합 영역 계산
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # 각 박스 면적
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1 + area2 - intersection

    return intersection / union

# YOLO 자체 후처리
def postprocess_yolo(predictions, orig_shape, conf_threshold=0.5, iou_threshold=0.5, img_size=640):
    predictions = predictions.permute(0, 2, 1)[0]  # [8400, 79]

    scores = predictions[:, 4]
    mask = scores > conf_threshold
    predictions = predictions[mask]

    if predictions.shape[0] == 0:
        return []

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5].int()
    keypoints = predictions[:, 7:].reshape(-1, 24, 3)

    # xywh -> xyxy 변환
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes_xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2

    # NMS 수행
    keep = simple_nms(boxes_xyxy, scores, iou_threshold)

    results = []
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]
    keypoints = keypoints[keep]

    # 원본 영상 크기로 스케일링
    orig_h, orig_w = orig_shape
    scale_x, scale_y = orig_w / img_size, orig_h / img_size
    boxes_xyxy[:, [0,2]] *= scale_x
    boxes_xyxy[:, [1,3]] *= scale_y
    keypoints[..., 0] *= scale_x
    keypoints[..., 1] *= scale_y

    for box, score, cls, kp in zip(boxes_xyxy, scores, classes, keypoints):
        results.append({
            "box": box.int().tolist(),
            "score": score.item(),
            "class": cls.item(),
            "keypoints": kp.cpu().numpy()
        })

    return results

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame.")
        break

    orig_h, orig_w = frame.shape[:2]
    input_tensor = preprocess_frame(frame).to(device)

    with torch.no_grad():
        predictions = model(input_tensor)

    detections = postprocess_yolo(predictions, (orig_h, orig_w))

    class_labels = ["default", "sitting", "lying"]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]
        class_id = det["class"]
        keypoints = det["keypoints"]

        # 바운딩 박스 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # 클래스 라벨 + confidence 점수 시각화
        label = f"{class_labels[class_id]}: {conf:.2f}"
        cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 키포인트 시각화
        for kp_x, kp_y, kp_conf in keypoints:
            if kp_conf > 0.5:
                cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0,0,255), -1)

    cv2.imshow("Dog Pose Detection: default / sitting / lying", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
