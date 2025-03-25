import cv2
import torch
import numpy as np
from ultralytics.utils.ops import non_max_suppression, scale_boxes, scale_coords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = torch.jit.load("model/dongguya.torchscript", map_location=device)
model.to(device).eval()

def preprocess_frame(frame, img_size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

# 후처리 (정확한 모델 출력 기반)
def postprocess_yolo(predictions, orig_shape, img_size=640, conf_threshold=0.5, iou_threshold=0.5):
    predictions = predictions.permute(0, 2, 1)

    preds = non_max_suppression(predictions, conf_threshold, iou_threshold, nc=1, max_det=100, kpt_label=True)

    results = []
    if preds[0] is not None:
        pred = preds[0].cpu().numpy()
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = pred[:, 5]
        keypoints = pred[:, 6:].reshape(-1, 24, 3)

        boxes = scale_boxes((img_size, img_size), boxes, orig_shape).astype(int)
        keypoints[..., :2] = scale_coords((img_size, img_size), keypoints[..., :2], orig_shape).astype(int)

        for box, score, cls, kp in zip(boxes, scores, classes, keypoints):
            results.append({
                "box": box,
                "score": score,
                "class": int(cls),
                "keypoints": kp
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

    # 위의 후처리 호출
    detections = postprocess_yolo(predictions, (orig_h, orig_w))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]
        keypoints = det["keypoints"]

        # 박스 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 키포인트 시각화
        for kp in keypoints:
            kp_x, kp_y, kp_conf = kp
            if kp_conf > 0.5:
                cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (0,0,255), -1)

    cv2.imshow("Dog Pose Detection: default / sitting / lying", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
