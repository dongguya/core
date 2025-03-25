import cv2
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. TorchScript 형식의 YOLOv11 모델 로드 (.torchscript 확장자 사용)
model = torch.jit.load("model/dongguya.torchscript", map_location=device)
model.to(device).eval()

# 2. 전처리 함수
def preprocess_frame(frame, img_size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

# 3. 후처리 함수 (키포인트 포함)
def postprocess_detections(predictions, orig_size, img_size=640, conf_threshold=0.5):
    if not predictions or len(predictions) != 4:
        return []

    boxes, scores, classes, keypoints = predictions

    boxes = boxes.cpu().numpy()[0]
    scores = scores.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    keypoints = keypoints.cpu().numpy()[0]

    # confidence threshold 필터링
    mask = scores >= conf_threshold
    boxes, scores, classes, keypoints = boxes[mask], scores[mask], classes[mask], keypoints[mask]

    if boxes.shape[0] == 0:
        return []

    # 원본 영상 크기로 변환
    orig_h, orig_w = orig_size
    scale_x, scale_y = orig_w / img_size, orig_h / img_size

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    keypoints[:, :, 0] *= scale_x
    keypoints[:, :, 1] *= scale_y

    results = []
    for box, score, cls, kp in zip(boxes, scores, classes, keypoints):
        results.append({
            "box": box,
            "score": score,
            "class": int(cls),
            "keypoints": kp
        })

    return results

# 4. 카메라 열기
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

    # 5. 모델 추론
    with torch.no_grad():
        predictions = model(input_tensor)

        # 구조 및 shape를 정확히 확인
        if isinstance(predictions, (tuple, list)):
            print("출력은 tuple/list이며 길이는:", len(predictions))
            for i, pred in enumerate(predictions):
                if isinstance(pred, torch.Tensor):
                    print(f"predictions[{i}] shape:", pred.shape)
                else:
                    print(f"predictions[{i}] type:", type(pred))
        else:
            print("predictions type:", type(predictions))
            if isinstance(predictions, torch.Tensor):
                print("predictions shape:", predictions.shape)

    # 6. 후처리
    # detections = postprocess_detections(predictions, (orig_h, orig_w))

    # 7. 결과 시각화 (바운딩박스 + 키포인트)
    # for det in detections:
    #     x1, y1, x2, y2 = map(int, det["box"])
    #     conf = det["score"]
    #     cls = det["class"]
    #     keypoints = det["keypoints"]

    #     # 바운딩 박스 표시
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     label = f"Class {cls}: {conf:.2f}"
    #     cv2.putText(frame, label, (x1, y1 - 10), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #     # 키포인트 표시
    #     for kp in keypoints:
    #         kp_x, kp_y = map(int, kp)
    #         cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1)

    # 8. 영상 출력
    # cv2.imshow("Dog Pose Detection: default / sitting / lying", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
