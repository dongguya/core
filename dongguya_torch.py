import cv2
import torch
import numpy as np

# GPU 사용 여부 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. TorchScript 형식의 YOLO 모델 불러오기
# model/dongguya.pt 파일이 TorchScript 형식으로 저장되어 있어야 함.
model = torch.jit.load("model/dongguya.pt", map_location=device)
model.to(device)
model.eval()

def preprocess_frame(frame, img_size=640):
    """
    전처리: BGR -> RGB 변환, 리사이즈, 정규화, 텐서 변환
    """
    # BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 원하는 입력 크기로 리사이즈 (예: 640x640)
    img = cv2.resize(img, (img_size, img_size))
    # 0~1 범위로 정규화
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW 변환 후 배치 차원 추가
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def postprocess_detections(detections, orig_size, img_size=640, conf_threshold=0.5):
    """
    후처리: 모델 출력(예상: [N, 6] -> [x1, y1, x2, y2, conf, class])를
    원본 영상 크기로 스케일 조정 후, confidence threshold 필터링
    """
    # detections가 텐서라고 가정 (예: shape: [N, 6])
    if detections is None or detections.numel() == 0:
        return []
    detections = detections[0]  # 배치 차원 제거 (모델에 따라 달라질 수 있음)
    
    # confidence threshold로 필터링
    mask = detections[:, 4] >= conf_threshold
    detections = detections[mask]
    if detections.shape[0] == 0:
        return []
    
    # 원본 영상 크기로 스케일 조정
    orig_h, orig_w = orig_size
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    
    detections[:, [0, 2]] *= scale_x
    detections[:, [1, 3]] *= scale_y
    
    return detections.cpu().numpy()

# 2. 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    orig_h, orig_w = frame.shape[:2]
    
    # 3. 프레임 전처리
    input_tensor = preprocess_frame(frame, img_size=640).to(device)
    
    # 4. 모델 추론 (forward pass)
    with torch.no_grad():
        # 모델 출력 형식은 모델에 따라 다름 (여기서는 [batch, N, 6]이라고 가정)
        predictions = model(input_tensor)
    
    # 5. 후처리: 예측 결과를 원본 영상 크기로 스케일 조정 및 confidence threshold 적용
    detections = postprocess_detections(predictions, (orig_h, orig_w), img_size=640, conf_threshold=0.5)
    
    # 6. 결과 그리기 (각 detection은 [x1, y1, x2, y2, conf, class] 형식)
    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 7. 결과 프레임 출력
    cv2.imshow("YOLO Pose Detection (Torch Only)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
