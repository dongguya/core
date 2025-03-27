import cv2
from ultralytics import YOLO

# 1. 모델 불러오기 (YOLO Pose 모델)
model = YOLO("../model/dongguya.pt")

# 2. 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open the camera.")
    exit()

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame.")
        break

    # 3. 모델 예측 수행 (ultralytics API 사용)
    results = model.predict(source=frame, verbose=False)

    # 4. 결과 후처리 및 시각화
    # ultralytics YOLO 모델은 결과를 리스트로 반환합니다.
    for result in results:
        # 바운딩 박스 정보 (xyxy 형식)와 클래스 정보 추출
        boxes = result.boxes.xyxy.cpu().numpy()  # shape: (num_instances, 4)
        cls_ids = result.boxes.cls.cpu().numpy()   # shape: (num_instances,)
        
        # keypoints 정보가 있을 경우 추출 (shape: (num_instances, num_keypoints, 2))
        keypoints = None
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls_ids[i])
            # 모델에 names 속성이 있으면 라벨을 사용, 없으면 class id 문자열 사용
            label = model.names.get(class_id, str(class_id)) if hasattr(model, 'names') else str(class_id)
            
            # 바운딩 박스 및 라벨 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # keypoints가 있으면 각 keypoint를 원으로 표시
            if keypoints is not None:
                for kp in keypoints[i]:
                    kp_x, kp_y = map(int, kp)
                    cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 0), -1)

    # 5. 결과 프레임 화면 출력
    cv2.imshow("Dog Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
