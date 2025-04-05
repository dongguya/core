import os
import random
import time
import cv2
from utils.audio import speak
from logic.tensorrt_yolo import infer  # TensorRT 기반 추론 함수만 사용

# 클래스 라벨 정의
class_labels = ["default", "sitting", "lying"]

# 명령 설정
command_config = {
    "앉아": {
        "label": 1,
        "audios": ["sitting1.mp3", "sitting2.mp3", "sitting2.mp3", "sitting1.mp3"]
    },
    "엎드려": {
        "label": 2,
        "audios": ["lying1.mp3", "lying2.mp3", "lying2.mp3", "lying1.mp3"]
    }
}

# 명령 선택 및 오디오 출력
def choose_command():
    command = random.choice(list(command_config.keys()))
    for audio in command_config[command]["audios"]:
        speak(audio)
    return command, command_config[command]["label"]

# 실시간 추론 및 자세 유지 검증
def detect_and_verify(cap, target_label_id, hold_sec=1, timeout_sec=60):
    start_time = None
    training_start_time = time.time()

    while time.time() - training_start_time < timeout_sec:
        ret, frame = cap.read()
        if not ret:
            break

        detections = infer(frame)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = class_labels[det["class"]]
            conf = det["score"]
            kp = det["keypoints"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for x, y, p in kp:
                if p > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        detected = any(det["class"] == target_label_id for det in detections)

        if detected:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= hold_sec:
                return True
        else:
            start_time = None

        if os.environ.get("DISPLAY"):
            cv2.imshow("Dog Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

    return False
