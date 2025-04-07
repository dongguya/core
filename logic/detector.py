import os
import random
import time
import cv2
from utils.audio import speak
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dongguya.engine")

model = YOLO(MODEL_PATH, task="pose")

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

def choose_command():
    command = random.choice(list(command_config.keys()))
    for audio in command_config[command]["audios"]:
        speak(audio)
    return command, command_config[command]["label"]

def detect_and_verify(cap, target_label_id, hold_sec=1, timeout_sec=60):
    start_time = None
    training_start_time = time.time()

    while time.time() - training_start_time < timeout_sec:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False)
        detected = False  # 초기 상태

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()

            keypoints = None
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls_ids[i])
                label = model.names.get(class_id, str(class_id)) if hasattr(model, 'names') else str(class_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if keypoints is not None:
                    for kp in keypoints[i]:
                        kp_x, kp_y = map(int, kp)
                        cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 0), -1)

                # 검출된 class_id가 목표 label과 일치하는지 확인
                if class_id == target_label_id:
                    detected = True
                    break

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
