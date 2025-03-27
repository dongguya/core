import os
import torch
import random
import time
import cv2
from utils.audio import speak
from logic.yolo import model, preprocess_frame, postprocess_yolo

class_labels = ["default", "sitting", "lying"]

command_config = {
    "앉아": {
        "label": 1,
        "audios": ["sitting1.mp3", "sitting2.mp3"]
    },
    "엎드려": {
        "label": 2,
        "audios": ["lying1.mp3", "lying2.mp3"]
    }
}

def choose_command():
    command = random.choice(list(command_config.keys()))
    for audio in command_config[command]["audios"]:
        speak(audio)
    return command, command_config[command]["label"]

def detect_and_verify(cap, target_label_id, hold_sec=3, timeout_sec=60):
    start_time = None
    training_start_time = time.time()
    while time.time() - training_start_time < timeout_sec:
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
            cv2.imshow("훈련 중", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
            
    return False