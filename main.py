from logic.detector import detect_and_verify, choose_command
from utils.audio import speak
from utils.arduino import send_command
import pygame
import cv2

def show_live_preview(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Dog Pose Detection", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            cv2.destroyWindow("Dog Pose Detection")
            break
        elif key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit("❌ 사용자 종료")

if __name__ == "__main__":
    pygame.mixer.init()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    training_rounds = 5

    try:
        show_live_preview(cap)

        for round_num in range(training_rounds):
            speak("dongu_come_on1.mp3")
            speak("dongu_come_on2.mp3")

            command, label_id = choose_command()

            print(f"[ROUND {round_num + 1}] 명령: {command}")

            success = detect_and_verify(cap, label_id)
            
            if success:
                send_command()
                speak("good_job.mp3")
            else:
                speak("timeover.mp3")
                print("❌ 행동 인식 실패")
    finally:
        pygame.mixer.quit()
        cap.release()
        cv2.destroyAllWindows()