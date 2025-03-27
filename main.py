from logic.detector import detect_and_verify, choose_command
from utils.audio import speak
from utils.arduino import send_command
import pygame
import cv2

if __name__ == "__main__":
    pygame.mixer.init()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    training_rounds = 5

    try:
        for round_num in range(training_rounds):
            speak("dongu_come_on1.mp3")
            speak("dongu_come_on2.mp3")

            command, label_id = choose_command()

            print(f"[ROUND {round_num + 1}] 명령: {command}")

            success = detect_and_verify(cap, label_id)
            
            if success:
                speak("good_job.mp3")
                send_command()
            else:
                speak("timeover.mp3")
                print("❌ 행동 인식 실패")
    finally:
        pygame.mixer.quit()
        cap.release()
        cv2.destroyAllWindows()