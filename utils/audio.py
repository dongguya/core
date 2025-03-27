import os
import pygame

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

def speak(file_name):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    file_path = os.path.join(AUDIO_DIR, file_name)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
