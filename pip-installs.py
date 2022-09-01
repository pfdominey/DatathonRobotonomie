import pip
import os

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

needed_packages = [ "cmake",
                    "face_recognition",
                    "opencv-python",
                    "numpy",
                    "gtts",
                    "playsound",
                    "pandas",
                    "sentence_transformers",
                    "screeninfo",
                    "PyAudio",
                    "SpeechRecognition" ]

for package in needed_packages:
    install(package)