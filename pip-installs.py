import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

needed_packages = [ "dlib",
                    "face_recognition",
                    "opencv-python",
                    "numpy",
                    "gtts",
                    "playsound",
                    "pandas",
                    "sentence_transformers",
                    "screeninfo" ]

for package in needed_packages:
    install(package)