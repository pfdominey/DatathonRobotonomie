import pip

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
                    "SpeechRecognition",
                    "pyqt5",
                    "https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl" ]

for package in needed_packages:
    install(package)