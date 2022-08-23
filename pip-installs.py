import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

needed_packages = [ "dlib-19.22.99-cp310-cp310-win_amd64.whl"
                    "cmake",
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