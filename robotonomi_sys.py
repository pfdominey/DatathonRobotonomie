import face_recognition
import cv2
import numpy as np
import os
import gtts
from playsound import playsound

known_face_encodings = []
known_face_names = []


def convert_text_to_speech(txt):
    tts = gtts.gTTS(txt, lang="en")
    tts.save(f"{txt}.mp3")
    playsound(f"{txt}.mp3", True)


def compute_encodings(pth, image_path, person_name):
    """

    :param pth:
    :param image_path:
    :param person_name:
    :return:
    """
    person_image = face_recognition.load_image_file(image_path)
    person_face_encoding = face_recognition.face_encodings(person_image)[0]
    np.save(pth + f'{person_name}.npy', person_face_encoding)


def recognize_face(img):
    """

    :param img:
    :return:
    """

    frame_or = cv2.imread(img, cv2.IMREAD_COLOR)
    frame = frame_or.copy()
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom + 10), font, 5, (255, 255, 255), 3)

    print(face_names)
    # Display the resulting image
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # cv2.imshow('Result', small_frame)
    # cv2.waitKey(0)
    return face_names


def visualize_img(test):
    """

    :param test:
    :return:
    """
    frame_or = cv2.imread(test)
    small_frame = cv2.resize(frame_or, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    cv2.imshow("name", small_frame)
    cv2.waitKey(0)


def read_encodings_from_files(pth, person_name):
    """

    :param pth:
    :param person_name:
    :return:
    """

    person_face_encod = np.load(pth + f'{person_name}.npy')
    known_face_encodings.append(person_face_encod)
    known_face_names.append(person_name)


if __name__ == '__main__':
    folder_path = "ROBOTONOMIE_TRAINING"
    persons = os.listdir(folder_path)
    for person in persons:
        print(person)
        gt_image = folder_path+"/"+person+f"/{person}_gt.png"
        path = folder_path+"/"+person + "/"

        # Run only one time at the beginning to save encodings
        # compute_encodings(path, gt_image, person)

        read_encodings_from_files(path, person)

    print("========== DONE! ==========")
    face_names = []

    # pyglet.app.run()
    img_wl = cv2.imread("robot2.jpg", cv2.IMREAD_COLOR)
    # small_frame_img_wl = cv2.resize(img_wl, (0, 0), fx=0.25, fy=0.25)
    # cv2.imshow("Robotonomie", img_wl)

    winname = "Robotonomie"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 900, 10)
    cv2.imshow(winname, img_wl)

    wl = f"Hi, Thanks for contact me, I'm happy to meet you, My name is Robotonomie and I'm here to help you"
    convert_text_to_speech(wl)

    for i in range(2):
        test_img = folder_path+"/"+persons[i]+f"/{persons[i]}_selfie2.png"
        face_names.append(recognize_face(test_img)[0])
        img = cv2.imread(test_img, cv2.IMREAD_COLOR)
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow("result", small_frame)

        text = f"Welcome to challenge number 3, hello {face_names[i]} how are you doing?"
        convert_text_to_speech(text)

        cv2.destroyWindow("result")

        # Photo
        place_img = folder_path + "/" + persons[i] + f"/{persons[i]}_photo_7_image.png"
        img2 = cv2.imread(place_img, cv2.IMREAD_COLOR)
        small_frame2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow("Place", small_frame2)

        # Title
        title_img_path = folder_path + "/" + persons[i] + f"/{persons[i]}_photo_7_title.txt"
        with open(title_img_path) as f:
            title = f.readline()
        title_img = title
        title_txt_dialog = f"Are you remember this image? it's a {title_img}"
        convert_text_to_speech(title_txt_dialog)

        # Text
        text_img_path = folder_path + "/" + persons[i] + f"/{persons[i]}_photo_7_text.txt"
        with open(text_img_path) as f1:
            txt = f1.readline()
        text_img = txt
        text_img_dialog = f"I can tell you about and remember it together"
        convert_text_to_speech(text_img_dialog)
        text_img_dialog2 = f"you told me before that, {text_img}"
        convert_text_to_speech(text_img_dialog2)

        # Goodbye
        goodby = f"Thanks for your time, See you soon, Goodbye {persons[i]}"
        convert_text_to_speech(goodby)

        cv2.destroyWindow("Place")
