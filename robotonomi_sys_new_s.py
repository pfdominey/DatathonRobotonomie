import face_recognition
import cv2
import numpy as np
import os
import gtts
from playsound import playsound
import screeninfo
import time
import matplotlib.pyplot as plt


screen_id = 0
is_color = False

# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height

known_face_encodings = []
known_face_names = []
lang = "fr"
dict_photos = {"VINCENT": 5, "PETER": 3}


def convert_text_to_speech(txt, lang, ind):
    tts = gtts.gTTS(txt, lang=lang)
    tts.save(f"{ind}.mp3")
    playsound(f"{ind}.mp3", True)


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
        cv2.putText(frame, name, (left, top), font, 2, (255, 255, 255), 3)

    print(face_names)
    # Display the resulting image
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # cv2.imshow('Result', small_frame)
    # cv2.waitKey(0)
    return face_names, frame


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


def draw_rectangle_around_face(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom + 10), font, 0.35, (255, 255, 255), 1)
        return frame


def read_encodings_from_files(pth, person_name):
    """

    :param pth:
    :param person_name:
    :return:
    """

    person_face_encod = np.load(pth + f'{person_name}.npy')
    known_face_encodings.append(person_face_encod)
    known_face_names.append(person_name)


def recognize_2_person_on_senario(persons):
    for i in range(2):
        test_img = folder_path+"/"+persons[i]+f"/{persons[i]}_selfie2.png"
        f_name, frame = recognize_face(test_img)
        face_names.append(f_name[0])

        small_frame = cv2.resize(frame, (500, 500))
        result = f"{face_names[i]}"
        cv2.namedWindow(result)
        cv2.moveWindow(result, (i*1500) + 10, 10)
        cv2.imshow(result, small_frame)

        text = f"Bienvenue sur Robotonomy System, bonjour {face_names[i]} je te reconnais."
        convert_text_to_speech(text, lang, "2")

        time.sleep(10)


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

    img_wl = cv2.imread("robot3.png", cv2.IMREAD_COLOR)
    window_name = "Robotonomie"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img_wl)

    wl = f"Bonjour, je m’appelle Robotonomie et je suis heureuse de vous voir. "
    convert_text_to_speech(wl, lang, "1")

    for i in range(2):
        test_img = folder_path+"/"+persons[i]+f"/{persons[i]}_selfie2.png"
        f_name, frame = recognize_face(test_img)
        face_names.append(f_name[0])

        small_frame = cv2.resize(frame, (500, 500))
        result = f"{face_names[i]}"
        cv2.namedWindow(result)
        cv2.moveWindow(result, (i*1500) + 10, 10)
        cv2.imshow(result, small_frame)

        text = f"Bienvenue sur Robotonomy System, bonjour {face_names[i]} je te reconnais."
        convert_text_to_speech(text, lang, "2")

        if i == 0:
            time.sleep(5)

    text2 = f"Laissez-moi voir si vous avez quelque chose en commun."
    convert_text_to_speech(text2, lang, "3")
    time.sleep(3)
    text3 = f"Oh oui, {face_names[0]} et {face_names[1]} je remarque que vous avez des photos similaires. "
    convert_text_to_speech(text3, lang, "4")

    places = []
    titles = []
    for i in range(2):
        # Photo
        place_img = folder_path + "/" + persons[i] + f"/{persons[i]}_photo_{dict_photos[persons[i]]}_image.png"
        img2 = cv2.imread(place_img, cv2.IMREAD_COLOR)
        small_frame2 = cv2.resize(img2, (500, 500))
        places.append(small_frame2)

        # Title
        title_img_path = folder_path + "/" + persons[i] + f"/{persons[i]}_photo_{dict_photos[persons[i]]}_title.txt"
        with open(title_img_path) as f:
            title = f.readline()
        title_img = title
        titles.append(title_img)

        Place = f"{title_img}"
        cv2.namedWindow(Place)
        cv2.moveWindow(Place, (i*1500) + 10, 700)
        cv2.imshow(Place, small_frame2)

        title_txt_dialog = f"Vous souvenez-vous de cette image {persons[i]}? c'est un {title_img}"
        convert_text_to_speech(title_txt_dialog, lang, "3")

        # Text
        text_img_path = folder_path + "/" + persons[i] + f"/{persons[i]}_photo_{dict_photos[persons[i]]}_text.txt"
        with open(text_img_path) as f1:
            txt = f1.readline()
        text_img = txt
        text_img_dialog = f"Je peux vous en parler et m'en souvenir ensemble"
        convert_text_to_speech(text_img_dialog, lang, "4")

        text_img_dialog2 = f"tu m'as dit avant ça, {text_img}"
        convert_text_to_speech(text_img_dialog2, lang, "5")

        time.sleep(1)
        ext_img_dialog3 = f"Voulez-vous en parler et de nous donner plus de détails"
        convert_text_to_speech(ext_img_dialog3, lang, "6")

        time.sleep(10)

        if i == 0:
            text_img_dialog4 = f"Voici maintenant une photo similaire prise par {persons[i+1]}"
            convert_text_to_speech(text_img_dialog4, lang, "7")

    end = f"{persons[0]} et {persons[1]}, Maintenant, je vais vous laisser parler de vos expériences communes !"
    convert_text_to_speech(end, lang, "8")

    time.sleep(1)

    cv2.destroyWindow(f"{titles[0]}")
    cv2.destroyWindow(f"{titles[1]}")

    Hori = np.concatenate((places[0], places[1]), axis=1)
    final = 'Similar Photos'
    cv2.namedWindow(final)
    cv2.moveWindow(final, 500, 700)
    cv2.imshow(final, Hori)

    time.sleep(10)
    cv2.destroyAllWindows()


        # cv2.destroyWindow("result")
        # cv2.destroyWindow("Place")
