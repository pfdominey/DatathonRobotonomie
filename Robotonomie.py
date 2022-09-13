# edited by Tristan Pinon
# tristan.pinon@reseau.eseo.fr

import face_recognition
import cv2
import numpy as np
import os
import gtts
from playsound import playsound
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import screeninfo
import speech_recognition as sr
import threading

screen_id = 0
is_color = False

# get the directory where patients folders are stored
image_directory = f"{os.getcwd()}/data"

#CAREFUL!!! you need to check the filepath to the images before you run the code

#Reference to webcam
video_capture = cv2.VideoCapture(0)

# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]

known_face_encodings = []
known_face_names = []
lang = "fr"
microphone_name = "Microphone PC"
wakewords = ['robot']
keywords = ['montre', 'montrer']
facewords = ['reconnaissance', 'faciale']
killwords = ['revoir', 'à bientôt']

def convert_text_to_speech(txt, lang):
    tts = gtts.gTTS(txt, lang=lang)
    tts.save(f"gtts.mp3")
    playsound(f"gtts.mp3", True)
    os.remove(f"gtts.mp3")
    print("Robotonomy : " + txt)                                                                      #DEBUG


def compute_encodings(pth, image_path, person_name):
    """

    :param pth:
    :param image_path:
    :param person_name:
    :return:
    """
    person_image = face_recognition.load_image_file(image_path)
    person_face_encoding = face_recognition.face_encodings(person_image)[0]
    np.save(pth + f'{person_name}_numpy.npy', person_face_encoding)


def recognize_face():
    """

    :param img:
    :return:
    """

    ret, frame_or = video_capture.read()
    frame = frame_or.copy()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

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

    frame = draw_rectangle_around_face(frame, face_locations, face_names)
    
    return face_names, frame


model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def get_filenames():

    # get patients folder names
    patient_names = os.listdir(image_directory)

    # set empty arrays to fill with filenames and files content
    descriptions = []
    photo_names = []
    photo_titles = []

    # explore patient folders and fill arrays according to value (image, title or text)
    for patient_name in patient_names:
        dir_files = os.listdir(image_directory + '/' + patient_name)
        temp_list = []
        temp_list_photos = []
        temp_list_titles = []
        for dir_file in dir_files:
            if (dir_file.split('_')[1] == "photo"):
                if (dir_file.split('_')[3] == "text.txt"):
                    desc = open(image_directory + '/'+ patient_name + "/" + dir_file, encoding='utf-8').read().rstrip('\n')
                    temp_list.append(desc)
                elif (dir_file.split('_')[3] == "image.png"):
                    temp_list_photos.append(dir_file)
                elif (dir_file.split('_')[3] == "title.txt"):
                    title = open(image_directory +'/'+ patient_name + "/" + dir_file, encoding='utf-8').read().rstrip('\n')
                    temp_list_titles.append(title)

        descriptions.append(temp_list)
        photo_names.append(temp_list_photos)
        photo_titles.append(temp_list_titles)

    return (patient_names, descriptions, photo_names, photo_titles)


def create_dataframe(patient_names, array_to_df):
    df = pd.DataFrame(array_to_df).transpose()
    df.columns = patient_names
    return df


def get_similarity(patient_1, patient_2, df_texts, df_photos, df_titles):
    embeddings_patient_1 = np.empty([df_texts[patient_1].dropna().size, 512])
    embeddings_patient_2 = np.empty([df_texts[patient_2].dropna().size, 512])
    for i in range(df_texts[patient_1].dropna().size):
        sentence_embedding = model.encode(df_texts[patient_1].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_1[i] = sentence_embedding

    for i in range(df_texts[patient_2].dropna().size):
        sentence_embedding = model.encode(df_texts[patient_2].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_2[i] = sentence_embedding

    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embeddings_patient_1, embeddings_patient_2)

    cosine_max=cosine_scores.max().item()
    index_1=-789
    index_2=-345
    for k in range(len(cosine_scores)):
        for j in range(len(cosine_scores[0])):
            if (cosine_scores[k][j].item() == cosine_max):
                index_1 = k
                index_2 = j

    photo_file1 = df_photos[patient_1].dropna().iloc[index_1]
    photo_title1 = df_titles[patient_1].dropna().iloc[index_1]
    photo_text1 = df_texts[patient_1].dropna().iloc[index_1]

    photo_file2 = df_photos[patient_2].dropna().iloc[index_2]
    photo_title2 = df_titles[patient_2].dropna().iloc[index_2]
    photo_text2 = df_texts[patient_2].dropna().iloc[index_2]

    ''' return à checker pour la suite'''
    return (photo_file1, photo_title1, photo_text1, photo_file2, photo_title2, photo_text2)


def search_picture_from_desc(patient, desc):
    patient_names, descriptions, photo_names, photo_titles = get_filenames()
    df_texts = create_dataframe(patient_names, descriptions)
    df_photos = create_dataframe(patient_names, photo_names)
    df_titles = create_dataframe(patient_names, photo_titles)

    embeddings_patient_1 = np.empty([df_texts[patient].dropna().size, 512])
    embedding_researched = np.empty([df_texts[patient].dropna().size, 512])
    for i in range(df_texts[patient].dropna().size):
        sentence_embedding = model.encode(df_texts[patient].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_1[i] = sentence_embedding
        embedding_researched[i] = model.encode(desc, convert_to_tensor=True)
    

    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embeddings_patient_1, embedding_researched)

    cosine_max=cosine_scores.max().item()
    index_1=-789
    for k in range(len(cosine_scores)):
        for j in range(len(cosine_scores[0])):
            if (cosine_scores[k][j].item() == cosine_max):
                index_1 = k

    photo_file1 = df_photos[patient].dropna().iloc[index_1]
    photo_title1 = df_titles[patient].dropna().iloc[index_1]
    photo_text1 = df_texts[patient].dropna().iloc[index_1]

    ''' return à checker pour la suite'''
    return (photo_file1, photo_title1, photo_text1)


def get_similar_files(patient_1, patient_2):
    patient_names, descriptions, photo_names, photo_titles = get_filenames()
    df_texts = create_dataframe(patient_names, descriptions)
    df_photos = create_dataframe(patient_names, photo_names)
    df_titles = create_dataframe(patient_names, photo_titles)
    photo_file1, photo_title1, photo_text1, photo_file2, photo_title2, photo_text2 = get_similarity(patient_1, patient_2, df_texts, df_photos, df_titles)
    return [[photo_file1, photo_title1, photo_text1], [photo_file2, photo_title2, photo_text2]]


def draw_rectangle_around_face(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left*4, top*4), font, 2, (255, 255, 255), 2)
    return frame


def read_encodings_from_files(pth, person_name):
    """

    :param pth:
    :param person_name:
    :return:
    """

    person_face_encod = np.load(pth + f'{person_name}_numpy.npy')
    known_face_encodings.append(person_face_encod)
    known_face_names.append(person_name)

def resize_keeping_aspect_ratio(img, width = None, height = None):
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(img, dim)

    return resized

def process_users_encoding():
    persons = os.listdir(folder_path)

    for person in persons:
        gt_image = folder_path+"/"+person+f"/{person}_selfie.png"
        path = folder_path+"/"+person + "/"

        # Save encodings from a new user
        if (f"{person}.npy" not in os.listdir(f"{folder_path}/{person}")):
            compute_encodings(path, gt_image, person)
        read_encodings_from_files(path, person)

def get_microphone_index(microphone_name):
    for i, s in enumerate(sr.Microphone.list_microphone_names()):
        if microphone_name in s:
            return i

def hey_listen(r, audio):
    while True:
        if 'mainSpeechRecogThread' in threads.keys():
            threads['mainSpeechRecogThread'].start()
            print("Waiting")
            threads['mainSpeechRecogThread'].join()
            print("done!")
            del threads['mainSpeechRecogThread']
        print("Hey Listen !")
        with m as source:
            audio = r.listen(source, phrase_time_limit=3)
        
        threading.Thread(target=heyListenRecog, args=[audio]).start()
        print("Hey Listen Stop")

def heyListenRecog(audio):
    print("recog")
    try:
        speech_as_text = r.recognize_google(audio, language="fr-FR")
        print("user : " + speech_as_text)
        for wakeword in wakewords:
            if speech_as_text.count(wakeword) > 0:
                threads['mainSpeechRecogThread'] = threading.Thread(target=main_speech_recog)   

    except sr.UnknownValueError:
        print("Oops! Didn't catch that")

    except sr.RequestError as e:
        print("Le service Google Speech API a rencontré une erreur" + format(e))

    print("recog stop")

def recognize(r, m, text):
    convert_text_to_speech(text, lang)
    with m as source:
        audio = r.listen(source, phrase_time_limit=5)
    speech_as_text = r.recognize_google(audio, language="fr-FR")
    print("user : " + speech_as_text)
    return speech_as_text

def animatedBackground():
    background_i = 0
    while True:
        img_wl = cv2.imread(f'background/{background_i}.png', cv2.IMREAD_COLOR)
        window_name = "Robotonomie"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, img_wl)
        cv2.waitKey(1000)

        background_i = background_i + 1 if background_i <3 else 0

def destroy_windows():
    windows = ("Camera", "Photo1", "Photo2", "Photo3")
    for window in windows:
        try:
            cv2.destroyWindow(window)
        except:
            pass

def main_face_recog():
    face_names, frame = recognize_face()

    if len(face_names) == 2:

        small_frame = resize_keeping_aspect_ratio(frame, width=500)
        cv2.namedWindow("Camera")
        cv2.moveWindow("Camera", (screen.width - small_frame.shape[1])//2, 10)
        cv2.imshow("Camera", small_frame)
        cv2.waitKey(1)

        text = f"Ravie de vous revoir, {face_names[0]} et {face_names[1]} "
        convert_text_to_speech(text, lang)

        similarite = get_similar_files(face_names[0], face_names[1])

        for i in range(2):
            # Photo
            place_img = folder_path +'/'+ face_names[i] + '/' + similarite[i][0]
            frame2 = cv2.imread(place_img, cv2.IMREAD_COLOR)
            small_frame2 = resize_keeping_aspect_ratio(frame2, width=700)

            Place = f"Photo{i+1}"
            cv2.namedWindow(Place)
            cv2.moveWindow(Place, (i*(screen.width - small_frame2.shape[1])) + 10 - 20 * i, screen.height - small_frame2.shape[0] - 30)
            cv2.imshow(Place, small_frame2)
            cv2.waitKey(1)

            # Title
            title_img = similarite[i][1]
            title_txt_dialog = f"{face_names[i]}, vous souvenez-vous de cette image ? Elle est intitulée : {title_img}"
            convert_text_to_speech(title_txt_dialog, lang)

            # Text
            text_img = similarite[i][2]
            text_img_dialog = f"Je peux vous rappeler ce que vous m'aviez dit à son sujet"
            convert_text_to_speech(text_img_dialog, lang)
            text_img_dialog2 = f"Vous m'aviez dit : {text_img}"
            convert_text_to_speech(text_img_dialog2, lang)

        end = f"{face_names[0]}, et {face_names[1]}, J'ai remarqué que vous aviez des intérêts communs, vous pouvez " \
            "discuter et partager ces intérêts l'un avec l'autre, pendant ce temps, je peux vous montrer d'autres photos. " \
            "Si vous me dites : Robot, montre-moi une photo de vélo, je vous montrerais une photo de vélo parmis vos photos. " \
            "Quand vous aurez fini, dites simplement Robot, au revoir ! "
        convert_text_to_speech(end, lang)

def main_speech_recog():
    research_completed = False
    try :
        speech_as_text = recognize(r, m, "oui ?")
        print(speech_as_text)
    
        for word in killwords + keywords + facewords :
            if word in speech_as_text and not research_completed:
                if word in killwords : 
                    convert_text_to_speech("Au-revoir !", lang)
                    destroy_windows()
                    research_completed = True
                elif word in keywords :
                    face_names = list()
                    convert_text_to_speech("Je cherche cela dans mes dossiers, laissez-moi juste le temps de vous reconnaître.", lang)

                    while len(face_names) < 1:
                        face_names, frame = recognize_face()
                        print(face_names)
                        print(len(face_names))
                    
                    if len(face_names) >= 2:
                        researched_name = recognize(r, m, f"Depuis les images de {face_names[0]} ? Ou bien celles de {face_names[1]} ?").upper()
                    elif len(face_names) == 1:
                        researched_name = face_names[0]
                        face_names.append('')

                    convert_text_to_speech(f"Ok, je cherche dans les dossiers de {researched_name}", lang)
                    for word in researched_name.split():
                        if word in (face_names[0], face_names[1]):
                            searchResult = search_picture_from_desc(word, speech_as_text)
                            img = folder_path +'/'+ word + '/' + searchResult[0]
                            frame = cv2.imread(img, cv2.IMREAD_COLOR)
                            small_frame = resize_keeping_aspect_ratio(frame, width=700)
                            cv2.namedWindow("Photo3")
                            cv2.moveWindow("Photo3", (screen.width - small_frame.shape[1])//2, (screen.height - small_frame.shape[0])//2 )
                            cv2.imshow("Photo3", small_frame)
                            cv2.waitKey(1)
                            convert_text_to_speech(f"Voici la photo, vous m'aviez dit cela à son sujet : {searchResult[2]}", lang)

                            research_completed = True
                elif word in facewords :
                    convert_text_to_speech("Pas de problème, lancement de la reconnaissance faciale.", lang)
                    main_face_recog()
                    research_completed = True
    except sr.UnknownValueError:
        pass
    if not research_completed : 
        convert_text_to_speech("Désolée, je n'ai pas compris.", lang)
            


if __name__ == '__main__':
    folder_path = image_directory
    process_users_encoding()

    r = sr.Recognizer()
    m = sr.Microphone(get_microphone_index(microphone_name))
    with m as source:
        r.adjust_for_ambient_noise(source)

    threads = { 'animatedBackgroundThread' : threading.Thread(target=animatedBackground),
                'heyListenThread'          : threading.Thread(target=hey_listen, args=[r,m]) }
    
    for thread in threads.values():
        thread.start()
