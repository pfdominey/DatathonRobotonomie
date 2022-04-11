import face_recognition
import cv2
import numpy as np
import os
import gtts
from playsound import playsound
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import screeninfo

screen_id = 0
is_color = False

# get the directory where patients folders are stored
image_directory = '/home/ubuntu/Desktop/ROBOTONOMIE_TRAINING'

#CAREFUL!!! you need to check the filepath to the images before you run the code


# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height

known_face_encodings = []
known_face_names = []
lang = "fr"
'''
dict_photos = {"VINCENT": 5, "LOUIS": 8}
'''

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
    np.save(pth + f'{person_name}_numpy.npy', person_face_encoding)


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


model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def get_filenames():

    # get patients folder names
    patient_names = os.listdir(image_directory)
    #print("test 1", patient_names)

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
                # print(dir_file)
                if (dir_file.split('_')[3] == "text.txt"):
                    # print(dir_file)
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
        #print(descriptions)

    return (patient_names, descriptions, photo_names, photo_titles)


def create_dataframe(patient_names, array_to_df):
    df = pd.DataFrame(array_to_df).transpose()
    #print(df)
    #print("test 2", patient_names)
    df.columns = patient_names
    return df


def get_similarity(patient_1, patient_2, df_texts, df_photos, df_titles):
    embeddings_patient_1 = np.empty([df_texts[patient_1].dropna().size, 512])
    embeddings_patient_2 = np.empty([df_texts[patient_2].dropna().size, 512])
    #sim_matrix = np.empty([10, 10])
    for i in range(df_texts[patient_1].dropna().size):
        sentence_embedding = model.encode(df_texts[patient_1].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_1[i] = sentence_embedding

    for i in range(df_texts[patient_2].dropna().size):
        sentence_embedding = model.encode(df_texts[patient_2].dropna().iloc[i], convert_to_tensor=True)
        embeddings_patient_2[i] = sentence_embedding

    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embeddings_patient_1, embeddings_patient_2)

    cosine_max=cosine_scores.max().item()
    # print ( cosine_max )
    # print ( cosine_max == cosine_max )
    index_1=-789
    index_2=-345
    for k in range(len(cosine_scores)):
        for j in range(len(cosine_scores[0])):
            if (cosine_scores[k][j].item() == cosine_max):
                # print("Description", patient_1, ":", df_texts['patient_1'].dropna().iloc[i])
                # print("Description", patient_2, ":", df_texts['patient_2'].dropna().iloc[j])
                # print("Similarity Score:", cosine_scores[i][j].item())
                index_1 = k
                index_2 = j

    photo_file1 = df_photos[patient_1].dropna().iloc[index_1]
    photo_title1 = df_titles[patient_1].dropna().iloc[index_1]
    photo_text1 = df_texts[patient_1].dropna().iloc[index_1]

    photo_file2 = df_photos[patient_2].dropna().iloc[index_2]
    photo_title2 = df_titles[patient_2].dropna().iloc[index_2]
    photo_text2 = df_texts[patient_2].dropna().iloc[index_2]

    print(photo_file1)
    print(photo_title1)
    print(photo_file2)
    print(photo_title2)

    ''' return à checker pour la suite'''
    return (photo_file1, photo_title1, photo_text1, photo_file2, photo_title2, photo_text2)


def get_similar_files(patient_1, patient_2):
    patient_names, descriptions, photo_names, photo_titles = get_filenames()
    df_texts = create_dataframe(patient_names, descriptions)
    df_photos = create_dataframe(patient_names, photo_names)
    df_titles = create_dataframe(patient_names, photo_titles)
    photo_file1, photo_title1, photo_text1, photo_file2, photo_title2, photo_text2 = get_similarity(patient_1, patient_2, df_texts, df_photos, df_titles)
    return [[patient_1 + '/'+ photo_file1, patient_1 + '/'+ photo_title1, patient_1 + '/'+ photo_text1], [patient_2 + '/'+ photo_file2, patient_2 + '/'+ photo_title2, patient_2 + '/'+ photo_text2]]


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

    person_face_encod = np.load(pth + f'{person_name}_numpy.npy')
    known_face_encodings.append(person_face_encod)
    known_face_names.append(person_name)


if __name__ == '__main__':

    folder_path = image_directory
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

    img_wl = cv2.imread('/home/ubuntu/Desktop/Cartoon_Robot.png', cv2.IMREAD_COLOR)
    window_name = "Robotonomie"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img_wl)

    wl = f"Salut, "
    convert_text_to_speech(wl, lang, "1")

    for i in range(2):
        test_img = folder_path+"/"+persons[i]+f"/{persons[i]}_selfie_2.png"
        f_name, frame = recognize_face(test_img)
        face_names.append(f_name[0])
        # img = cv2.imread(test_img, cv2.IMREAD_COLOR)
        small_frame = cv2.resize(frame, (500, 500))
        result = f"result{i}"
        cv2.namedWindow(result)
        cv2.moveWindow(result, (i*1500) + 10, 10)
        cv2.imshow(result, small_frame)
        # cv2.imshow("result", small_frame)

    text = f"Bienvenue sur{face_names[0]} et {face_names[1]} "
    convert_text_to_speech(text, lang, "2")

    similarite = get_similar_files(face_names[0], face_names[1])

    for i in range(2):
        # Photo
        place_img = folder_path +'/'+ similarite[i][0]
        print(place_img)
        img2 = cv2.imread(place_img, cv2.IMREAD_COLOR)
        small_frame2 = cv2.resize(img2, (500, 500))

        Place = f"Place{i}"
        cv2.namedWindow(Place)
        #cv2.moveWindow(Place, (i*1500) + 10, 700)
        cv2.imshow(Place, small_frame2)
        # cv2.imshow("Place", small_frame2)

        # Title
        #title_img_path = folder_path +'/'+ similarite[i][1]
        #with open(title_img_path) as f:
        #    title = f.readline()
        title_img = similarite[i][1]
        title_txt_dialog = f"Vous souvenez-vous de cette image ? c'est un {title_img}"
        convert_text_to_speech(title_txt_dialog, lang, "3")

        # Text
        #text_img_path = folder_path +'/'+ similarite[i][2]
        #with open(text_img_path) as f1:
        #    txt = f1.readline()
        text_img = similarite[i][2]
        text_img_dialog = f"Je peux vous en parler et m'en souvenir ensemble"
        convert_text_to_speech(text_img_dialog, lang, "4")
        text_img_dialog2 = f"tu m'as dit avant ça, {text_img}"
        convert_text_to_speech(text_img_dialog2, lang, "5")

        # Goodbye
        # goodby = f"Merci pour votre temps, à bientôt, au revoir {persons[i]}"
        # convert_text_to_speech(goodby, lang, "6")

    end = f"{persons[0]} et {persons[1]}, J'ai remarqué qu'il y a des intérêts communs entre vous, vous pouvez " \
          f"communiquer et partager ces intérêts les uns avec les autres"
    convert_text_to_speech(end, lang, "6")

        # cv2.destroyWindow("result")
        # cv2.destroyWindow("Place")
