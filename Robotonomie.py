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

SCREEN_ID = 0

# get the directory where patients folders are stored
IMAGE_DIRECTORY = f"{os.getcwd()}/data"

#CAREFUL!!! you need to check the filepath to the images before you run the code

#Reference to webcam
VIDEO_CAPTURE = cv2.VideoCapture(0)

# get the size of the SCREEN
SCREEN = screeninfo.get_monitors()[SCREEN_ID]

knownFaceEncodings = []
knownFaceNames = []
LANG = "fr"
MICROPHONE_NAME = "Microphone PC"
wakewords = ['robot']
keywords = ['montre', 'montrer']
facewords = ['reconnaissance', 'faciale']
killwords = ['revoir', 'à bientôt']

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def convertTextToSpeech(txt, LANG):
    tts = gtts.gTTS(txt, LANG=LANG)
    tts.save(f"gtts.mp3")
    playsound(f"gtts.mp3", True)
    os.remove(f"gtts.mp3")
    print("Robotonomy : " + txt)                                                                      #DEBUG


def computeEncodings(pth, imagePath, personName):
    """

    :param pth:
    :param image_path:
    :param personName:
    :return:
    """
    personImage = face_recognition.load_image_file(imagePath)
    personFaceEncoding = face_recognition.faceEncodings(personImage)[0]
    np.save(pth + f'{personName}_numpy.npy', personFaceEncoding)


def recognizeFace():
    """

    :param img:
    :return:
    """

    ret, frameOr = VIDEO_CAPTURE.read()
    frame = frameOr.copy()
    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgbSmallFrame = smallFrame[:, :, ::-1]

    faceLocations = face_recognition.faceLocations(rgbSmallFrame)
    faceEncodings = face_recognition.faceEncodings(rgbSmallFrame, faceLocations)

    faceNames = []
    for faceEncoding in faceEncodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
        name = "Unknown"

        faceDistance = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
        bestMatchIndex = np.argmin(faceDistance)
        if matches[bestMatchIndex]:
            name = knownFaceNames[bestMatchIndex]
            print(name + " reconnu")

        faceNames.append(name)

    frame = drawRectangleAroundFace(frame, faceLocations, faceNames)
    
    return faceNames, frame


def getFilename():

    # get patients folder names
    patientNames = os.listdir(IMAGE_DIRECTORY)

    # set empty arrays to fill with filenames and files content
    descriptions = []
    photoNames = []
    photoTitles = []

    # explore patient folders and fill arrays according to value (image, title or text)
    for patientName in patientNames:
        files = os.listdir(IMAGE_DIRECTORY + '/' + patientName)
        tempList = []
        tempListPhoto = []
        tempListTitle = []
        for file in files:
            if (file.split('_')[1] == "photo"):
                if (file.split('_')[3] == "text.txt"):
                    desc = open(IMAGE_DIRECTORY + '/'+ patientName + "/" + file, encoding='utf-8').read().rstrip('\n')
                    tempList.append(desc)
                elif (file.split('_')[3] == "image.png"):
                    tempListPhoto.append(file)
                elif (file.split('_')[3] == "title.txt"):
                    title = open(IMAGE_DIRECTORY +'/'+ patientName + "/" + file, encoding='utf-8').read().rstrip('\n')
                    tempListTitle.append(title)

        descriptions.append(tempList)
        photoNames.append(tempListPhoto)
        photoTitles.append(tempListTitle)

    return (patientNames, descriptions, photoNames, photoTitles)


def createDataframe(patientNames, arrayToDf):
    df = pd.DataFrame(arrayToDf).transpose()
    df.columns = patientNames
    return df


def getsimilarity(patient_1, patient_2, dfTexts, dfPhotos, dfTitles):
    embeddingsPatient_1 = np.empty([dfTexts[patient_1].dropna().size, 512])
    embeddingsPatient_2 = np.empty([dfTexts[patient_2].dropna().size, 512])
    for i in range(dfTexts[patient_1].dropna().size):
        sentenceEmbedding = model.encode(dfTexts[patient_1].dropna().iloc[i], convert_to_tensor=True)
        embeddingsPatient_1[i] = sentenceEmbedding

    for i in range(dfTexts[patient_2].dropna().size):
        sentenceEmbedding = model.encode(dfTexts[patient_2].dropna().iloc[i], convert_to_tensor=True)
        embeddingsPatient_2[i] = sentenceEmbedding

    # compute similarity scores of two embeddings
    cosineScore = util.pytorch_cos_sim(embeddingsPatient_1, embeddingsPatient_2)

    cosineMax=cosineScore.max().item()
    index_1=-789
    index_2=-345
    for k in range(len(cosineScore)):
        for j in range(len(cosineScore[0])):
            if (cosineScore[k][j].item() == cosineMax):
                index_1 = k
                index_2 = j

    photoFile_1 = dfPhotos[patient_1].dropna().iloc[index_1]
    photoTitle_1 = dfTitles[patient_1].dropna().iloc[index_1]
    photoText_1 = dfTexts[patient_1].dropna().iloc[index_1]

    photoFile_2 = dfPhotos[patient_2].dropna().iloc[index_2]
    photoTitle_2 = dfTitles[patient_2].dropna().iloc[index_2]
    photoText_2 = dfTexts[patient_2].dropna().iloc[index_2]

    ''' return à checker pour la suite'''
    return (photoFile_1, photoTitle_1, photoText_1, photoFile_2, photoTitle_2, photoText_2)


def searchPictureFromDescription(patient, desc):
    patientNames, descriptions, photoNames, photoTitles = getFilename()
    dfTexts = createDataframe(patientNames, descriptions)
    dfPhotos = createDataframe(patientNames, photoNames)
    dfTitles = createDataframe(patientNames, photoTitles)

    embeddingsPatient_1 = np.empty([dfTexts[patient].dropna().size, 512])
    researchedEmbeddings = np.empty([dfTexts[patient].dropna().size, 512])
    for i in range(dfTexts[patient].dropna().size):
        sentenceEmbedding = model.encode(dfTexts[patient].dropna().iloc[i], convert_to_tensor=True)
        embeddingsPatient_1[i] = sentenceEmbedding
        researchedEmbeddings[i] = model.encode(desc, convert_to_tensor=True)
    

    # compute similarity scores of two embeddings
    cosineScore = util.pytorch_cos_sim(embeddingsPatient_1, researchedEmbeddings)

    cosineMax=cosineScore.max().item()
    index_1=-789
    for k in range(len(cosineScore)):
        for j in range(len(cosineScore[0])):
            if (cosineScore[k][j].item() == cosineMax):
                index_1 = k

    photoFile_1 = dfPhotos[patient].dropna().iloc[index_1]
    photoTitle_1 = dfTitles[patient].dropna().iloc[index_1]
    photoText_1 = dfTexts[patient].dropna().iloc[index_1]

    ''' return à checker pour la suite'''
    return (photoFile_1, photoTitle_1, photoText_1)


def getSimilarFile(patient_1, patient_2):
    patientNames, descriptions, photoNames, photoTitles = getFilename()
    dfTexts = createDataframe(patientNames, descriptions)
    dfPhotos = createDataframe(patientNames, photoNames)
    dfTitles = createDataframe(patientNames, photoTitles)
    photoFile_1, photoTitle_1, photoText_1, photoFile_2, photoTitle_2, photoText_2 = getsimilarity(patient_1, patient_2, dfTexts, dfPhotos, dfTitles)
    return [[photoFile_1, photoTitle_1, photoText_1], [photoFile_2, photoTitle_2, photoText_2]]


def drawRectangleAroundFace(frame, faceLocations, faceNames):
    for (top, right, bottom, left), name in zip(faceLocations, faceNames):
        
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left*4, top*4), font, 2, (255, 255, 255), 2)
    return frame


def readEncodingsFromFile(pth, personName):
    """

    :param pth:
    :param personName:
    :return:
    """

    personFaceEncoding = np.load(pth + f'{personName}_numpy.npy')
    knownFaceEncodings.append(personFaceEncoding)
    knownFaceNames.append(personName)


def resizeKeepingAspectRatio(img, width = None, height = None):
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


def processUsersEncodings():
    persons = os.listdir(folderPath)

    for person in persons:
        gtImage = folderPath+"/"+person+f"/{person}_selfie.png"
        path = folderPath+"/"+person + "/"

        # Save encodings from a new user
        if (f"{person}.npy" not in os.listdir(f"{folderPath}/{person}")):
            computeEncodings(path, gtImage, person)
        readEncodingsFromFile(path, person)
        print(person + " processed")


def getMicrophoneIndex(MICROPHONE_NAME):
    for i, s in enumerate(sr.Microphone.list_microphone_names()):
        if MICROPHONE_NAME in s:
            return i


def heyListen(r, audio):
    while True:
        if 'mainSpeechRecogThread' in threads.keys():
            threads['mainSpeechRecogThread'].start()
            print("Waiting")
            threads['mainSpeechRecogThread'].join()
            print("done!")
            del threads['mainSpeechRecogThread']
        with m as source:
            audio = r.listen(source, phrase_time_limit=3)
        
        threading.Thread(target=heyListenRecog, args=[audio]).start()


def heyListenRecog(audio):
    try:
        speechAsText = r.recognize_google(audio, language="fr-FR")
        print("user : " + speechAsText)
        for wakeword in wakewords:
            if speechAsText.count(wakeword) > 0:
                threads['mainSpeechRecogThread'] = threading.Thread(target=main_speech_recog)   

    except sr.UnknownValueError:
        pass

    except sr.RequestError as e:
        print("Le service Google Speech API a rencontré une erreur" + format(e))


def recognize(r, m, text):
    convertTextToSpeech(text, LANG)
    with m as source:
        audio = r.listen(source, phrase_time_limit=5)
    speechAsText = r.recognize_google(audio, language="fr-FR")
    print("user : " + speechAsText)
    return speechAsText


def animatedBackground():
    background_i = 0
    while True:
        img = cv2.imread(f'background/{background_i}.png', cv2.IMREAD_COLOR)
        windowName = "Robotonomie"
        cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(windowName, SCREEN.x - 1, SCREEN.y - 1)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(windowName, img)
        cv2.waitKey(1000)

        background_i = background_i + 1 if background_i <3 else 0


def destroyWindows():
    windows = ("Camera", "Photo1", "Photo2", "Photo3")
    for window in windows:
        try:
            cv2.destroyWindow(window)
        except:
            pass


def mainFaceRecog():
    faceNames, frame = recognizeFace()

    if len(faceNames) == 2:

        smallFrame = resizeKeepingAspectRatio(frame, width=500)
        cv2.namedWindow("Camera")
        cv2.moveWindow("Camera", (SCREEN.width - smallFrame.shape[1])//2, 10)
        cv2.imshow("Camera", smallFrame)
        cv2.waitKey(1)

        convertTextToSpeech(f"Ravie de vous revoir, {faceNames[0]} et {faceNames[1]}", LANG)

        similarite = getSimilarFile(faceNames[0], faceNames[1])

        convertTextToSpeech("Je vais vous montrer deux images similaires parmis celles que vous m'aviez données, une image chacun.")
        for i in range(2):
            convertTextToSpeech(f"Voyons l'image de {faceNames[i]}")

            # Photo
            placeImg = folderPath +'/'+ faceNames[i] + '/' + similarite[i][0]
            frame_2 = cv2.imread(placeImg, cv2.IMREAD_COLOR)
            smallFrame_2 = resizeKeepingAspectRatio(frame_2, width=700)

            place = f"Photo{i+1}"
            cv2.namedWindow(place)
            cv2.moveWindow(place, (i*(SCREEN.width - smallFrame_2.shape[1])) + 10 - 20 * i, SCREEN.height - smallFrame_2.shape[0] - 30)
            cv2.imshow(place, smallFrame_2)
            cv2.waitKey(1)

            # Title
            imgTitle = similarite[i][1]
            convertTextToSpeech(f"{faceNames[i]}, vous souvenez-vous de cette image ? Elle est intitulée : {imgTitle}", LANG)

            # Text
            imgText = similarite[i][2]
            convertTextToSpeech("Je peux vous rappeler ce que vous m'aviez dit à son sujet", LANG)
            convertTextToSpeech(f"Vous m'aviez dit : {imgText}", LANG)

        end = f"{faceNames[0]}, et {faceNames[1]}, J'ai remarqué que vous aviez des intérêts communs, vous pouvez " \
            "discuter et partager ces intérêts l'un avec l'autre, pendant ce temps, je peux vous montrer d'autres photos. " \
            "Si vous me dites : Robot, montre-moi une photo de vélo, je vous montrerais une photo de vélo parmis vos photos. " \
            "Quand vous aurez fini, dites simplement Robot, au revoir ! "
        convertTextToSpeech(end, LANG)


def imageSpeechResearch(speechAsText):
    faceNames = list()
    convertTextToSpeech("Je cherche cela dans mes dossiers, laissez-moi juste le temps de vous reconnaître.", LANG)

    while len(faceNames) < 1:
        faceNames, frame = recognizeFace()
        print(faceNames)
        print(len(faceNames))
    
    if len(faceNames) >= 2:
        researchedName = recognize(r, m, f"Depuis les images de {faceNames[0]} ? Ou bien celles de {faceNames[1]} ?").upper()
    elif len(faceNames) == 1:
        researchedName = faceNames[0]
        faceNames.append('')

    convertTextToSpeech(f"Ok, je cherche dans les dossiers de {researchedName}", LANG)
    for word in researchedName.split():
        if word in (faceNames[0], faceNames[1]):
            searchResult = searchPictureFromDescription(word, speechAsText)
            img = folderPath +'/'+ word + '/' + searchResult[0]
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            smallFrame = resizeKeepingAspectRatio(frame, width=700)
            cv2.namedWindow("Photo3")
            cv2.moveWindow("Photo3", (SCREEN.width - smallFrame.shape[1])//2, (SCREEN.height - smallFrame.shape[0])//2 )
            cv2.imshow("Photo3", smallFrame)
            cv2.waitKey(1)
            convertTextToSpeech(f"Voici la photo, vous m'aviez dit cela à son sujet : {searchResult[2]}", LANG)


def main_speech_recog():
    researchCompleted = False
    try :
        speechAsText = recognize(r, m, "oui ?")
        print(speechAsText)
    
        for word in killwords + keywords + facewords :
            if word in speechAsText and not researchCompleted:
                if word in killwords : 
                    convertTextToSpeech("Au-revoir !", LANG)
                    destroyWindows()
                    researchCompleted = True

                elif word in keywords :
                    imageSpeechResearch(speechAsText)
                    researchCompleted = True

                elif word in facewords :
                    convertTextToSpeech("Pas de problème, lancement de la reconnaissance faciale.", LANG)
                    destroyWindows()
                    mainFaceRecog()
                    researchCompleted = True
    except sr.UnknownValueError:
        pass
    if not researchCompleted : 
        convertTextToSpeech("Désolée, je n'ai pas compris.", LANG)
            


if __name__ == '__main__':
    folderPath = IMAGE_DIRECTORY
    processUsersEncodings()

    r = sr.Recognizer()
    m = sr.Microphone(getMicrophoneIndex(MICROPHONE_NAME))
    with m as source:
        r.adjust_for_ambient_noise(source)

    threads = { 'animatedBackgroundThread' : threading.Thread(target=animatedBackground),
                'heyListenThread'          : threading.Thread(target=heyListen, args=[r,m]) }
    
    for thread in threads.values():
        thread.start()
