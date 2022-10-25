# by Tristan Pinon
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
import sys
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton, QWidget, QApplication, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5 import QtCore

# screen used to open the Robotonomie window
SCREEN_ID = 0

# get the directory where patients folders are stored
IMAGE_DIRECTORY = f"{os.getcwd()}/data"

# reference to webcam
VIDEO_CAPTURE = cv2.VideoCapture(0)

# get the size of the SCREEN
SCREEN = screeninfo.get_monitors()[SCREEN_ID]

knownFaceEncodings = []
knownFaceNames = []

LANG = "fr"
MICROPHONE_NAME = "Stereo"
wakewords = ['robot']
keywords = ['montre', 'montrer']
facewords = ['reconnaissance', 'faciale']
killwords = ['au revoir', 'bientot']
secondPictureWords = ['passe', 'prochaine']

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def processUsersEncodings():                
    persons = os.listdir(folderPath)

    for person in persons:
        gtImage = folderPath+"/"+person+f"/{person}_selfie.png"
        path = folderPath+"/"+person + "/"

        # Save encodings from a new user
        if (f"{person}.npy" not in os.listdir(f"{folderPath}/{person}")):
            computeEncodings(path, gtImage, person)
        readEncodingsFromFile(path, person)
        print(person + " processed")                                            #DEBUG


def computeEncodings(pth, imagePath, personName):
    """
    :param pth:
    :param image_path:
    :param personName:
    :return:
    """
    personImage = face_recognition.load_image_file(imagePath)
    personFaceEncoding = face_recognition.face_encodings(personImage)[0]
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

    faceLocations = face_recognition.face_locations(rgbSmallFrame)
    faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

    faceNames = []
    for faceEncoding in faceEncodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
        name = "Unknown"

        faceDistance = face_recognition.face_distance(knownFaceEncodings, faceEncoding)
        bestMatchIndex = np.argmin(faceDistance)
        if matches[bestMatchIndex]:
            name = knownFaceNames[bestMatchIndex]
            print(name + " reconnu")                                                #DEBUG

        faceNames.append(name)

    if len(faceNames) > 0:
        frame = drawRectangleAroundFace(frame, faceLocations, faceNames)
    
    return faceNames, frame


def convertTextToSpeech(txt, LANG):
    print("Robotonomy : " + txt)                                                                      #DEBUG
    tts = gtts.gTTS(txt, lang=LANG)
    tts.save(f"gtts.mp3")
    playsound(f"gtts.mp3", True)
    os.remove(f"gtts.mp3")


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
    cosineMax
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


def resizeKeepingAspectRatio(label, img, width = None, height = None):
    (h, w) = (img.height(), img.width())

    if width is None:
        r = height / float(h)
        width = int(w * r)

    else:
        r = width / float(w)
        height = int(h * r)

    label.resize(width, height)
    return (width, height)


def getMicrophoneIndex(MICROPHONE_NAME):
    for i, s in enumerate(sr.Microphone.list_microphone_names()):
        if MICROPHONE_NAME in s:
            return i


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Robot Narratif")
        self.setFixedSize(SCREEN.height, SCREEN.width)

        self.automatic = True
        self.goSecondPictureFaceRecog = False

        backImg = QPixmap(f'{os.getcwd()}/background/PepperHead.png')
        self.background = QLabel(self)
        self.background.setPixmap(backImg)
        resizeKeepingAspectRatio(self.background, backImg, height=SCREEN.height, width=SCREEN.width)
        self.background.move(0, 0)
        self.background.setScaledContents(True)

        self.setFullscreen = QShortcut("f", self)
        self.setFullscreen.activated.connect(self.switchFullscreen)

        self.camera = QLabel(self)
        self.photo1 = QLabel(self)
        self.photo2 = QLabel(self)
        self.photo3 = QLabel(self)

        self.photo = (self.photo1, self.photo2, self.photo3)

        self.threads = { 'heyListenThread' : threading.Thread(target=self.heyListen, args=[r,m]) }
    
        for thread in self.threads.values():
            thread.start()

        self.recognizedFaces = []
        self.facesFrame = None


    def convert_cv_qt(self, cv_img):                 #Convert from an opencv image to QPixmap
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    
    def switchFullscreen(self):
        if self.windowState() & QtCore.Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()


    def heyListenRecog(self, audio):            #Recognizer for the wake up function
        try:
            speechAsText = r.recognize_google(audio, language="fr-FR")
            print("user : " + speechAsText)                                                         #DEBUG
            for wakeword in wakewords:
                if speechAsText.count(wakeword) > 0:
                    self.threads['mainSpeechRecogThread'] = threading.Thread(target=self.mainSpeechRecog, args=[True])   

        except sr.UnknownValueError:
            print("user : ...")
        except sr.RequestError as e:
            print("Le service Google Speech API a rencontré une erreur " + format(e))                            #DEBUG


    def recognize(self, r, m, text):            #base speech recognizer for everything but wake-up functions
        try :
            convertTextToSpeech(text, LANG)
            with m as source:
                audio = r.listen(source, phrase_time_limit=5)
            speechAsText = r.recognize_google(audio, language="fr-FR")
            print("user : " + speechAsText)                                                     #DEBUG
            return speechAsText

        except sr.UnknownValueError:
            pass


    def destroyWindows(self):               #clear the photo labels
        self.photo1.clear()
        self.photo2.clear()
        self.photo3.clear()
        self.camera.clear()
        


    def heyListen(self, r, audio):                  #Wait for the user to use wakewords to wake up the system
        while True:
            if 'mainSpeechRecogThread' in self.threads.keys():
                self.threads['mainSpeechRecogThread'].start()
                print("Waiting")                                                                        #DEBUG
                self.threads['mainSpeechRecogThread'].join()
                print("done!")                                                                          #DEBUG
                del self.threads['mainSpeechRecogThread']

            if 'mainFaceRecogThread' in self.threads.keys():                        #Thread utilisé uniquement lors de triche
                self.threads['mainFaceRecogThread'].start()
                print("Waiting")
                self.threads['mainFaceRecogThread'].join()
                print("done")
                del self.threads['mainFaceRecogThread']

            if 'mainSpeechResearchThread' in self.threads.keys():                   #Thread utilisé uniquement lors de triche
                self.threads['mainSpeechResearchThread'].start()
                print('Waiting')
                self.threads['mainSpeechResearchThread'].join()
                print('done')
                del self.threads['mainSpeechResearchThread']
            if self.automatic:
                with m as source:
                    audio = r.listen(source, phrase_time_limit=3)
                
                threading.Thread(target=self.heyListenRecog, args=[audio]).start()


    def mainFaceRecog(self, profile=None):          #Face recog function (2 subjects and similarity to show pictures)

        if profile != None:
            faceNames = profile
            ret, frameOr = VIDEO_CAPTURE.read()
            img = self.convert_cv_qt(frameOr)

        else:
            while len(self.recognizedFaces) < 2:
                self.recognizedFaces, self.facesFrame = recognizeFace()
                print(self.recognizedFaces)

            faceNames = self.recognizedFaces.copy()
            img = self.convert_cv_qt(self.facesFrame)      

        print(faceNames)
        self.camera.setPixmap(img)
        (width, height) = resizeKeepingAspectRatio(self.camera, img, width=400)
        self.camera.move((SCREEN.width - width)//2, 10)
        self.camera.setScaledContents(True)

        convertTextToSpeech(f"Ravie de vous revoir, {faceNames[0]} et {faceNames[1]}", LANG)

        similarite = getSimilarFile(faceNames[0], faceNames[1])

        convertTextToSpeech("Je vais vous montrer deux images similaires parmi celles que vous m'aviez données, une image chacun.", LANG)
        img = list()
        for i in range(2):
            convertTextToSpeech(f"Voyons l'image de {faceNames[i]}", LANG)

            # Photo
            img.append(QPixmap(folderPath +'/'+ faceNames[i] + '/' + similarite[i][0])) 
            
            self.photo[i].setPixmap(img[i])
            (width, height) = resizeKeepingAspectRatio(self.photo[i], img[i], height=int(SCREEN.height*0.9))
            self.photo[i].move((SCREEN.width - width)//2, (SCREEN.height - height)//2)
            self.photo[i].setScaledContents(True)

            # Title
            imgTitle = similarite[i][1]
            convertTextToSpeech(f"{faceNames[i]}, vous souvenez-vous de cette image ? Elle est intitulée : {imgTitle}", LANG)

            # Text
            imgText = similarite[i][2]
            convertTextToSpeech("Je peux vous rappeler ce que vous m'aviez dit à son sujet", LANG)
            convertTextToSpeech(f"Vous m'aviez dit : {imgText}", LANG)

            if i == 0:
                convertTextToSpeech("Je vous laisse discuter, lorsque vous voudrez passer à la photo suivante, dites passe, ou, prochaine.", LANG)
                self.goSecondPicture()

        (width, height) = resizeKeepingAspectRatio(self.photo1, img[0], width=700)
        self.photo1.move(10, SCREEN.height - height - 30)
        (width, height) = resizeKeepingAspectRatio(self.photo2, img[1], width=700)
        self.photo2.move(((SCREEN.width - width)) + 10 - 20, SCREEN.height - height - 30)

        end = f"{faceNames[0]}, et {faceNames[1]}, J'ai remarqué que vous aviez des intérêts communs, vous pouvez " \
            "discuter et partager ces intérêts l'un avec l'autre, pendant ce temps, je peux vous montrer d'autres photos. " \
            "Si vous me dites : Robot, montre-moi une photo de vélo, je vous montrerais une photo de vélo parmi vos photos. "
        convertTextToSpeech(end, LANG)
        
    
    def imageSpeechResearch(self, speechAsText, profile=None):          #Vocal image research function ("Show me ...")

        if profile != None:
            faceNames = profile

        else:
            while len(self.recognizedFaces) < 1:
                self.recognizedFaces, self.facesFrame = recognizeFace()
            faceNames = self.recognizedFaces.copy()

        researchedName = ''

        if len(faceNames) == 2:
            researchedName = self.recognize(r, m, f"Depuis les images de {faceNames[0]} ? Ou bien celles de {faceNames[1]} ?").upper()
        elif len(faceNames) == 1:
            researchedName = faceNames[0]
            faceNames.append('')


        convertTextToSpeech(f"Ok, je cherche dans les dossiers de {researchedName}", LANG)
        for word in researchedName.split():
            if word in (faceNames[0], faceNames[1]):
                word = faceNames[0] if word in faceNames[0] else faceNames[1]
                searchResult = searchPictureFromDescription(word, speechAsText)
                img = QPixmap(folderPath +'/'+ word + '/' + searchResult[0])

                self.photo3.setPixmap(img)
                (width, height) = resizeKeepingAspectRatio(self.photo3, img, height=int(SCREEN.height*0.9))
                self.photo3.move((SCREEN.width - width)//2, (SCREEN.height - height)//2)
                self.photo3.setScaledContents(True)
                convertTextToSpeech(f"Voici la photo, vous m'aviez dit cela à son sujet : {searchResult[2]}", LANG)


    def mainSpeechRecog(self, newChance, forcedText=None):          #"Crossroad" function, it recognizes the user's request
        researchCompleted = False
        
        if forcedText:
            speechAsText = forcedText
        else :
            speechAsText = self.recognize(r, m, "oui ?")
        print(speechAsText)                                             #DEBUG

        while len(self.recognizedFaces) < 1:
            self.recognizedFaces, self.facesFrame = recognizeFace()

        for word in keywords + facewords + killwords :
            if word in speechAsText and not researchCompleted:
                if word in keywords :
                    try:
                        self.photo3.clear()
                    except:
                        pass
                    self.imageSpeechResearch(speechAsText)
                    researchCompleted = True

                elif word in facewords :
                    convertTextToSpeech("Pas de problème, lancement de la reconnaissance faciale.", LANG)
                    self.destroyWindows()
                    self.mainFaceRecog()
                    researchCompleted = True

                elif word in killwords :
                    convertTextToSpeech("Au revoir !", LANG)
                    self.destroyWindows()
                    self.recognizedFaces = []
                    self.facesFrame = None
                    researchCompleted = True

        if not researchCompleted : 
            convertTextToSpeech("Désolée, je n'ai pas compris.", LANG)
            if newChance:
                self.mainSpeechRecog(False)


    def goSecondPicture(self):          #Wait for the user to allow the second picture to appear during the face recognition procedure
        while self.goSecondPictureFaceRecog == False:
            with m as source:
                    audio = r.listen(source, phrase_time_limit=3)
                
            threading.Thread(target=self.goSecondPictureRecog, args=[audio]).start()


    def goSecondPictureRecog(self, audio):          #Recognizer for the goSecondPicture() function
        try:
            speechAsText = r.recognize_google(audio, language="fr-FR")
            print("user : " + speechAsText)                                                         #DEBUG
            for secondPictureWord in secondPictureWords:
                if speechAsText.count(secondPictureWord) > 0:
                    self.goSecondPictureFaceRecog = True

        except sr.UnknownValueError:
            print("user : ...")
        except sr.RequestError as e:
            print("Le service Google Speech API a rencontré une erreur " + format(e))                            #DEBUG


class CheatWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Terminal Intelligent de Triche Supervisée")
        self.setFixedSize(385, 185)

        self.btnForcedText = QPushButton("Force speech recog", self)
        self.btnForcedText.move(265, 10)
        self.btnForcedText.resize(110, 20)
        self.btnForcedText.clicked.connect(self.forceSpeechRecog)

        self.lineForcedText = QLineEdit(self)
        self.lineForcedText.setPlaceholderText("Forcer la reconnaissance vocale")
        self.lineForcedText.move(10, 10)
        self.lineForcedText.resize(250, 20)

        self.btnForceFaceRecog = QPushButton("Force face recog", self)
        self.btnForceFaceRecog.move(265, 35)
        self.btnForceFaceRecog.resize(110, 20)
        self.btnForceFaceRecog.clicked.connect(self.forceFaceRecog)

        self.lineForcedFaces = QLineEdit(self)
        self.lineForcedFaces.setPlaceholderText("Exemple: Tristan, Julien (FACULTATIF)")
        self.lineForcedFaces.move(10, 35)
        self.lineForcedFaces.resize(250, 20)

        self.btnForceGoSecondPicture = QPushButton("Go to second picture", self)
        self.btnForceGoSecondPicture.move(10, 60)
        self.btnForceGoSecondPicture.resize(110,20)
        self.btnForceGoSecondPicture.clicked.connect(self.forceGoSecondPicture)

        self.btnForceSpeechResearch = QPushButton("Force pic research", self)
        self.btnForceSpeechResearch.move(265, 85)
        self.btnForceSpeechResearch.resize(110, 20)
        self.btnForceSpeechResearch.clicked.connect(self.forceSpeechResearch)

        self.lineForcedSpeechResearch = QLineEdit(self)
        self.lineForcedSpeechResearch.setPlaceholderText("Recherche vocale")
        self.lineForcedSpeechResearch.move(10, 85)
        self.lineForcedSpeechResearch.resize(122, 20)

        self.lineForcedSpeechResearchFace = QLineEdit(self)
        self.lineForcedSpeechResearchFace.setPlaceholderText("Profil(s) (FACULTATIF)")
        self.lineForcedSpeechResearchFace.move(138, 85)
        self.lineForcedSpeechResearchFace.resize(122, 20)

        self.currentFaces = QLabel(self)
        self.currentFaces.move(10, 110)
        self.currentFaces.setText("Current Faces : ")
        self.currentFaces.resize(190, 20)

        self.btnRefreshCurrentFaces = QPushButton("Refresh", self)
        self.btnRefreshCurrentFaces.move(265, 110)
        self.btnRefreshCurrentFaces.resize(110, 20)
        self.btnRefreshCurrentFaces.clicked.connect(self.refreshCurrentFaces)

        self.lineForcedTTS = QLineEdit(self)
        self.lineForcedTTS.setPlaceholderText("Text to speech")
        self.lineForcedTTS.move(10, 135)
        self.lineForcedTTS.resize(250, 20)

        self.btnForceTTS = QPushButton("Force TTS", self)
        self.btnForceTTS.move(265, 135)
        self.btnForceTTS.resize(110, 20)
        self.btnForceTTS.clicked.connect(lambda : convertTextToSpeech(self.lineForcedTTS.text(), LANG))

        self.btnForceByeBye = QPushButton("ByeBye (IDLE ONLY)", self)
        self.btnForceByeBye.move(10, 160)
        self.btnForceByeBye.resize(180, 20)
        self.btnForceByeBye.clicked.connect(self.forceGoodbye)

        self.btnSwitchManualAuto = QPushButton("Manual", self)
        self.btnSwitchManualAuto.move(195, 160)
        self.btnSwitchManualAuto.resize(180, 20)
        self.btnSwitchManualAuto.clicked.connect(self.switchAutoRun)


    def forceSpeechRecog(self):
        window.threads['mainSpeechRecogThread'] = threading.Thread(target=window.mainSpeechRecog, args=[True, self.lineForcedText.text()])
        print("Forced speech recog : " + self.lineForcedText.text())


    def forceFaceRecog(self):
        convertTextToSpeech("Pas de problème, lancement de la reconnaissance faciale.", LANG)
        window.destroyWindows()
        input = self.lineForcedFaces.text()
        forcedFaceNames = input.upper().replace(' ', '').split(",")
        window.threads['mainFaceRecogThread'] = threading.Thread(target=window.mainFaceRecog, args=[forcedFaceNames])
        print("Forced face recog " + input)


    def forceSpeechResearch(self):
        window.photo3.clear()
        forcedSpeechResearchText = self.lineForcedSpeechResearch.text()
        forcedSpeechResearchFace = self.lineForcedSpeechResearchFace.text().upper().replace(' ', '').split(",")
        if len(forcedSpeechResearchFace[0]) < 1:
            forcedSpeechResearchFace = None

        window.threads['mainSpeechResearchThread'] = threading.Thread(target=window.imageSpeechResearch, args=[forcedSpeechResearchText, forcedSpeechResearchFace])
        print("Forced speech research : " + forcedSpeechResearchText)

    def forceGoodbye(self):
        convertTextToSpeech("Au revoir !", LANG)
        window.destroyWindows()
        window.recognizedFaces = []
        window.facesFrame = None


    def switchAutoRun(self):
        window.automatic = False if window.automatic == True else True
        print("Automatic systems : " + str(window.automatic))
        if window.automatic == True:
            self.btnSwitchManualAuto.setText("Manual")
        else:
            self.btnSwitchManualAuto.setText("Automatic")

    def refreshCurrentFaces(self):
        text = "Current Faces :"
        for name in window.recognizedFaces:
            text = text + ' ' + name
        
        print(text)
        self.currentFaces.setText(text)

    def forceGoSecondPicture(self):
        window.goSecondPictureFaceRecog = True


if __name__ == '__main__':
    folderPath = IMAGE_DIRECTORY
    processUsersEncodings()
    recognizeFace()

    r = sr.Recognizer()
    m = sr.Microphone(getMicrophoneIndex(MICROPHONE_NAME))
    with m as source:
        r.adjust_for_ambient_noise(source)
        
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.showFullScreen()

    cheatWindow = CheatWindow()
    cheatWindow.show()

    app.exec()
