# by Tristan Pinon
# tristan.pinon@reseau.eseo.fr

from email.mime import image
import sys
import os
import shutil
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget, QLabel, QLineEdit, QPushButton, QPlainTextEdit, QWidget, QFileDialog, QListWidgetItem, QMessageBox, QApplication
from PyQt5.QtGui import QPixmap


imageDirectory = "data"

def resize_keeping_aspect_ratio(label, img, width = None, height = None):
    (h, w) = (img.height(), img.width())

    if width is None:
        r = height / float(h)
        width = int(w * r)

    else:
        r = width / float(w)
        height = int(h * r)

    label.resize(width, height)
    return (width, height)

# Subclass QMainWindow to customize your application's main window
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()


        self.setWindowTitle("Robotonomie Data Collection")
        self.setFixedSize(600, 660)

        self.listePatients = QListWidget(self)
        self.listePatients.resize(250, 200)
        patients = os.listdir(imageDirectory)
        for i in range(len(patients)):
            self.listePatients.insertItem(i, patients[i])
        self.listePatients.clicked.connect(self.listClicked)
        self.listePatients.move(10,10)

        self.selfie = QLabel(self)

        self.searchBar = QLineEdit(self)
        self.searchBar.setPlaceholderText("Entrez un nom de profil")
        self.searchBar.move(10, 215)
        self.searchBar.resize(200, 20)
        self.searchBar.textEdited.connect(self.search)

        self.btnCreerProfil = QPushButton("Créer", self)
        self.btnCreerProfil.move(215, 215)
        self.btnCreerProfil.resize(45, 20)
        self.btnCreerProfil.clicked.connect(self.createProfile)

        self.labelListePhotos = QLabel("Liste des photos :", self)
        self.labelListePhotos.move(10, 250)

        self.listePhotos = QListWidget(self)
        self.listePhotos.resize(250, 200)
        self.listePhotos.clicked.connect(self.photoClicked)
        self.listePhotos.move(10, 270)

        self.photo = QLabel(self)

        self.labelTitre = QLabel("Titre de la photo :", self)
        self.labelTitre.move(10, 480)

        self.lineTitre = QLineEdit(self)
        self.lineTitre.setPlaceholderText("Entrez un titre pour la photo")
        self.lineTitre.move(10, 500)
        self.lineTitre.resize(250, 20)
        self.lineTitre.setDisabled(True)

        self.labelDesc = QLabel("Description de la photo :", self)
        self.labelDesc.move(10, 530)

        self.textDesc = QPlainTextEdit(self)
        self.textDesc.setPlaceholderText("Entrez une description pour la photo")
        self.textDesc.move(10, 550)
        self.textDesc.resize(580, 100)
        self.textDesc.setDisabled(True)

        self.btnSupprPhoto = QPushButton("Supprimer", self)
        self.btnSupprPhoto.move(525, 500)
        self.btnSupprPhoto.resize(60, 20)
        self.btnSupprPhoto.clicked.connect(self.deletePhoto)
        self.btnSupprPhoto.setDisabled(True)

        self.btnUpload = QPushButton("Nouvelle photo", self)
        self.btnUpload.move(270, 500)
        self.btnUpload.resize(100, 20)
        self.btnUpload.clicked.connect(self.newPhoto)
        self.btnUpload.setDisabled(True)

        self.btnRefresh = QPushButton("Rafraîchir les informations", self)
        self.btnRefresh.move(380, 500)
        self.btnRefresh.resize(135, 20)
        self.btnRefresh.clicked.connect(self.refreshInfo)
        self.btnRefresh.setDisabled(True)

        self.btnSupprProfil = QPushButton("Supprimer", self)
        self.btnSupprProfil.move(200, 240)
        self.btnSupprProfil.resize(60, 20)
        self.btnSupprProfil.clicked.connect(self.deleteProfile)
        self.btnSupprProfil.setDisabled(True)

        self.show()

    def deleteProfile(self):
        shutil.rmtree(f"./{imageDirectory}/{self.patient}")
        self.btnSupprProfil.setDisabled(True)
        self.btnRefresh.setDisabled(True)
        self.btnUpload.setDisabled(True)
        self.btnSupprPhoto.setDisabled(True)
        self.textDesc.setDisabled(True)
        self.lineTitre.setDisabled(True)

        self.selfie.clear()
        self.photo.clear()
        self.listePatients.takeItem(self.listePatients.row(self.listePatients.currentItem()))

    def deletePhoto(self):
        photo = f"./{imageDirectory}/{self.patient}/{self.listePhotos.currentItem().text()}"
        titre = f"{photo[:len(photo)-9]}title.txt"
        desc = f"{photo[:len(photo)-9]}text.txt"

        for file in (photo, titre, desc):
            os.remove(file)

        files = os.listdir(f"{imageDirectory}/{self.patient}")
        photoRelatedFiles = []
        for file in files:
            if "photo" in file:
                photoRelatedFiles.append(file)
        
        for i in range(len(photoRelatedFiles)):
            splittedName = photoRelatedFiles[i].split('_')
            newName = f"{splittedName[0]}_{splittedName[1]}_{str(i//3 + 1)}_{splittedName[3]}"
            os.rename(f"./{imageDirectory}/{self.patient}/{photoRelatedFiles[i]}", f"./{imageDirectory}/{self.patient}/{newName}")

        self.listePhotos.clear()
        self.listClicked()

    def newPhoto(self):
        i=1
        for file in os.listdir(f"{imageDirectory}/{self.patient}"):
            if file.endswith("image.png"):
                i+=1
        filename = QFileDialog.getOpenFileName(self, caption="Sélectionnez une photo", filter="Image files (*.png *.jpg *.jpeg)")
        savename = f"{self.patient}_photo_{i}_image.png"
        shutil.copy(filename[0], f"{imageDirectory}/{self.patient}/{savename}")
        self.listePhotos.insertItem(i, savename)

        titleFile = open(f"{imageDirectory}/{self.patient}/{self.patient}_photo_{i}_title.txt", 'w')
        titleFile.close()
        descFile = open(f"{imageDirectory}/{self.patient}/{self.patient}_photo_{i}_text.txt", 'w')
        descFile.close()

    def refreshInfo(self):
        photo = self.listePhotos.currentItem().text()
        title = self.lineTitre.text()
        desc = self.textDesc.toPlainText()

        titleFile = open(f"{imageDirectory}/{self.patient}/{photo[:len(photo)-9]}title.txt", 'w',encoding='utf-8')
        titleFile.write(title)
        titleFile.close()

        descFile = open(f"{imageDirectory}/{self.patient}/{photo[:len(photo)-9]}text.txt", 'w',encoding='utf-8')
        descFile.write(desc)
        descFile.close()

    def photoClicked(self):
        photo = self.listePhotos.currentItem().text()

        image = QPixmap(f"./{imageDirectory}/{self.patient}/{photo}")
        self.photo.setPixmap(image)
        (width, height) = resize_keeping_aspect_ratio(self.photo, image, height=180)
        self.photo.move(600 - 10 - width, 280)
        self.photo.setScaledContents(True)

        title = open(f"{imageDirectory}/{self.patient}/{photo[:len(photo)-9]}title.txt", encoding='utf-8').read()
        self.lineTitre.setText(title)

        desc = open(f"{imageDirectory}/{self.patient}/{photo[:len(photo)-9]}text.txt", encoding='utf-8').read()
        self.textDesc.setPlainText(desc)

        self.btnRefresh.setDisabled(False)
        self.btnSupprPhoto.setDisabled(False)
        self.lineTitre.setDisabled(False)
        self.textDesc.setDisabled(False)


    def search(self, text):
        model = self.listePatients.model()
        match = model.match(model.index(0, self.listePatients.modelColumn()), Qt.DisplayRole, text, hits=1, flags=Qt.MatchStartsWith)
        if match and text:
            self.listePatients.setCurrentIndex(match[0])
            self.listClicked()

    def createProfile(self):
        name = self.searchBar.text().upper()
        try :
            os.mkdir(f"./{imageDirectory}/{name}")
            item = QListWidgetItem(name)
            self.listePatients.insertItem(len(os.listdir(imageDirectory))+1, item)
        except FileExistsError:
            self.popupFileExist()
        
        filename = QFileDialog.getOpenFileName(self, caption="Sélectionnez un selfie", filter="Image files (*.png *.jpg *.jpeg)")
        savename = f"{name}_selfie.png"
        shutil.copy(filename[0], f"{imageDirectory}/{name}/{savename}")
        self.search(name)

    def popupFileExist(self):
        msg = QMessageBox()
        msg.setWindowTitle("Erreur!")
        msg.setText("Erreur : l'utilisateur existe déjà.")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    def listClicked(self):
        self.patient = self.listePatients.currentItem().text()

        self.photo.clear()
        image = QPixmap(f"./{imageDirectory}/{self.patient}/{self.patient}_selfie.png")
        self.selfie.setPixmap(image)
        (width, height) = resize_keeping_aspect_ratio(self.selfie, image, height=200)
        self.selfie.move(600 - 10 - width, 10)
        self.selfie.setScaledContents(True)

        self.listePhotos.clear()
        fichiersPatient = os.listdir(f"{imageDirectory}/{self.patient}")
        for i in range(len(fichiersPatient)):
            if fichiersPatient[i].endswith(".png") and not fichiersPatient[i].endswith("selfie.png"):
                self.listePhotos.insertItem(i, fichiersPatient[i])

        self.btnUpload.setDisabled(False)
        self.btnRefresh.setDisabled(True)
        self.btnSupprPhoto.setDisabled(True)
        self.lineTitre.clear()
        self.textDesc.clear()
        self.lineTitre.setDisabled(True)
        self.textDesc.setDisabled(True)
        self.btnSupprProfil.setDisabled(False)


app = QApplication(sys.argv)
app.setStyle('Fusion')

window = MainWindow()
window.show()

app.exec()