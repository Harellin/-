import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from interface import Ui_MainWindow
from source import run_style_transfer, imshow

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


def resource_path(relative):
    return os.path.join(
        os.environ.get(
            "_MEIPASS2",
            os.path.abspath(".")
        ),
        relative
    )


class Style(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.resize(700, 400)
        self.pushButton.clicked.connect(self.download_img)
        self.pushButton_2.clicked.connect(self.download_img)
        self.pushButton_3.clicked.connect(self.run)
        self.pushButton_3.setEnabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_img1 = False
        self.is_img2 = False
        self.setWindowIcon(QIcon(resource_path("icon.ico")))

    def download_img(self):
        if self.sender() is self.pushButton:
            filename = QFileDialog.getOpenFileName(self, 'Выбрать картинку', '',
                                                   'Картинка (*.jpg);;Картинка (*.png);;Все файлы (*)')[0]
            print(filename)
            self.image = Image.open(filename)
            pixmap_1 = QPixmap(filename)
            self.label.setPixmap(pixmap_1.scaled(500, 500, Qt.KeepAspectRatio))
            self.is_img1 = True
        else:
            filename_2 = QFileDialog.getOpenFileName(self, 'Выбрать картинку', '',
                                                     'Картинка (*.jpg);;Картинка (*.png);;Все файлы (*)')[0]
            self.image_2 = Image.open(filename_2)
            pixmap_2 = QPixmap(filename_2)
            self.label_2.setPixmap(pixmap_2.scaled(500, 500, Qt.KeepAspectRatio))
            self.is_img2 = True
        if self.is_img1 and self.is_img2:
            self.pushButton_3.setEnabled(True)

    def run(self):
        imsize = 256
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        if self.image.size != self.image_2.size:
            x_s = min(self.image.size[0], self.image_2.size[0])
            y_s = min(self.image.size[1], self.image_2.size[1])
            self.image = self.image.resize((x_s, y_s))
            self.image_2 = self.image_2.resize((x_s, y_s))

        self.image = loader(self.image).unsqueeze(0)
        self.style_img = self.image.to(self.device, torch.float)

        self.image_2 = loader(self.image_2).unsqueeze(0)
        self.content_img = self.image_2.to(self.device, torch.float)

        cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        input_img = self.content_img.clone()

        dlg = QMessageBox(self)
        dlg.setWindowTitle('Важная информация')
        dlg.setText('После нажатия на "ок" запустится генерация нового изображения... Просьба немного подождать :)')
        self.statusBar().showMessage('Оптимизация началась. Ждите...')
        self.statusBar().setStyleSheet('background-color: #F0E68C')
        self.pushButton_3.setEnabled(False)
        btn = dlg.exec()
        if btn == QMessageBox.Ok:
            print('Оптимизация началась')

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    self.content_img, self.style_img, input_img, self.device)

        self.statusBar().showMessage('Готово!!!')
        self.statusBar().setStyleSheet('background-color: #00FF7F')

        unloader = transforms.ToPILImage()
        plt.figure()
        imshow(output, unloader, title='Новое творение')
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Style()
    ex.show()
    sys.exit(app.exec())
