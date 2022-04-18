import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from cnn_why_3 import *
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import (QPainter, QPen,QImage, QPixmap)
from PyQt5.QtCore import Qt
from PIL import ImageGrab, Image
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('cnnmodel_final.h5')


class MyLabel(QLabel):
    pos_xy = []

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        # 鼠标移动事件

    def mouseMoveEvent(self, event):
        '''
                    按住鼠标移动事件：将当前点添加到pos_xy列表中
                    调用update()函数在这里相当于调用paintEvent()函数
                    每次update()时，之前调用的paintEvent()留下的痕迹都会清空
                '''
        # 中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)
        self.update()
        # 绘制事件

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.begin(self)

        pen = QPen(Qt.black, 25, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])

                point_start = point_end
        painter.end()

class MyWindow(QMainWindow, Ui_MainWindow):
    img = cv.imread('white.jpg')
    height, width, bytesPerComponent = img.shape
    bytesPerLine = 3 * width
    cv.cvtColor(img, cv.COLOR_BGR2GRAY, img)
    QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)

        self.label_draw = MyLabel(self)
        self.label_draw.setEnabled(True)
        self.label_draw.setMinimumSize(QtCore.QSize(400, 400))
        self.label_draw.setMaximumSize(QtCore.QSize(400, 400))
        self.label_draw.setMouseTracking(False)
        self.label_draw.setFrameShape(QtWidgets.QFrame.Box)
        self.label_draw.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_draw.setLineWidth(2)
        self.label_draw.setText("")
        self.label_draw.setAlignment(QtCore.Qt.AlignCenter)
        self.label_draw.setObjectName("label_draw")

        pixmap = QPixmap.fromImage(self.QImg)
        self.label_draw.setPixmap(pixmap)
        self.label_draw.setCursor(Qt.CrossCursor)
        # self.show()
        self.horizontalLayout_2.addWidget(self.label_draw)

        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)


    def btn_clear_on_clicked(self):  #清除按钮
        self.label_draw.pos_xy = []
        print(self.label_draw.pos_xy)
        pixmap = QPixmap.fromImage(self.QImg)
        self.label_draw.setPixmap(pixmap)
        self.show()
        self.lineEdit_result.setText('')

    def btn_recognize_on_clicked(self):  #识别按钮
        image_old = self.label_draw.grab()
        image = image_old.toImage()
        # fdir, ftype = QFileDialog.getSaveFileName(self, "Save Image","./", "Image Files (*.jpg)")
        # image.save(fdir)
        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB
        arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
        new_image = Image.fromarray(arr)

        # convert to gray
        # new_image.convert("L")
        # new_image.thumbnail((28, 28))
        recognize_result = self.recognize_img(new_image)  # 调用识别函数

        self.lineEdit_result.setText(str(recognize_result))  # 显示识别结果
        self.update()

    def recognize_img(self, img): #识别程序

        def pre_img(image):
            myimage = image.convert('L')  # 转换成灰度图
            myimage = np.array(myimage)
            ret, img1 = cv.threshold(myimage, 100, 255, cv.THRESH_BINARY_INV)
            # cv.namedWindow('img1',0)
            # cv.resizeWindow('img1',600,600)
            # cv.imshow('img1',img1)
            # print(type(img1))
            # print(img1.shape)
            # print(img1.size)
            # cv.waitKey(2)
            kernel1 = np.ones((10, 10), np.uint8)  # 做一次膨胀
            img2 = cv.dilate(img1, kernel1)
            # cv.namedWindow('img2', 0)
            # cv.resizeWindow('img2', 600, 600)
            # cv.imshow('img2', img2)
            '剔除小连通域'
            contours, hierarchy = cv.findContours(img2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # print(len(contours),hierarchy)
            for i in range(len(contours)):
                area = cv.contourArea(contours[i])
                if area < 150:  # '设定连通域最小阈值，小于该值被清理'
                    cv.drawContours(img2, [contours[i]], 0, 0, -1)

            img5 = cv.resize(img2, (28, 28))
            # cv.namedWindow('img5', 0)
            # cv.resizeWindow('img5', 600, 600)
            # cv.imshow('img5', img5)
            return img5

        img_pre = pre_img(img)   #
        # cv.imshow('img_pre', img_pre)
        # 将数据类型由uint8转为float32
        img = img_pre.astype(np.float32)

        # 图片数据归一化
        img = img / 255
        # 进行预测
        img = img.reshape((1, 28, 28, 1))

        prediction = model.predict(img)
        prediction = np.array(prediction)
        maxpre = np.argmax(prediction)



        print(maxpre)

        return maxpre


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())

