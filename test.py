import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import operator

model = keras.models.load_model(r'A:\SVM_Project\CNNProject\cnnmodel_final.h5')
def main():
    cv2.namedWindow("Photo_Detect")  # 定义一个窗口
    cap = cv2.VideoCapture(0)  # 捕获摄像头图像  0位默认的摄像头 笔记本的自带摄像头  1为外界摄像头
    while (True):  # 值为1不断读取图像
        ret, frame = cap.read()  # 视频捕获帧
        img = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
        # cv2.imwrite('cap_RGB.jpg', frame)  # 写入捕获到的视频帧  命名为cap_RGB.jpg
        # cv2.imshow('Photo_Detect', frame)  # 显示窗口 查看实时图像
        # # 按S 读取灰度图
        # if (cv2.waitKey(1) & 0xFF) == ord('S'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
        #     gray = cv2.resize(gray, (640, 480))  # 图像大小为640*480
        #     cv2.imshow('cap', gray)  # 显示灰度图窗口
        #     cv2.imwrite('cap.jpg', gray)  # 写入灰度图
        # if (cv2.waitKey(1) & 0xFF) == ord('S'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
        kernel = np.ones((20, 20))
        imgOpen = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开操作，突出黑
        # imshow("open",imgOpen)

        ret, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 二值化
        # cv2.imshow('threshold', img_bin)
        # 闭操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bin_close = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('bin_close', bin_close)
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取连通区域
        for cnt in contours:  # 外接矩形
            x, y, width, height = cv2.boundingRect(cnt)
            if width <= height & height > 40:
                img = img_bin[y:y + height, x:x + width]
                # cv2.imshow('2', img)
                img_R = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                images = img_R.reshape(-1, 28, 28, 1)
                images = images/255
                prediction = model.predict(images)
                prediction = np.array(prediction)
                test_pred = np.argmax(prediction, axis=-1)
                # print(test_pred)
                frame = cv2.putText(frame, str(test_pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 1)
        cv2.imshow("capture", frame)





        if cv2.waitKey(1) & 0xFF == ord('Q'):  # 按Q关闭所有窗口  一次没反应的话就多按几下
            break
    # 执行完后释放窗口
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
