########################图片获取#############################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# cv2.namedWindow("Photo_Detect")  #定义一个窗口
# cap=cv2.VideoCapture(0) #捕获摄像头图像  0位默认的摄像头 笔记本的自带摄像头  1为外界摄像头
# while(True):                  #值为1不断读取图像
#     ret, frame = cap.read()  #视频捕获帧
#     cv2.imwrite('cap_RGB.jpg',frame)     #写入捕获到的视频帧  命名为cap_RGB.jpg
#     cv2.imshow('Photo_Detect',frame)     #显示窗口 查看实时图像
#     #按S 读取灰度图
#     if (cv2.waitKey(1) & 0xFF) == ord('S'): #不断刷新图像，这里是1ms 返回值为当前键盘按键值
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #RGB图像转为单通道的灰度图像
#         gray = cv2.resize(gray,(640,480)) #图像大小为640*480
#         cv2.imshow('cap',gray)              #显示灰度图窗口
#         cv2.imwrite('cap.jpg',gray)         #写入灰度图
#
#     if cv2.waitKey(1) & 0xFF == ord('Q'):   #按Q关闭所有窗口  一次没反应的话就多按几下
#         break
# #执行完后释放窗口
# cap.release()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
####################################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import operator
img_path = 'cap.jpg'
img = cv2.imread(img_path)
img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
clahe = cv2.createCLAHE(3, (8, 8))
dst = clahe.apply(img_Gray)



#model = keras.models.load_model(r'A:\SVM_Project\CNNProject\cnnmodel_final.h5')
# def updateAlpha(x):
#     global alpha, img, img2
#     alpha = cv2.getTrackbarPos('Alpha', 'image')
#     alpha = alpha * 0.01
#     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
# def updateBeta(x):
#     global beta, img, img2
#     beta = cv2.getTrackbarPos('Beta', 'image')
#     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness

def aug(src):
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel
    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    return out

def update(x):
    global  dst,alpha,beta
    alpha = cv2.getTrackbarPos('Alpha', 'image')
    beta = cv2.getTrackbarPos('Beta', 'image')
    ret, dst = cv2.threshold(dst, alpha,beta, cv2.THRESH_BINARY)

# def updateBeta(x):
#     global beta, dst
#     beta = cv2.getTrackbarPos('Beta', 'image')
#     ret, dst = cv2.threshold(dst, alpha, beta, cv2.THRESH_BINARY_INV)
def nothing(x):
    pass

def main():

    cv2.namedWindow('image')
    cv2.createTrackbar('Alpha', 'image', 0, 255,update)
    cv2.createTrackbar('Beta', 'image', 0, 255, update)
    # cv2.setTrackbarPos('Alpha', 'image', 1)
    cv2.setTrackbarPos('Beta', 'image', 255)

    while (True):

        cv2.imshow('image',dst)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    # frame = cv2.imread('cap.jpg')
    # cv2.imshow('frame', frame)  # 显示窗口 查看实时图像
    #
    # frame = aug(frame)
    # cv2.imshow('frame1', frame)  # 显示窗口 查看实时图像
    # cv2.waitKey(0)
    # img_GaussianBlur = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
    # cv2.imshow('img_GaussianBlur', img_GaussianBlur)  # 显示窗口 查看实时图像
    # cv2.waitKey(0)
    #img_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
    # kernel = np.ones((20, 20))
    # img_Open = cv2.morphologyEx(img_Gray, cv2.MORPH_OPEN, kernel)  # 开操作，突出黑
    # cv2.imshow("open",img_Gray)
    # cv2.waitKey(0)
    # clahe = cv2.createCLAHE(3, (8, 8))
    # dst = clahe.apply(img_Gray)
    # cv2.imshow('img_norm', dst)
    # cv2.waitKey(0)

    # ret, img_bin = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 二值化
    #
    # cv2.imshow('threshold', img_bin)
    # cv2.waitKey(0)
    # 闭操作
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # bin_close = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    # # cv2.imshow('bin_close', bin_close)
    # contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取连通区域
    # for cnt in contours:  # 外接矩形
    #     x, y, width, height = cv2.boundingRect(cnt)
    #     if width <= height & height > 40:
    #         img = img_bin[y:y + height, x:x + width]
    #         # cv2.imshow('2', img)
    #         img_R = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    #         images = img_R.reshape(-1, 28, 28, 1)
    #         images = images / 255
    #         prediction = model.predict(images)
    #         prediction = np.array(prediction)
    #         test_pred = np.argmax(prediction, axis=-1)
    #         # print(test_pred)
    #         frame = cv2.putText(frame, str(test_pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    #
    #         cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 1)
    # cv2.imshow("capture", frame)


if __name__ == '__main__':
    main()