import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam
from Data_load import *
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import confusion_matrix
import keras

def show_confusion_matrix(confusion, classes, x_rot=-60):
    """
    绘制混淆矩阵
    :param confusion:
    :param classes:
    :param x_rot:
    :param figsize:
    :param save:
    :return:
    """
    # if figsize is not None:
    #     plt.rcParams['figure.figsize'] = figsize

    plt.imshow(confusion, cmap=plt.cm.Oranges)
    indices = range(len(confusion))
    plt.xticks(indices, classes, rotation=x_rot, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)
    plt.colorbar()
    plt.title("Confusion_Matrix")
    plt.xlabel('y_pred')
    plt.ylabel('y_true')

    # 显示数据
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])

    # if save:
    #     plt.savefig("./confusion_matrix.png")
    plt.show()

if __name__ == '__main__':
    #载入数据
    test_images = load_test_images()
    test_labels = load_test_labels()
    train_images = load_train_images()
    train_labels = load_train_labels()
    # print(test_images.shape)
    # print(test_labels.shape)

    # 在tensorflow 中，在做卷积的时候需要把数据变成4 维的格式
    # 这4 个维度是(数据数量，图片高度，图片宽度，图片通道数)
    # 所以这里把数据reshape变成4 维数据，黑白图片的通道数是1，彩色图片通道数是3
    test_images = test_images.reshape(-1, 28, 28, 1)
    train_images = train_images.reshape(-1, 28, 28, 1)
    # print(test_images.shape)

    #归一化
    print(train_images[0][5])
    train_images = train_images / 255
    test_images = test_images / 255
    # print(train_images[0][5])

    # # 把训练集和测试集的标签转为独热编码(one-hot格式)
    # print(train_labels[0])
    # train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    # test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    # print(train_labels[0])

    model = keras.models.load_model('cnnmodel_final.h5')
    prediction = model.predict(test_images)

    prediction = np.array(prediction)
    test_pred = np.argmax(prediction, axis=-1)
    print(test_pred)

    # 转为列向量
    test_ret = test_pred.reshape(-1, )
    test_labels = test_labels.reshape(-1, )

    #计算测试集中各数字个数
    test_num = np.zeros(10)
    for n in range(0, 10):
        test_num[n] = np.sum(test_labels == n)
    # print(test_num)
    # print(np.sum(test_num))

    #计算测试集中各数字正确的个数
    pre_num = np.zeros(10)
    for n in range(0, 10):
        for i in range(0, len(test_ret)):
            if test_ret[i] == test_labels[i] and test_ret[i] == n:
                pre_num[n] += 1
    print(pre_num)

    #显示每一个数字的识别精度
    for n in range(0, 10):
        print('数字 %d 的测试结果展示：'%(n))
        print('测试样本个数为：%d'%(test_num[n]))
        print('正确样本个数为：%d' % (pre_num[n]))
        print('数字 %d 的识别准确率为：%f'%(n,float(pre_num[n]/test_num[n])))
        print('........................................')

    sum = np.sum(pre_num)
    print('总测试结果展示：')
    print('总体样本数为：10000')
    print('正确样本个数为：%d' % (sum))
    print('总体准确率为：',sum/100)

    precision = precision_score(test_labels, test_ret, average='micro')
    recall = recall_score(test_labels, test_ret, average='micro')
    f1_score = f1_score(test_labels, test_ret, average='micro')
    accuracy_score = accuracy_score(test_labels, test_ret)
    print("Precision_score:", precision)
    print("Recall_score:", recall)
    print("F1_score:", f1_score)
    print("Accuracy_score:", accuracy_score)


    # test_sum = (test_ret == test_labels)
    # print(test_sum)
    # acc = test_sum.mean()
    # print(acc)

    confusion = confusion_matrix(test_labels, test_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    show_confusion_matrix(confusion, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(confusion)