import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam
from Data_load import *

#载入数据
test_images = load_test_images()
test_labels = load_test_labels()
train_images = load_train_images()
train_labels = load_train_labels()
print(test_images.shape)
print(test_labels.shape)

# 在tensorflow 中，在做卷积的时候需要把数据变成4 维的格式
# 这4 个维度是(数据数量，图片高度，图片宽度，图片通道数)
# 所以这里把数据reshape变成4 维数据，黑白图片的通道数是1，彩色图片通道数是3
test_images = test_images.reshape(-1, 28, 28, 1)
train_images = train_images.reshape(-1, 28, 28, 1)
print(test_images.shape)

#归一化
print(train_images[0][5])
train_images = train_images / 255
test_images = test_images / 255
print(train_images[0][5])

# 把训练集和测试集的标签转为独热编码(one-hot格式)
print(train_labels[0])
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
print(train_labels[0])

# 定义顺序模型
model = Sequential()

# 第一个卷积层
# input_shape 输入数据
# filters 滤波器个数32，生成32 张特征图
# kernel_size 卷积窗口大小5*5
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数（Relu激活函数）
model.add(Convolution2D(
                         input_shape=(28, 28, 1),
                         filters=32,
                         kernel_size=5,
                         strides=1,
                         padding='same',
                         activation='relu'
                        ))

# 第一个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling2D(
                        pool_size=2,
                        strides=2,
                        padding='same',
                      ))

#第二个卷积层
# filters 滤波器个数64，生成64 张特征图
# kernel_size 卷积窗口大小5*5
# strides 步长1
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution2D(
                         filters = 64,
                         kernel_size = 5,
                         strides = 1,
                         padding = 'same',
                         activation = 'relu'
                        ))

#第二个池化层
# pool_size 池化窗口大小2*2
# strides 步长2
# padding padding方式 same/valid
model.add(MaxPooling2D(
                        pool_size= 2,
                        strides= 2,
                        padding= 'same'
))

# 把第二个池化层的输出进行数据扁平化
# 相当于把(64,7,7,64)数据->(64,7*7*64)
model.add(Flatten())

# 第一个全连接层
model.add(Dense(1024, activation= 'relu'))

# Dropout
model.add(Dropout(0.5))

# 第二个全连接层
model.add(Dense(10, activation='softmax'))

# 定义优化器(以Adam方式调整参数)
# leaning_rate 学习率
# 以交叉熵作为损失函数，训练过程中计算准确率
adam = Adam(learning_rate=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# mini-batch大小为64，训练集遍历10次
cnn = model.fit(train_images, train_labels, batch_size=64, epochs=9, validation_data=(test_images, test_labels))
model.summary()
# 保存模型
model.save('cnnmodel_final.h5')
# model.save('model.xml')

# new_model = keras.models.load_model('my_model.h5')
# new_model.compile(optimizer=tf.train.AdamOptimizer(),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])
# new_model.summary()



history_dict = cnn.history
print("---------------------history_dict.keys()------------------:", history_dict.keys())

acc = cnn.history['accuracy']
val_acc = cnn.history['val_accuracy']
loss = cnn.history['loss']
val_loss = cnn.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


