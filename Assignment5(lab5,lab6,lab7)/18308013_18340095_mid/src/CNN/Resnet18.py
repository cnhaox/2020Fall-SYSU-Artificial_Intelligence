# 最原始的模型
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np


def load_batch(path):
    import pickle
    with open(path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


def my_load_data():
    train_labels = None
    train_images = None
    for i in range(1, 6):
        batch = load_batch('cifar-10-batches-py\\data_batch_' + str(i))
        x = batch[b'data']
        x = np.reshape(x, (10000, 3, 32, 32))
        x = x.transpose((0, 2, 3, 1))
        y = batch[b'labels']
        if i == 1:
            train_images = x
            train_labels = y
        else:
            train_images = np.concatenate([train_images, x])
            train_labels = np.concatenate([train_labels, y])

    test_batch = load_batch('cifar-10-batches-py\\test_batch')
    test_images = test_batch[b'data']
    test_images = np.reshape(test_images, (10000, 3, 32, 32))
    test_images = test_images.transpose((0, 2, 3, 1))
    test_labels = test_batch[b'labels']

    test_labels = np.array(test_labels)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    return (train_images, train_labels), (test_images, test_labels)


# 使用 batch normalization 的 Conv2D 层，无激活函数
def con2d(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay)
                          )(x)
    layer = layers.BatchNormalization()(layer)
    return layer


# 在 conv2d 层上添加激活函数
def con2d_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = con2d(x, filters, kernel_size, weight_decay, strides)
    layer = layers.Activation('relu')(layer)
    return layer


# 残差单元，其中 feature_extend 参数用于确定是否要进行下采样，使 x 可以和 F(x) 相加
def residual_block(x, filters, kernel_size, weight_decay=.0, down_sample=True):
    if down_sample:
        shortcut = con2d(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        shortcut = x
        stride = 1

    simple = con2d_relu(x, filters=filters, kernel_size=kernel_size,
                        weight_decay=weight_decay, strides=stride)
    simple = con2d(simple, filters = filters, kernel_size=kernel_size,
                   weight_decay=weight_decay, strides=1)

    output = layers.add([simple, shortcut])
    output = layers.Activation('relu')(output)
    return output


def build_resnet18_model(class_num, input_shape, weight_decay=1e-4):

    input = layers.Input(shape=input_shape)
    x = input
    # x = con2d_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = con2d_relu(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # conv2
    x = residual_block(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=False)
    x = residual_block(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=False)
    # conv3
    x = residual_block(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=True)
    x = residual_block(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=False)
    # conv4
    x = residual_block(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=True)
    x = residual_block(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=False)
    # conv5
    x = residual_block(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=True)
    x = residual_block(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, down_sample=False)
    x = layers.AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(class_num, activation='softmax')(x)
    model = models.Model(input, x, name='ResNet18')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    (train_images, train_labels), (test_images, test_labels) = my_load_data()

    # 图像保存在float32格式的Numpy张量中
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # 把类别标签转化为one-hot编码
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    val_images = train_images[40000:]
    train_images = train_images[:40000]
    val_labels = train_labels[40000:]
    train_labels = train_labels[:40000]

    model = build_resnet18_model(class_num=10, input_shape=(32, 32, 3))
    history = model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_data=(val_images, val_labels))

    model.save('cifar10_resnet.h5')

    # 绘制训练过程中的损失曲线和精度曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig('resnet_acc.png')

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('resnet_loss.png')

    plt.show()


if __name__ == '__main__':
    main()
