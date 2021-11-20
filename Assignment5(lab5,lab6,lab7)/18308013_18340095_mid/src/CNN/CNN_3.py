# 使用dropout的模型
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


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


def build_model():

    # 实例化一个简单的CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # 在CNN上添加分类器
    model.add(layers.Flatten())  # 摊平成1D向量输入到密集连接分类器中
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5));
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5));
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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

    model = build_model()
    history = model.fit(train_images, train_labels, epochs=50, batch_size=64, validation_data=(val_images, val_labels))

    model.save('cifar10_1.h5')  # _1为最原始的CNN

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
    plt.savefig('1_acc.png')

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('1_loss.png')

    plt.show()


if __name__ == '__main__':
    main()
