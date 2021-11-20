from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
import os

MAX_LEN = 100
max_words = 20000
Embedding_DIM = 100


# F1-score
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def my_load_data():
    punctuation_string = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'  # 符号字符串
    path = 'archive/Laptop_Train_v2.xml'  # xml文件路径
    tree = ET.parse(path)
    root = tree.getroot()

    texts = []  # 文本列表
    labels = []  # 文本的单词标签列表
    max_len = 0

    for sentence in root.findall('sentence'):
        text = sentence.find('text').text             # 获取每个文本
        for i in punctuation_string:
            text = text.replace(i, ' ')               # 去除文本的标点符号
        texts.append(text)
        text_words = text.split()
        label = [0] * len(text_words)                 # 该文本的单词原始标签列表
        max_len = max(max_len, len(text_words))
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms == None:
            labels.append(label)                      # 没有aspectTerms，单词标签全为0
        else:
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                from_index = int(aspectTerm.get('from'))         # aspectTerm字符起始索引
                to_index = int(aspectTerm.get('to'))             # aspectTerm字符结束索引
                index = 0
                for i in range(len(text_words)):
                    index = text.find(text_words[i], index, -1)
                    if index >= from_index and index < to_index:  # 若该单词首字符的索引在term中
                        label[i] = 1
            labels.append(label)

    print(max_len)
    # 分词
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 转换成编码list
    word_index = tokenizer.word_index                # 单词编码
    print('Found %s unique tokens.' % len(word_index))
    train_data = pad_sequences(sequences, maxlen=MAX_LEN)
    train_labels = pad_sequences(labels, maxlen=MAX_LEN)
    train_labels = np.array(train_labels)

    # 加载词嵌入
    glove_dir = 'glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return train_data, train_labels, embedding_matrix


def main():
    train_data, train_labels, embedding_matrix = my_load_data()

    # 构建模型
    model = models.Sequential()
    model.add(layers.Embedding(max_words, Embedding_DIM, input_length=MAX_LEN))
    model.add(layers.Bidirectional(layers.SimpleRNN(32, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = True
    model.compile('adam', loss='binary_crossentropy', metrics=['acc', f1])
    model.summary()
    history = model.fit(train_data, train_labels, batch_size=32, epochs=50, validation_split=0.1, verbose=1)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_f1 = history.history['f1']
    val_f1 = history.history['val_f1']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.legend(loc='best')
    plt.savefig('BRNN_acc.png')

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('BRNN_loss.png')

    plt.figure()
    plt.plot(epochs, train_f1, 'bo', label='Training F1-score')
    plt.plot(epochs, val_f1, 'b', label='Validation F1-score')
    plt.title('Training and validation F1-score')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('BRNN_f1.png')
    plt.show()


if __name__ == '__main__':
    main()