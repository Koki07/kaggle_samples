import h5py
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils


def format_data(data):
    data = data.reshape(data.shape[0], 784)
    data = data.astype('float32')
    data /= 255
    return data

def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    # An "activation" is just a non-liner function applied to the output
    # of the layer above. Here, with a "rectified linear unit",
    # we clamp all values below 0 to 0.
    model.add(Dropout(0.2))
    # Dropout helps protect the model from memorizing or "overfitting" the training data.
    model.add(Activation('relu')) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    # This special "softmax" Activation among other things,
    # ensures the output is a valid probaility distribution, that is
    # that its values are all non-negative and sum to 1.
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def main():
    # nb_classes = 10
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = format_data(x_train)
    # x_test = format_data(x_test)

    # y_train = np_utils.to_categorical(y_train, nb_classes)
    # y_test = np_utils.to_categorical(y_test, nb_classes)

    # model = build_model()
    # model.fit(x_train, y_train, batch_size=128, epochs=4, validation_data=(x_test, y_test))
    # model.save_weights('mnist_weights.h5')
    # 
    # test_data = pd.read_csv('./test.csv').as_matrix().astype('float32')
    # test_data /= 255
    # pred = model.predict(test_data)
    # pred = np.argmax(pred, axis=1)
    # np.save('pred.npy', pred)
    pred = np.load('pred.npy')
    result = list(range(1, len(pred)+1))
    result = {'ImageId': result, 'Label': pred.tolist()}
    result = pd.DataFrame(result)
    result.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    main()
