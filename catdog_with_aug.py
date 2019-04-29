from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import numpy.random as random
import os
import cv2


TRAIN_DIR = "./train/"
TEST_DIR = "./test/"

train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
random.shuffle(train_images)

CHANNELS = 3
ROWS = 64
COLS = 64


def network(input_shape=(3, ROWS, COLS), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3,  input_shape=input_shape,
                     padding="same", activation="relu"))
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           dim_ordering="th", data_format="channels_first"))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           dim_ordering="th", data_format="channels_first"))
    model.add(Conv2D(128, kernel_size=3,
                     padding="same", activation="relu"))
    model.add(Conv2D(128, kernel_size=3,
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           dim_ordering="th", data_format="channels_first"))
    model.add(Conv2D(256, kernel_size=3,
                     padding="same", activation="relu"))
    model.add(Conv2D(256, kernel_size=3,
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           dim_ordering="th", data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("sigmoid"))
    return model


class CatDogDataset():
    def __init__(self):
        self.image_shape = (64, 64, 3)
        self.num_classes = 2

    def read_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    def prep_data(self, train_images=train_images, test_images=test_images):
        train_count = len(train_images)
        test_count = len(test_images)
        train_data = np.ndarray(
            (train_count, CHANNELS, ROWS, COLS), dtype=np.uint8)
        test_data = np.ndarray(
            (test_count, CHANNELS, ROWS, COLS), dtype=np.uint8)

        y_train = []
        for i, train_image_file in enumerate(train_images):
            if "dog" in train_image_file:
                y_train.append(1)
            else:
                y_train.append(0)

            train_image = self.read_image(train_image_file)
            train_data[i] = train_image.T
            x_train = train_data[:]

        for i, test_image_file in enumerate(test_images):
            test_image = self.read_image(test_image_file)
            test_data[i] = test_image.T
            x_test = test_data[:]

        return x_train, y_train, x_test

    def get_batch(self):
        x_train, y_train, x_test = self.prep_data(train_images, test_images)

        X_train = [self.preprocess(d) for d in [x_train]]
        Y_train = [self.preprocess(d, label_data=True) for d in [y_train]]
        X_test = [self.preprocess(d) for d in [x_test]]

        return X_train, Y_train, X_test

    def preprocess(self, data, label_data=False):
        if label_data:
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)

        return data


class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
        )
        self.verbose = 1
        logdir = "logdir_catdog"
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
        self.model_file_name = "catdog_model_file.hdf5"

    def train(self, X_train, Y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        model_path = os.path.join(self.log_dir, self.model_file_name)
        self._target.fit(
            X_train, Y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True)
            ],
            verbose=self.verbose
        )


dataset = CatDogDataset()

model = network(dataset.image_shape, dataset.num_classes)

X_train, Y_train, X_test = dataset.get_batch()
trainer = Trainer(model, loss="binary_crossentropy", optimizer=RMSprop())
trainer.train(X_train, Y_train, batch_size=64,
              epochs=3, validation_split=0.2)
prediction = model.predict(X_test)
print(prediction)
