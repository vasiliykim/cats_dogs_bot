import os
import cv2
import random

import numpy as np
import pickle

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

IMAGE_DATASET_DIR_PATH = "C:\\dataset\\PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_WIDTH = 50
IMG_HEIGHT = 50
INPUT_IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
TEST_DATASET_SIZE = 500
training_dataset = []
test_dataset = []


def initialize_data():
    for category in CATEGORIES:
        path = os.path.join(IMAGE_DATASET_DIR_PATH, category)  # path to dogs and cats images
        class_num = CATEGORIES.index(category)
        image_count = 0
        for img in os.listdir(path):
            image_count = image_count + 1
            if image_count > TEST_DATASET_SIZE:
                try:
                    loaded_img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_img = cv2.resize(loaded_img, INPUT_IMG_SHAPE)
                    training_dataset.append([resized_img, class_num])
                except Exception as e:
                    pass
            else:
                try:
                    loaded_img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_img = cv2.resize(loaded_img, INPUT_IMG_SHAPE)
                    test_dataset.append([resized_img, class_num])
                except Exception as e:
                    pass

    random.shuffle(training_dataset)
    random.shuffle(test_dataset)
    print("training_dataset_size = ", len(training_dataset))
    print("test_dataset_size = ", len(test_dataset))


initialize_data()


def create_pickle_data():
    random.shuffle(training_dataset)

    x_train = []
    y_train = []
    for features, label in training_dataset:
        x_train.append(features)
        y_train.append(label)
    x_train = np.array(x_train).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    pickle_out = open("x_train.pickle", "wb")
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    x_test = []
    y_test = []

    for features, label in test_dataset:
        x_test.append(features)
        y_test.append(label)

    x_test = np.array(x_test).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    pickle_out = open("x_test.pickle", "wb")
    pickle.dump(x_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()


create_pickle_data()


def create_model():
    import time
    name = "CATS_VS_DOGS_CNN-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir='logs\\{}'.format(name))

    x_train = pickle.load(open("x_train.pickle", "rb"))
    y_train = pickle.load(open("y_train.pickle", "rb"))
    x_test = pickle.load(open("x_test.pickle", "rb"))
    y_test = pickle.load(open("y_test.pickle", "rb"))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=x_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    import time
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=32, epochs=15, validation_split=0.35, callbacks=[tensorboard])

    model.save('network.model')
    print("--- %s seconds ---" % (time.time() - start_time))

    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=32, callbacks=[tensorboard])
    print('test loss, test acc:', results)


create_model()
