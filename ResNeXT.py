import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

import pickle
import numpy as np

keras = tf.keras


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

    base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(50, 50, 1))
    base_model.trainable = False

    keras.applications.

    # Create the model
    model = Sequential()

    # Add the vgg convolutional base model
    model.add(base_model)

    # Add new layers
    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()

    base_model.trainable = True

    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine tune from this layer onwards
    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    import time
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=32, epochs=15, validation_split=0.35, callbacks=[tensorboard])

    model.save('network.model')
    print("--- %s seconds ---" % (time.time() - start_time))

    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=32, callbacks=[tensorboard])
    print('test loss, test acc:', results)


create_model()
