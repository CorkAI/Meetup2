# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

Altered for Cork_AI meetup to illustrate keras

See original documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc

import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

FLAGS = None


# build a model using keras
def deepnn_keras(model_input_shape):
    my_model = Sequential()
    my_model.add(Conv2D(input_shape=model_input_shape, kernel_size=(5, 5), filters=32, padding="same", activation="relu"))
    my_model.add(MaxPooling2D())
    my_model.add(Conv2D(kernel_size=(5, 5), filters=64, padding="same", activation="relu"))
    my_model.add(Flatten())
    my_model.add(Dense(units=1024, activation="relu"))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(10, activation='softmax'))
    return my_model


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    my_model = Sequential()

    # If pre-trained model exists on disk, then just load that
    if os.path.isfile(os.path.join(os.getcwd(), 'saved_model/cork_ai_model_keras_deep.h5')):
        my_model = load_model("saved_model/cork_ai_model_keras_deep.h5")
        print("Model restored from disk")

    # Build and train a model using keras
    else:
        my_model = deepnn_keras(input_shape=(28, 28, 1))
        my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_images = np.reshape(mnist.train.images, [-1, 28, 28, 1])
        print("train set shape is ", train_images.shape)
        print("train labels shape is ", mnist.train.labels.shape)
        my_model.fit(train_images, mnist.train.labels, epochs=18, batch_size=50)
        my_model.save("saved_model/cork_ai_model_keras_deep.h5")

    test_images = np.reshape(mnist.test.images, [-1, 28, 28, 1])
    accuracy_test = my_model.evaluate(test_images, mnist.test.labels, batch_size=50)
    print('\n\ntest accuracy is ', accuracy_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    parser.add_argument('--write_samples', type=int, default=0)
    parser.add_argument('--extra_test_imgs', type=int, default=0)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
