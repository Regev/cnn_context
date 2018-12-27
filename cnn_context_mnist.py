#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_context_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Flat one_hot_tensor of context class
    # Input Context_index
    # Reshape context_index to one hot tensor: [batch_size, number_of_context_labels]
    # context label is MNIST images or MNIST-Fashion images
    number_of_context_labels = 2
    context_one_hot = tf.reshape(tf.one_hot(features["context_index"], depth=number_of_context_labels),
                                 [-1, number_of_context_labels])

    # Concatenates pool2_flat and  context_one_hot tensors along batch_size dimension
    # Input Tensor Shape: [batch_size, 7, 7, 64] , [batch_size, number_of_context_labels]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64 + number_of_context_labels]
    pool2_context_flat = tf.concat([pool2_flat, context_one_hot], 1)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_context_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 20]
    logits = tf.layers.dense(inputs=dropout, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_vgg16_context_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #1_1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1_1 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2_1
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2_1 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Flat one_hot_tensor of context class
    # Input Context_index
    # Reshape context_index to one hot tensor: [batch_size, number_of_context_labels]
    # context label is MNIST images or MNIST-Fashion images
    number_of_context_labels = 2
    context_one_hot = tf.reshape(tf.one_hot(features["context_index"], depth=number_of_context_labels),
                                 [-1, number_of_context_labels])

    # Concatenates pool2_flat and  context_one_hot tensors along batch_size dimension
    # Input Tensor Shape: [batch_size, 7, 7, 64] , [batch_size, number_of_context_labels]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64 + number_of_context_labels]
    pool2_context_flat = tf.concat([pool2_flat, context_one_hot], 1)

    # Dense Layer #1
    # Densely connected layer with 512 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 512]
    dense1 = tf.layers.dense(inputs=pool2_context_flat, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer #2
    # Densely connected layer with 512 neurons
    # Input Tensor Shape: [batch_size, 512]
    # Output Tensor Shape: [batch_size, 512]
    dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 512]
    # Output Tensor Shape: [batch_size, 20]
    logits = tf.layers.dense(inputs=dropout2, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn():
    inputs = {
        "x": tf.placeholder(tf.float32, [None, 28, 28, 1]),
        "context_index": tf.placeholder(tf.int32, [None, 1]),
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main(unused_argv):
    estimator_go(
        model_dir=r"/tmp/mnist_convnet_vgg_16_context_model",
        model_fn=cnn_vgg16_context_model_fn,
        steps=100000)

    # estimator_go(
    #     model_dir=r"/tmp/mnist_convnet_context_model",
    #     model_fn=cnn_context_model_fn,
    #     steps=100)


def estimator_go(model_dir, model_fn, batch_size=1000, num_epochs=None, steps=100):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_mnist_data = mnist.train.images  # Returns np.array
    train_mnist_data_context = np.zeros((len(train_mnist_data), 1), dtype=np.int32)
    train_mnist_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_mnist_data = mnist.test.images.astype('float64')  # Returns np.array
    eval_mnist_data_context = np.zeros((len(eval_mnist_data), 1), dtype=np.int32)
    eval_mnist_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_fmnist_data, train_fmnist_labels), (eval_fmnist_data, eval_fmnist_labels) = fashion_mnist.load_data()
    train_fmnist_data = train_fmnist_data / 255.0
    train_fmnist_data = train_fmnist_data.reshape(len(train_fmnist_data), 28*28)
    train_fmnist_labels = train_fmnist_labels + 10
    eval_fmnist_data = eval_fmnist_data / 255.0
    eval_fmnist_data = eval_fmnist_data.reshape(len(eval_fmnist_data), 28 * 28)
    eval_fmnist_labels = eval_fmnist_labels + 10
    train_fmnist_data_context = np.ones((len(train_fmnist_data), 1), dtype=np.int32)
    eval_fmnist_data_context = np.ones((len(eval_fmnist_data), 1), dtype=np.int32)

    train_data = np.concatenate([train_mnist_data, train_fmnist_data]).astype('float32')
    train_data_context = np.concatenate([train_mnist_data_context, train_fmnist_data_context]).astype('int32')
    train_labels = np.concatenate([train_mnist_labels, train_fmnist_labels]).astype('int32')

    eval_data = np.concatenate([eval_mnist_data, eval_fmnist_data]).astype('float32')
    eval_data_context = np.concatenate([eval_mnist_data_context, eval_fmnist_data_context]).astype('int32')
    eval_labels = np.concatenate([eval_mnist_labels, eval_fmnist_labels]).astype('int32')

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                              model_dir=model_dir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data,
           "context_index": train_data_context},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=steps,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data,
           "context_index": eval_data_context},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    full_model_dir = mnist_classifier.export_savedmodel(export_dir_base=model_dir,
                                                        serving_input_receiver_fn=serving_input_receiver_fn)
    print(full_model_dir)


if __name__ == "__main__":
    tf.app.run()
