#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================

==============================================================
Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================
'''

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import time
import os
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

from tensorflow import keras

tf.compat.v1.disable_eager_execution()
'''
Environment settings:
Set MKLDNN_VERBOSE=1 to show DNNL run time verbose
Set KMP_AFFINITY=verbose to show OpenMP thread information
'''
#import os; os.environ["MKLDNN_VERBOSE"] = "1"
import os; os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

def cnn_model_fn(feature, target, mode):

  """Model function for CNN."""
  """2-layer convolution model."""

  # Convert the target to a one-hot tensor of shape (batch_size, 10) and
  # with a on-value of 1 for each one-hot vector of length 10.
  target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)


  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  feature = tf.reshape(feature, [-1, 28, 28, 1])

  # First Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.compat.v1.layers.conv2d(
      inputs=feature,
      filters=32,
      kernel_size=[5, 5],
      padding="SAME",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.nn.max_pool2d(input=conv1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding="SAME")

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.compat.v1.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="SAME",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.nn.max_pool2d(input=conv2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding="SAME")

  # Flatten tensor into a batch of vectors
  # Reshape tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.compat.v1.layers.dense(inputs=pool2_flat,
                          units=1024,
                          activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.compat.v1.layers.dropout(inputs=dense,
                              rate=0.6,
                              training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.compat.v1.layers.dense(dropout, 10, activation=None)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.compat.v1.losses.softmax_cross_entropy(target, logits)

  return tf.argmax(input=logits, axis=1), loss

def train_input_generator(x_train, y_train, batch_size=64):
    if len(x_train) == len(y_train):
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size
    else:
        None

def main(unused_argv):
  # Initialize Horovod
  hvd.init()
    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
  cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets') 
  if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise
                
  # Load training and eval data
  # Download and load MNIST dataset.
  (x_train, y_train), (x_test, y_test) = \
        keras.datasets.mnist.load_data('MNIST-data-%d' % hvd.rank())
   # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
  x_train = np.reshape(x_train, (-1, 784)) / 255.0
  x_test = np.reshape(x_test, (-1, 784)) / 255.0
  # Build model...
  with tf.compat.v1.name_scope('input'):
        image = tf.compat.v1.placeholder(tf.float32, [None, 784], name='image')
        label = tf.compat.v1.placeholder(tf.float32, [None], name='label') 
  predict, loss = cnn_model_fn(image, label, tf.estimator.ModeKeys.TRAIN)

  # Horovod: adjust learning rate based on number of MPI Tasks.
  opt = tf.compat.v1.train.RMSPropOptimizer(0.001 * hvd.size())
  opt = hvd.DistributedOptimizer(opt)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  train_op = opt.minimize(loss, global_step=global_step)  

  hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of MPI tasks.
        tf.estimator.StopAtStepHook(last_step=1000 // hvd.size()),

        tf.estimator.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=100),
    ]
 # Horovod: save checkpoints only on worker 0 to prevent other workers from 
 # corrupting them.
  checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
  training_batch_generator = train_input_generator(x_train,
                                                     y_train, batch_size=100)
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    
  config = tf.compat.v1.ConfigProto()
#  config.inter_op_parallelism_threads = 2
#  config.intra_op_parallelism_threads = 4
  

  time_start = time.time()
  
  with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = next(training_batch_generator)
            mon_sess.run(train_op, feed_dict={image: image_, label: label_}) 
  
  if hvd.rank() == 0:
        print('============================')
        print('Number of tasks: ', hvd.size())
        print('Total time is: %g' % (time.time() - time_start))


if __name__ == '__main__':
    tf.compat.v1.app.run()
