#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''
import tensorflow.compat.v1 as tf
import numpy as np
import sys


tf.disable_v2_behavior()


'''
Environment settings:
Set MKLDNN_VERBOSE=1 to show DNNL run time verbose
Set KMP_AFFINITY=verbose to show OpenMP thread information
'''
import os; os.environ["MKLDNN_VERBOSE"] = "1"
import os; os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
'''
Sanity Check: once Intel-optimized TensorFlow is installed, Intel DNNL optimizations are present by default.
'''
#TODO for TF2.0
#print("Intel DNNL optimizations are present : ", tf.pywrap_tensorflow.IsMklEnabled())

'''
learning_rate = 0.1
batch_size : Batch size
BS_TRAIN: Number of Batch size for training data
EPOCHNUM: Number of epoch for training
'''
learning_rate = 0.1
batch_size = 4
BS_TRAIN = 10
EPOCHNUM = 5

'''
Perform training and inference in main function
'''
def main():
    '''Define input/output data size'''
    N, C_in, C_out, H, W = batch_size, 4, 10, 128, 128

    ''' Create random array to hold inputs and outputs '''
    x_data = np.float32(np.random.random([N*BS_TRAIN, H, W, C_in]))
    y_data = np.float32(np.random.random([N*BS_TRAIN, H, W, C_out]))

    '''
    Create l layer conv2d + relu neural network.
    given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], output is [batch, in_height/strides, in_width/striders, out_channels]
    '''
    filter = tf.Variable(tf.random.uniform([3,3,C_in, C_out],-1.0,1.0))
    bias = tf.Variable(tf.constant(0.1, shape=[C_out]))

    x = tf.placeholder("float", [N, H, W, C_in])
    y = tf.placeholder("float", [N, H, W, C_out])

    y_con2d = tf.nn.conv2d(x, filter, strides=[1,1,1,1], data_format='NHWC', padding='SAME')
    y_pred = tf.nn.relu(y_con2d + bias)

    '''Define the loss function and optimizser'''
    loss = tf.reduce_mean(tf.square(y-y_pred))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    '''Initialize the session and set intra/inter op threads'''
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 8 # Set to number of physical cores
    config.inter_op_parallelism_threads = 1 # Set to number of sockets
    tf.Session(config=config)
    s = tf.Session()
    s.run(init)

    '''start train step'''
    for epoch in range(0, EPOCHNUM):
        for step in range(0, BS_TRAIN):
            x_batch = x_data[step*N:(step+1)*N, :, :, :]
            y_batch = y_data[step*N:(step+1)*N, :, :, :]
            s.run(train, feed_dict={x: x_batch, y: y_batch})
        '''Compute and print loss. We pass Tensors containing the predicted and true values of y, and the loss function returns a Tensor containing the loss.'''
        print(epoch, s.run(loss,feed_dict={x: x_batch, y: y_batch}))

if __name__ == '__main__':
    main()
    print("Tensorflow HelloWorld Done!")
    print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")
