#Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))
import alexnet
import mnist_dataset
import sys
import os
import subprocess
import json
import time
import argparse
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


#Splitting the dataset for alexnet dataset
x_train, y_train, label_train, x_test, y_test, label_test = alexnet.read_data()
print('train', x_train.shape, y_train.shape, label_train.shape)
print('test', x_test.shape, y_test.shape, label_test.shape)

#Splitting the dataset for mnist_dataset
_x_train, _y_train, _label_train, _x_test, _y_test, _label_test = mnist_dataset.read_data()
print('train',_x_train.shape,_y_train.shape,_label_train.shape)
print('test',_x_test.shape,_y_test.shape,_label_test.shape)

#Building the model
classes = 10
width = 28
channels = 1


model = alexnet.create_model(width ,channels ,classes)
model.summary()

epochs = 3
mod_path = "./path/to/save/model"

#Testing the model for alexnet dataset
train1 = model.fit(x_train, y_train, epochs=epochs, batch_size=600, validation_data=(x_test, y_test), verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
alexnet.save_mod(model,mod_path)

# Function to save the frozen model for both dataset

def save_frozen_model(model, mod_path):
    # Get the concrete function from the Keras model
    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Convert the model to a frozen graph
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    tf.io.write_graph(graph_or_graph_def=frozen_model.graph,
                      logdir=".",
                      name=mod_path,
                      as_text=False)

int8_pb_file = "alexnet_int8_model.pb"
save_frozen_model(model, int8_pb_file)

#Executing INT8 model with PB model
command = [
    "python3.10", "profiling_inc.py",
    "--input-graph=./alexnet_int8_model.pb",
    "--omp-num-threads=4",
    "--num-inter-threads=1",
    "--num-intra-threads=4",
    "--index=8"
]

result = subprocess.run(command, capture_output=True, text=True)

print("Return code:", result.returncode)
print("Output:", result.stdout)
print("Error:", result.stderr)

#Testing the model for mnist_dataset
                                                           
train2 = model.fit(_x_train, _y_train, epochs=epochs, batch_size=600, validation_data=(_x_test, _y_test), verbose=1)
score = model.evaluate(_x_test, _y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

fp32_frozen_pb_file = "fp32_frozen.pb"
save_frozen_model(model, fp32_frozen_pb_file)

#Executing the FP32 model with PB model
command = [
    "python3.10", "profiling_inc.py",
    "--input-graph=./fp32_frozen.pb",
    "--omp-num-threads=4",
    "--num-inter-threads=1",
    "--num-intra-threads=4",
    "--index=32"
]

result = subprocess.run(command, capture_output=True, text=True)

print("Return code:", result.returncode)
print("Output:", result.stdout)
print("Error:", result.stderr)

display.Code('32.json')
command = ["echo", " "]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output
print("Output:", result.stdout)
display.Code('8.json')


#Inference

def load_res(json_file):
    with open(json_file) as f:
        data = json.load(f)
        return data

res_32 = load_res('32.json')
res_8 = load_res('8.json')

accuracys = [res_32['accuracy'], res_8['accuracy']]
throughputs = [res_32['throughput'], res_8['throughput']]
latencys = [res_32['latency'], res_8['latency']]

print('throughputs', throughputs)
print('latencys', latencys)
print('accuracys', accuracys)

accuracys_perc = [accu*100 for accu in accuracys]

throughputs_times = [1, throughputs[1]/throughputs[0]]
latencys_times = [1, latencys[1]/latencys[0]]
accuracys_times = [0, accuracys_perc[1] - accuracys_perc[0]]

print('throughputs_times', throughputs_times)
print('latencys_times', latencys_times)
print('accuracys_times', accuracys_times)