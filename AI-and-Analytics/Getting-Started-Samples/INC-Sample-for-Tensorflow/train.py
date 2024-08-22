#Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import alexnet
import mnist_dataset


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
#Training the model for alexnet dataset
model.fit(x_train, y_train, epochs=epochs, batch_size=600, validation_data=(x_test, y_test), verbose=1)
score = model.evaluate(x_train, y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#Training the model for mnist_dataset
model.fit(_x_train, _y_train, epochs=epochs, batch_size=600, validation_data=(_x_test, _y_test), verbose=1)
score = model.evaluate(_x_train, _y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])