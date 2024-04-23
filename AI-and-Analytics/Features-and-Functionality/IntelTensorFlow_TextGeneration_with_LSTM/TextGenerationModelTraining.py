#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================
# Copyright © 2023 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # Leveraging Intel Extension for TensorFlow with LSTM for Text Generation
# 
# The sample will present the way to train the model for text generation with LSTM (Long short-term Memory) using the Intel extension for TensorFlow. It will focus on the parts that are relevant for faster execution on Intel hardware enabling transition of existing model training notebooks to use Intel extension for TensorFlow (later in the text Itex).
# 
# In order to have text generated, one needs a deep learning model. The goal of text generation model is to predict the probability distribution of the next word in a sequence given the previous words. For that, large amount of text is feed to the model training.

# ## Preparing the data
# 
# Training data can be text input (in form of a book, article, etc). For this sample we will [The Republic by Plato](https://www.gutenberg.org/cache/epub/1497/pg1497.txt). 
# 
# Once downloaded, the data should be pre-processed: remove punctuation, remove special characters and set all letter to lowercase.

# In[ ]:


import string
import requests

response = requests.get('https://www.gutenberg.org/cache/epub/1497/pg1497.txt')
data = response.text.split('\n')
data = " ".join(data)


# In[ ]:


def clean_text(doc):
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [(w.translate(table)) for w in tokens] # list without punctuations
    tokens = [word for word in tokens if word.isalpha()] # remove  alphanumeric special characters
    tokens = [word.lower() for word in tokens]
    return tokens


# In[ ]:


tokens = clean_text(data)


# Depending on the data, training data width (the number of words/tokens) should be updated. For instance: for longer texts (e.g. novels, lecture books, etc) that need a context the width could be 40 or 50.
# This means we would provide the input of training data width to get our model to generate next word.
# 
# According to the width (the number of words/tokens), training data needs to be updated.

# In[ ]:


def get_aligned_training_data(text_tokens, train_data_width):
    text_tokens[:train_data_width]
    
    length = train_data_width + 1
    lines = []
    
    for i in range(length, len(text_tokens)): 
        seq = text_tokens[i - length:i]
        line = ' '.join(seq)
        lines.append(line)
    return lines

lines = get_aligned_training_data(tokens, 50)
len(lines)


# ## Checking available devices
# 
# Since we want to leverage Intel's GPU for model training, here are simple instructions on how to check if the environment is setup.
# 
# In order to see which devices are available for TensorFlow to run its training on, run the next cell.
# 
# NOTE: GPU will be displayed as `XPU` The line should look like:
# ```
# PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')
# ```

# In[ ]:


import tensorflow as tf

xpus = tf.config.list_physical_devices()
xpus


# When it comes to the TensorFlow execution, by default, it is using eager mode. In case user wants to run graph mode, it can be done by adding following line:
# ```
# tf.compat.v1.disable_eager_execution()
# ```

# In[ ]:


tf.compat.v1.disable_eager_execution()


# ## Preparing and training the model
# 
# As a final step for training the model, the data needs to be tokenized (every word gets the index assigned) and converted to sequences.

# In[ ]:


# Tokenization
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
# Keras layers
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Sequential


# In[ ]:


def tokenize_prepare_dataset(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    
    # Get vocabulary size of our model
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(lines)
    
    # Convert to numpy matrix
    sequences = np.array(sequences)
    x, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    return x, y, tokenizer

x, y, itex_tokenizer = tokenize_prepare_dataset(lines)
seq_length = x.shape[1]
vocab_size = y.shape[1]
vocab_size


# ## LSTM Operator Override Intel Extension for TensorFlow 
# 
# Besides leveraging AI and oneAPI Kit for GPU execution, Intel extension for TensorFlow (Itex) offers operator overrides for some of the Keras layers. In this sample LSTM for text generation model will be used. A LSTM (Long Short-term Memory first proposed in Hochreiter & Schmidhuber, 1997) Neural Network is just another kind of Artificial Neural Network, containing LSTM cells as neurons in some of its layers. Every LSTM layer will contain many LSTM cells. Every LSTM cell looks at its own input column, plus the previous column's cell output. This is how an LSTM cell logic works on determining what qualifies to extract from the previous cell:
#  - Forget gate - determines influence from the previous cell (and its timestamp) on the current one. It uses the 'sigmoid' layer (the result is between 0.0 and 1.0) to decide whether should it be forgotten or remembered
#  - Input gate - the cell tries to learn from the input to this cell. It does that with a dot product of sigmoid (from Forget Gate) and tanh unit. 
#  - Output gate - another dot product of the tanh and sigmoid unit. Updated information is passed from the current to the next timestamp.
# 
# Passing a cell's state/information at timestamps is related to long-term memory and hidden state - to short-term memory.
# 
# Instead of using LSTM layer from Keras, Itex LSTM layer is better optimized for execution on Intel platform. LSTM layer, provided by Itex is semantically the same as [tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM). Based on available runtime hardware and constraints, this layer will choose different implementations (ITEX-based or fallback-TensorFlow) to maximize the performance.
# 
# After creating the model and adding Embedding layer, optimized LSTM layer from Itex can be added. Note, however, that this is sample for mentioned input text. Parameters are open to experiment, depending on the input text. Model accuracy with existing parameters is reaching 80%.

# In[ ]:


import intel_extension_for_tensorflow as itex


# In[ ]:


neuron_coef = 4
itex_lstm_model = Sequential()
itex_lstm_model.add(Embedding(input_dim=vocab_size, output_dim=seq_length, input_length=seq_length))
itex_lstm_model.add(itex.ops.ItexLSTM(seq_length * neuron_coef, return_sequences=True))
itex_lstm_model.add(itex.ops.ItexLSTM(seq_length * neuron_coef))
itex_lstm_model.add(Dense(units=seq_length * neuron_coef, activation='relu'))
itex_lstm_model.add(Dense(units=vocab_size, activation='softmax'))
itex_lstm_model.summary()
itex_lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
itex_lstm_model.fit(x,y, batch_size=256, epochs=200)


# ## Compared to LSTM from Keras
# 
# The training done with Itex LSTM has efficient memory management on Intel GPU. As a reference, on the system with Intel® Arc™ 770, GPU memory was constant at around 4.5GB.
# 
# Below is the example cell with the same dataset using LSTM layer from keras. To run on the same system, parameters such as sequence length, number of epochs, and other training layer parameters had to be lowered.
# 
# Compared to parameters that were used by training with Itex LSTM (3192280 total parameters), with keras LSTM only 221870 total parameters were used. Besides accelerating the model training, Itex LSTM offers better memory management in Intel platform.

# In[ ]:


from tensorflow.keras.layers import LSTM

# Reducing the sequence to 10 compared to 50 with Itex LSTM
lines = get_aligned_training_data(tokens, 10)

# Tokenization
x, y, keras_tokenizer = tokenize_prepare_dataset(lines)
seq_length = x.shape[1]
vocab_size = y.shape[1]

neuron_coef = 1
keras_lstm_model = Sequential()
keras_lstm_model.add(Embedding(input_dim=vocab_size, output_dim=seq_length, input_length=seq_length))
keras_lstm_model.add(LSTM(seq_length * neuron_coef, return_sequences=True))
keras_lstm_model.add(LSTM(seq_length * neuron_coef))
keras_lstm_model.add(Dense(units=seq_length * neuron_coef, activation='relu'))
keras_lstm_model.add(Dense(units=vocab_size, activation='softmax'))
keras_lstm_model.summary()
keras_lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
keras_lstm_model.fit(x,y, batch_size=256, epochs=20)


# ## Generating text based on the input
# 
# Now that that the model has been trained, it is time to use it for generating text based on given input (seed text).
# One can input its own line, but for better result it is best to take the input line from text.
# 
# A method for generating text has been created, which will take following parameters:
#  - trained model;
#  - tokenizer;
#  - data width that we used;
#  - input text - seed text;
#  - number of words to generate and append to the seed text.
# 
# For testing, a random line from input text will be taken as a seed text.

# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
def generate_text_seq(model, tokenizer, text_seq_length, seed_text, generated_words_count):
    text = []
    input_text = seed_text
    for _ in range(generated_words_count):
        encoded = tokenizer.texts_to_sequences([input_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating = 'pre')
        predict_x=model.predict(encoded)
        y_predict=np.argmax(predict_x, axis=1)
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_predict:
                predicted_word = word
                break
        input_text += ' ' + predicted_word
        text.append(predicted_word)
    return ' '.join(text)


# In[ ]:


import random
random.seed(101)
random_index = random.randint(0, len(lines))
random_seed_text = lines[random_index]
random_seed_text


# In[ ]:


number_of_words_to_generate = 10
generated_text = generate_text_seq(itex_lstm_model, itex_tokenizer, 50, random_seed_text, number_of_words_to_generate)
print("::: SEED TEXT::: " + random_seed_text)
print("::: GENERATED TEXT::: " + generated_text)


# # Summary
# 
# This was the sample with the basic concept of how to train the model for text generation. The main focus was on leveraging Intel's libraries and platform to address one of the challenges with LSTM and that it is a more complex architecture than simple RNN (Recurrent Neural Network). LSTM takes more memory and time to train due to additional parameters and operations. On the other hand, LSTM has the ability to learn long-term dependencies and capture complex patterns in sequential data.

# In[ ]:


print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")

