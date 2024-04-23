# details about this Python script 
# - based on a Kaggle solution - https://www.kaggle.com/code/xhlulu/disaster-nlp-distilbert-in-tf/notebook for disaster tweet classification
# - uses the pretrained huggingface distilbert model and fine tunes it based on test dataset
# - uses keras mixed precision API to run the BF16 version of the model 

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='train and infer DistilBERT model for disaster tweet classification (based on Kaggle competition)')
parser.add_argument("--tune_model", "-t", help="fine tune pre-trained distilbert model from hugging face", type=bool, default=True)
parser.add_argument("--precision", "-p", help="datatype precision used by model - fp32 or bf16", type=str, default='fp32')
parser.add_argument("--log_dir", "-ld", help="directory to store profile info in", type=str, default="logs")
parser.add_argument("--profile", "-pr", help="dump tensorboard profiling info", type=bool, default=False)
parser.add_argument("--execution_mode", "-m", help="execution mode - eager or graph", type=str, default="graph")
parser.add_argument("--load_weights_dir", "-lwd", help="directory to load weights from", type=str, default="weights")
parser.add_argument("--save_weights_dir", "-swd", help="directory to save weights in", type=str, default="weights")

args = parser.parse_args()
precision = args.precision
is_tune_model = args.tune_model
log_dir = args.log_dir
profiling_needed = args.profile
execution_mode = args.execution_mode
load_weights_dir = args.load_weights_dir
save_weights_dir = args.save_weights_dir

# if precision == 'bf16':
#   tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

from cpuinfo import get_cpu_info
info = get_cpu_info()
flags = info['flags']
amx_supported = False
for flag in flags:
    if "amx" in flag:
        amx_supported = True
        print("AMX is supported on current hardware. Code sample can be run.\n")
if not amx_supported:
    print("AMX is not supported on current hardware. Code sample cannot be run.\n")
    sys.exit("AMX is not supported on current hardware. Code sample cannot be run.")
    
if execution_mode == "graph":
  tf.compat.v1.disable_eager_execution()



# helper functions
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
    
    return np.array(all_tokens)
    
def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
# load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
classified_results = pd.read_csv("data/sample_submission.csv")

# load distilbert uncased pre-trained model and corresponding tokenizer from hugging face
# transformer_layer = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# build model
model = build_model(transformer_layer, max_len=160)

# fine tune model according to disaster tweets dataset
if is_tune_model:
    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    train_labels = train.target.values
    start_time = time.time()
    train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=4,
    batch_size=64
  )
    end_time = time.time()
  # save model weights so we don't have to fine tune it every time
    os.makedirs(save_weights_dir, exist_ok=True)
    model.save_weights(save_weights_dir + "/model_weights.h5")

# if model is alredy fine tuned, then load weights 
else:
    try:
        model.load_weights(load_weights_dir + "/model_weights.h5")
    except FileNotFoundError:
        sys.exit("\n\nTuned model weights not available. Tune model first by setting parameter -t=True")

fp32_training_time = end_time-start_time
print("Training model with FP32")

#########################################################################################

# BF16 without AMX
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})

transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = build_model(transformer_layer, max_len=160)

# fine tune model according to disaster tweets dataset
if is_tune_model:
    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    train_labels = train.target.values
    start_time = time.time()
    train_history = model.fit(train_input, train_labels, validation_split=0.2, epochs=4, batch_size=64)
    end_time = time.time()
  # save model weights so we don't have to fine tune it every time
    os.makedirs(save_weights_dir, exist_ok=True)
    model.save_weights(save_weights_dir + "/bf16_model_weights.h5")

else:
    try:
        model.load_weights(load_weights_dir + "/bf16_model_weights.h5")
    except FileNotFoundError:
        sys.exit("\n\nTuned model weights not available. Tune model first by setting parameter -t=True")

bf16_noAmx_training_time = end_time-start_time
print("Training model with BF16 without AMX")

#########################################################################################

# BF16 with AMX
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
# If not previously set, still need to setup the experiemntal options
# tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})

transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = build_model(transformer_layer, max_len=160)

# fine tune model according to disaster tweets dataset
if is_tune_model:
    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    train_labels = train.target.values
    start_time = time.time()
    train_history = model.fit(train_input, train_labels, validation_split=0.2, epochs=4, batch_size=64)
    end_time = time.time()
  # save model weights so we don't have to fine tune it every time
    os.makedirs(save_weights_dir, exist_ok=True)
    model.save_weights(save_weights_dir + "/AMX_bf16_model_weights.h5")

else:
    try:
        model.load_weights(load_weights_dir + "/AMX_bf16_model_weights.h5")
    except FileNotFoundError:
        sys.exit("\n\nTuned model weights not available. Tune model first by setting parameter -t=True")

bf16_withAmx_training_time = end_time-start_time
print("Training model with BF16 with AMX")

print("Summary")
print("FP32 training time: %.3f" %fp32_training_time)
print("BF16 without AMX training time: %.3f" %bf16_noAmx_training_time)
print("BF16 with AMX training time: %.3f" %bf16_withAmx_training_time)

plt.figure()
plt.title("DistilBERT Training Time")
plt.xlabel("Test Case")
plt.ylabel("Training Time (seconds)")
plt.bar(["FP32", "BF16 no AMX", "BF16 with AMX"], [fp32_training_time, bf16_noAmx_training_time, bf16_withAmx_training_time])

speedup_from_fp32 = fp32_training_time / bf16_withAmx_training_time
print("BF16 with AMX is %.2fX faster than FP32" %speedup_from_fp32)
speedup_from_bf16 = bf16_noAmx_training_time / bf16_withAmx_training_time
print("BF16 with AMX is %.2fX faster than BF16 without AMX" %speedup_from_bf16)

plt.figure()
plt.title("AMX Speedup")
plt.xlabel("Test Case")
plt.ylabel("Speedup")
plt.bar(["FP32", "BF16 no AMX"], [speedup_from_fp32, speedup_from_bf16])

print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')


# # run inference
# test_input = bert_encode(test.text.values, tokenizer, max_len=160)

# if profiling_needed:
#   tf.profiler.experimental.start(log_dir)
#   test_pred = model.predict(test_input, verbose=1)
#   tf.profiler.experimental.stop(save=True)
# else:
#   start = time.time()
#   test_pred = model.predict(test_input, verbose=1)
#   execution_time = time.time() - start
#   # print execution time
#   print(" Execution time is " + str(execution_time))

# # save results in the form of CSV file 
# classified_results['target'] = test_pred.round().astype(int)
# classified_results.to_csv('classified_results.csv', index=False)
