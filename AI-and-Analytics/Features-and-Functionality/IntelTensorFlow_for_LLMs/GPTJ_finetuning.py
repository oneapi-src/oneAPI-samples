#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning GPT-J for GLUE cola dataset.

# ## Model : GPT-J (6B)
#  **[GPT-J(6B)] (https://huggingface.co/EleutherAI/gpt-j-6b): released in March 2021.It was the largest open source GPT-3-style language model in the world at the time of release.**
#
#  **GPT-J is similar to ChatGPT in ability, although it does not function as a chat bot, only as a text predictor.   Developed using Mesh     Tranformer & xmap in JAX**
#
#  *The model consists of :
# >
#      - 28 layers
#      - Model dimension of 4096
#      - Feedforward dimension of 16384
#      - 16 heads, each with a dimension of 256.*
# >
# *The model is trained with a tokenization vocabulary of 50257, using the same set of Byte Pair Encoding(BPEs) as GPT-2/GPT-3.*
#

# ## Dataset : GLUE cola
# *CoLA The Corpus of Linguistic Acceptability (Warstadt et al., 2018) consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a
# sequence of words annotated with whether it is a grammatical English sentence.*
#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
# **Import python packages**

# In[1]:


import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# **Import needded classes from HuggingFace transformers library**

# In[2]:


import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    TFGPTJForSequenceClassification,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

check_min_version("4.27.0.dev0")


# **Default options for the data.These options can be converted to command line options**
# *Data, Model & Training options*

# In[3]:


class DataArgs :
  def __init__(self):
    self.task_name = "cola"
    self.precision = "bfloat16"
    self.intra_op_parallelism_threads=56
    self.inter_op_parallelism_threads=2
    self.max_seq_length=128
    self.checkpoint_save_freq = 500
    self.overwrite_cache=True
    self.max_train_samples=None
    self.max_eval_samples=None
    self.max_predict_samples=12
    self.output_dir ="./output"

class ModelArgs :
  def __init__(self):
    self.model_name_or_path = "EleutherAI/gpt-j-6B"
    self.cache_dir=None
    self.model_revision="main"
    self.steps=0

class TrainingArgs :
  def __init__(self):
    self.local_rank =-1
    self.seed =77
    self.num_replicas_in_sync=1
    self.per_device_train_batch_size=64
    self.per_device_eval_batch_size=64
    self.do_train=True
    self.do_predict=True
    self.do_eval=True
    self.num_train_epochs=1.0
    self.learning_rate=5e-06
    self.output_dir ="./output"
    self.xla =False


data_args = DataArgs()
model_args = ModelArgs()
training_args = TrainingArgs()


# ### Set precision and set inter_op and intra op thread settings for best performance
#
# *Bfloat16 training gives 2x+ performance compared to fp32 on 4th gen Xeon*
#

# In[4]:


if data_args.precision == "bfloat16" :
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
tf.config.threading.set_inter_op_parallelism_threads(data_args.inter_op_parallelism_threads)
tf.config.threading.set_intra_op_parallelism_threads(data_args.intra_op_parallelism_threads)


# *Default Setting for region Logging and transformer verbosity*

# In[5]:


logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

if is_main_process(training_args.local_rank):
   transformers.utils.logging.set_verbosity_info()
   transformers.utils.logging.enable_default_handler()
   transformers.utils.logging.enable_explicit_format()
   logger.info(f"Training/evaluation parameters {training_args}")
# endregion

set_seed(training_args.seed)


# ### Download and load the dataset from the hub.

# In[6]:


raw_datasets = load_dataset(
    "glue",
    data_args.task_name,
    cache_dir=model_args.cache_dir,
)

#Check the dataset schema and Sample data**

print(raw_datasets)
print(raw_datasets['train'][0])


# ### Load model config, tokenizer**
# **Toekization of dataset : Using gpt2 tokenizer**

# In[7]:


label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
)

#Load tokenizer for toekization of dataset : Using gpt2 tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2" if model_args.model_name_or_path == "EleutherAI/gpt-j-6B" else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=True,
    revision=model_args.model_revision,
)


# **Add special tokens for padding as GPT does not have a padding token.**.
# *Keys used by tokenizer to select text to be tokenized.
# Data set used cola.*

# In[8]:


task_to_keys = {
    "cola": ("sentence", None),
}
logger = logging.getLogger(__name__)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config.pad_token_id=0
sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

#Some models have set the order of the labels to use, so let's make sure we do use it*

label_to_id = None
config.label2id = {l: i for i, l in enumerate(label_list)}
config.id2label = {id: label for label, id in config.label2id.items()}
print("  Label to ID :", config.label2id)
print("  ID to Label :", config.id2label)


# ### Define the tokenizer process function. This is called by tokenizer to tokenize relevant data*

# In[9]:


max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
def preprocess_function(examples):
    # Tokenize the texts
    args = (
       (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)
    return result

#Let us no tokenize dataset and set a DataCollator for batching and any padding.

datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="np")


# ### A few utility fns
#    >
#      1. To convert raw dataset to tf_dataset.
#      2. Number of steps for trainng.
#      3. Adam optimizer with decay.
#      4. Call backs for model training.*
#

# In[10]:


def convert_to_tf_Dataset(datasets):
    # Convert data to a tf.data.Dataset
    dataset_options = tf.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    num_replicas = -1 #training_args.strategy.num_replicas_in_sync
    tf_data = {}
    max_samples = {
            "train": data_args.max_train_samples,
            "validation": data_args.max_eval_samples,
            "test": data_args.max_predict_samples,
    }
    num_replicas=1
    for key in datasets.keys():
        if key == "train" or key.startswith("validation"):
            assert "label" in datasets[key].features, f"Missing labels from {key} data!"
        if key == "train":
            shuffle = True
            batch_size = training_args.per_device_train_batch_size * num_replicas
        else:
            shuffle = False
            batch_size = training_args.per_device_eval_batch_size * num_replicas
        samples_limit = max_samples[key]
        dataset = datasets[key]
        if samples_limit is not None:
            dataset = dataset.select(range(samples_limit))

        # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
        # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
        # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
        # yourself if you use this method, whereas they are automatically inferred from the model input names when
        # using model.prepare_tf_dataset()
        # For more info see the docs:
        data = model.prepare_tf_dataset(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=data_collator,
                tokenizer=tokenizer,
        )
        data = data.with_options(dataset_options)
        tf_data[key] = data
    return tf_data

#Utility fn to compute total number of steps*

def compute_num_train_steps(tf_data):
    if training_args.do_train:
        if model_args.steps:
            num_train_steps = model_args.steps
            if num_train_steps > int(len(tf_data["train"])) :
                # for single epoch
                num_train_steps = int(len(tf_data["train"]))
        else :
            num_train_steps = len(tf_data["train"]) * training_args.num_train_epochs
    return num_train_steps

#Function to define Adam optimizer with Polynomialdecay*

def adam_optimizer_with_decay(num_train_steps):
    end_lr = (training_args.learning_rate)/np.sqrt(num_train_steps)
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=training_args.learning_rate,
        end_learning_rate=end_lr, decay_steps=num_train_steps
    )
    opt = Adam(learning_rate=lr_scheduler)
    return opt

#Call back for checkpointing if needed*

def get_callbacks():
    callbacks = []
    checkpoint=None
    if (checkpoint) :
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
           filepath=training_args.output_dir,
           save_weights_only=True,
           monitor='accuracy',
           mode='max',
           save_freq=data_args.checkpoint_save_freq,
           save_best_only=True,
        )
        callbacks.append(checkpoint_callback)
    return callbacks


# ### Main steps
# **Load the model : use model name and config**

# In[11]:


model = TFGPTJForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
)


# ### Convert raw dataset to tf dataset & compile the model**
#
# **Get Optimizerand loss. Compile the model**

# In[12]:


tf_data = convert_to_tf_Dataset(datasets)

#Get Optimizer,  and loss and compile the model*
num_train_steps = compute_num_train_steps(tf_data)
optimizer= adam_optimizer_with_decay(num_train_steps)
model.compile(optimizer=optimizer, metrics=["accuracy"], jit_compile=training_args.xla)


# ### Fit the model : Training

# In[13]:


callbacks= get_callbacks()
steps_pe = int(len(tf_data["train"]))
model.fit(
    tf_data["train"],
    validation_data=tf_data["validation"],
    epochs=int(training_args.num_train_epochs),
    steps_per_epoch=steps_pe,
    callbacks=callbacks,
    verbose=1,
)


# **Let us save and reload the model**

# if training_args.output_dir :
#     # If we're not pushing to hub, at least save a local copy when we're done
#     print("Save the model id dir :",training_args.output_dir)
#     model.save_pretrained(training_args.output_dir)

# ### Let us check some classifications

# In[14]:


# Show results for test
# Show results for test
metric = evaluate.load("glue", data_args.task_name)
def show_results(class_preds, key):
    for i in range(7):
      pred = int(class_preds[i])
      pred_label = config.id2label[pred]
      if data_args.task_name != 'mrpc':
        print(f"Sentence : {raw_datasets[key][i]['sentence']} : {pred_label}")
      else:
        sent = raw_datasets[key][i]['sentence1'] + " : " + raw_datasets[key][i]['sentence2']
        print(f"Sentences : {sent} : {pred_label}")

def val_predict(model, tf_data, key):
    print("====================",key, "=========================")
    preds = model.predict(tf_data[key])["logits"]
    print(" Done predictions:..")
    class_preds = tf.math.argmax(preds, axis=1)
    if key != "test":
      print(f"{key} Accuracy :", accuracy_score(class_preds,raw_datasets[key]["label"]))
      print(metric.compute(predictions=class_preds, references=raw_datasets[key]["label"]))
    else :
      show_results(class_preds, key)
    print("===================", key, " done.==================")


# In[15]:


val_predict(model, tf_data, "test")


# In[16]:


if training_args.output_dir :
    # If we're not pushing to hub, at least save a local copy when we're done
    print("Save the model id dir :",training_args.output_dir)
    model.save_pretrained(training_args.output_dir)

