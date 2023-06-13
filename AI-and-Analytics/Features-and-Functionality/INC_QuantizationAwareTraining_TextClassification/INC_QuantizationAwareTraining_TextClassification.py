#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================
# Copyright © 2023 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # Fine-tuning text classification model with Intel® Neural Compressor (INC) Quantization Aware Training
# 
# This code sample will show you how to fine tune BERT text model for text multi-class classification task using Quantization Aware Training provided as part of Intel® Neural Compressor (INC).
# 
# Before we start, please make sure you have installed all necessary libraries to run this code sample.

# ## Loading model 
# 
# We decided to use really small model for this code sample which is `prajjwal1/bert-tiny` but please feel free to use different model changing `model_id` to other name form Hugging Face library or your local model path (if it is compatible with Hugging Face API). 
# 
# Keep in mind that using bigger models like `bert-base-uncased` can improve the final result of the classification after fine-tuning process but it is also really resources and time consuming.

# In[ ]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "prajjwal1/bert-tiny"
model = AutoModelForSequenceClassification.from_pretrained(model_id,  num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)

# The directory where the quantized model will be saved
save_dir = "quantized_model"


# ## Dataset
# 
# We are using `emotion` [dataset form Hugging Face](https://huggingface.co/datasets/dair-ai/emotion). This dataset has 2 different configurations - **split** and **unsplit**. 
# 
# In this code sample we are using split configuration. It contains in total 20 000 examples split into train (16 000 texts), test (2 000 texts) and validation (2 000 text) datasets. We decided to use split dataset instead of unsplit configuration as it contains over 400 000 texts which is overkill for fine-tuning.
# 
# After loading selected dataset we will take a look at first 10 rows of train dataset. You can always change the dataset for different one, just remember to change also number of labels parameter provided when loading the model.

# In[ ]:


from datasets import load_dataset

dataset = load_dataset("emotion", name="split")
dataset['train'][:10]


# Dataset contains 6 different labels represented by digits from 0 to 5. Every digit symbolizes different emotion as followed:
# 
# * 0 - sadness
# * 1 - joy
# * 2 - love
# * 3 - anger
# * 4 - fear
# * 5 - surprise
# 
# In the cell below we conducted few computations on training dataset to better understand how the data looks like. We are analyzing only train dataset as the test and validation datasets have similar data distribution.
# 
# As you can see, distribution opf classed in dataset is not equal. Having in mind that the train, test and validation distributions are similar this is not a problem for our case. 


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

sadness = dataset['train']['label'].count(0)
joy = dataset['train']['label'].count(1)
love = dataset['train']['label'].count(2)
anger = dataset['train']['label'].count(3)
fear = dataset['train']['label'].count(4)
surprise = dataset['train']['label'].count(5)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
frames = [joy, sadness, anger, fear, love, surprise]
ax.bar(labels, frames)
plt.show()


# # Tokenization
# 
# Next step is to tokenize the dataset. 
# 
# **Tokenization** is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters etc. It means that tokenizer breaks unstructured data (natural language text) into chunks of information that can be considered as discrete elements. The tokens can be used later in a vector representation of that document. 
# 
# In other words tokenization change an text document into a numerical data structure suitable for machine and deep learning. 
# 
# To do that, we created function that takes every text from dataset and tokenize it with maximum token length being 128. After that we can se how the structure of the dataset change.


# In[ ]:


def tokenize_data(example):
    return tokenizer(example['text'], padding=True, max_length=128)

dataset = dataset.map(tokenize_data, batched=True)
dataset


# Before we start fine-tuning, let's see how the model in current state performs against validation dataset.
# 
# First, we need to prepare metrics showing model performance. We decided to use accuracy as a performance measure in this specific task. As the model was not created for this specific task, we can assume that the accuracy will not be high.


# In[ ]:


import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Then, we are creating evaluator to see how pre-trained model classify emotions.
# We have to specify:
# * model on which the evaluation will happen - provide the same `model_id` as before,
# * dataset - in our case this is validation dataset,
# * metrics - as specified before, in our case accuracy,
# * label mapping - to map label names with corresponding digits.
# 
# After the evaluation, we just show the results, which are as expected not the best. At this point model is not prepared for emotion classification task.


# In[ ]:


from evaluate import evaluator

task_evaluator = evaluator("text-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline=model_id,
    data=dataset['validation'],
    metric=metric,
    label_mapping={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2, "LABEL_3": 3, "LABEL_4": 4, "LABEL_5": 5}
)
eval_results


# # Quantization Aware Training
# 
# Now, we can move to fine-tuning with quantization. But first, please review the definition of quantization and quantization aware training.
# 
# **Quantization** is a systematic reduction of the precision of all or several layers within the model. This means, a higher-precision type, such as the single precision floating-point (FP32) is converted into a lower-precision type, such as FP16 (16 bits) or INT8 (8 bits).
# 
# **Quantization Aware Training** replicates inference-time quantization, resulting in a model that downstream tools may utilize to generate actually quantized models. In other words, it provides quantization to the model during training (or fine-tuning like in our case) based on provided quantization configuration.
# 
# Having that in mind, we can provide configuration for the Quantization Aware Training form Intel® Neural Compressor.


# In[ ]:


from neural_compressor import QuantizationAwareTrainingConfig

# The configuration detailing the quantization process
quantization_config = QuantizationAwareTrainingConfig()


# The next step is to create trainer for our model. We will use Intel® Neural Compressor optimize trainer form `optimum.intel` package.
# We need to provide all necessary parameters to the trainer:
# 
# * initialized model and tokenizer
# * configuration for quantization aware training
# * training arguments that includes: directory where model will be saved, number of epochs
# * datasets for training and evaluation
# * prepared metrics that allow us to see the progress in training
# 
# For purpose of this code sample, we decided to train model by just 2 epochs, to show you how the quantization aware training works and that the fine-tuning really improves the results of the classification. If you wan to receive better accuracy results, yoy can easily incise the number of epochs up to 5 and observe how model learns. Keep in mind that the process may take some time - the more epochs you will use, the training time will be longer.


# In[ ]:


from optimum.intel import INCModelForSequenceClassification, INCTrainer
from transformers import TrainingArguments

trainer = INCTrainer(
    model=model,
    quantization_config=quantization_config,
    args=TrainingArguments(save_dir, num_train_epochs=2.0, do_train=True, do_eval=False),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# ## Train the model
# 
# Now, let's train the model. We will use prepared trainer by executing `train` method on it.
# 
# You can see, that after the training information about the model are printed under `*****Mixed Precision Statistics*****`. 
# 
# Now, the model use INT8 instead of FP32 in every layer.


# In[ ]:


train_result = trainer.train()


# ## Evaluate the model
# 
# After the training we should evaluate our model using `evaluate()` method on prepared trainer. It will show results for prepared before evaluation metrics - evaluation accuracy and loss. Additionally we will have information about evaluation time, samples and steps per second and number of epochs model was trained by. 


# In[ ]:


metrics = trainer.evaluate()
metrics


# After the training it is important to save the model. One again we will use prepared trainer and other method - `save_model()`. Our model will be saved in the location provided before.
# After that, to use this model in the future you just need load it similarly as at the beginning, using dedicated Intel® Neural Compressor optimized method `INCModelForSequenceClassification.from_pretrained(...)`. 


# In[ ]:


# To use model in the future - save it!
trainer.save_model()
model = INCModelForSequenceClassification.from_pretrained(save_dir)


# In this code sample we use BERT-tiny and emotion dataset to create text classification model using Intel® Neural Compressor Quantization Aware Training. We encourage you to experiment with this code sample changing model and datasets to make text models for different classification tasks. 


# In[ ]:


print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")

