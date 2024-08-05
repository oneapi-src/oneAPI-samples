#!/usr/bin/env python
# coding: utf-8

# # Complete your thoughts with GPT-J On Intel Xeon using TensorFlow

# This notebook uses HuggingFace's GPT-J model to perform text generation on Intel Xeon

# ## Model :GPT-J (6B)
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
# The model is trained with a tokenization vocabulary of 50257, using the same set of Byte Pair Encoding(BPEs) as GPT-2/GPT-3.*
#

# In[1]:


# importing libraries
import tensorflow as tf
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModelForCausalLM
)
import time
import warnings
warnings.filterwarnings('ignore')


# ### Get Config and Tokenizer for the model

# In[2]:


tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

model_name = "EleutherAI/gpt-j-6B"
max_output_tokens = 32

# Initialize the text tokenizer
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'left'


# ### Load Model

# In[3]:


# Load the model weights
model = TFAutoModelForCausalLM.from_pretrained(model_name, config=config)
model.compile()


# In[4]:


generate_kwargs = dict(do_sample=False, num_beams=4, eos_token_id=model.config.eos_token_id)
gen = tf.function(lambda x: model.generate(x, max_new_tokens=max_output_tokens, **generate_kwargs))


# In[5]:


def complete_my_thought(x):
    tokenized_data = tokenizer([x], return_tensors="tf").input_ids
    output = gen(tokenized_data)
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    return decoded


# In[6]:


warmup_sentence = "This is a warmup sentence. Warmup helps get the model ready to showcase its capabilities."


# In[7]:


complete_my_thought(warmup_sentence);


# ## Start Text Generation

# In[8]:


input_sentence1 = "Ann Arbor is very pleasant in summers. The Huron river is an ideal spot for people to"
input_sentence2 = "Space is an intersting place. Stephen Hawking hypothesized that there might be multiple universes in which"
input_sentence3 = "In a shocking finding, scientists discovered a herd of unicorns living in a remote previously unexplored"
input_sentence4 = "Coffee is one of the most popular drinks in the world. It goes very well with"
input_sentence5 = "Dogs are often referred to as man's best friend. There are a number of reasons why"


# In[9]:


out = complete_my_thought(input_sentence1)
print(out)


# In[10]:


complete_my_thought(input_sentence2)


# In[11]:


complete_my_thought(input_sentence3)


# In[12]:


complete_my_thought(input_sentence4)


# In[13]:


complete_my_thought(input_sentence5)


# In[ ]:




