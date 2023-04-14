# details about this Python script 
# - Enable auto-mixed precision with few code changes for faster inference.
# - Image Classification task using TensorFlow Hub's ResNet50v1.5 pretrained model.
# - Export the optimized model in the SavedModel format.

# Importing libraries
import os
import numpy as np
import time
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import requests
from copy import deepcopy
print("We are using Tensorflow version: ", tf.__version__)

# Check if hardware supports AMX

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
    sys.exit("AMX is not supported on current hardware. Code sample cannot be run.\n")


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
data_root = tf.keras.utils.get_file(
  'flower_photos',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

batch_size = 512
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = np.array(train_ds.class_names)
print("The flower dataset has " + str(len(class_names)) + " classes: ", class_names)

# import pre-trained fp_32 model
fp32_model = tf.keras.models.load_model('models/my_saved_model_fp32')

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

import time
start = time.time()
fp32_history = fp32_model.evaluate(val_ds)
fp32_inference_time = time.time() - start



# Reload the model as the bf16 model with AVX512 to compare inference time
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_BF16"
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
bf16_model_noAmx = tf.keras.models.load_model('models/my_saved_model_fp32')

bf16_model_noAmx_export_path = "models/my_saved_model_bf16_noAmx"
bf16_model_noAmx.save(bf16_model_noAmx_export_path)

start = time.time()
bf16_noAmx_history = bf16_model_noAmx.evaluate(val_ds)
bf16_noAmx_inference_time = time.time() - start


# Reload the model as the bf16 model with AMX to compare inference time
os.environ["ONEDNN_MAX_CPU_ISA"] = "AMX_BF16"
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
bf16_model_withAmx = tf.keras.models.load_model('models/my_saved_model_fp32')

bf16_model_withAmx_export_path = "models/my_saved_model_bf16_with_amx"
bf16_model_withAmx.save(bf16_model_withAmx_export_path)

start = time.time()
bf16_withAmx_history = bf16_model_withAmx.evaluate(val_ds)
bf16_withAmx_inference_time = time.time() - start


# Summary of results
print("Summary")
print("FP32 inference time: %.3f" %fp32_inference_time)
print("BF16 with AVX512 inference time: %.3f" %bf16_noAmx_inference_time)
print("BF16 with AMX inference time: %.3f" %bf16_withAmx_inference_time)

import matplotlib.pyplot as plt
plt.figure()
plt.title("Resnet50 Inference Time")
plt.xlabel("Test Case")
plt.ylabel("Inference Time (seconds)")
plt.bar(["FP32", "BF16 with AVX512", "BF16 with AMX"], [fp32_inference_time, bf16_noAmx_inference_time, bf16_withAmx_inference_time])

fp32_inference_accuracy = fp32_history[1]
bf16_noAmx_inference_accuracy = bf16_noAmx_history[1]
bf16_withAmx_inference_accuracy = bf16_withAmx_history[1]
plt.figure()
plt.title("Resnet50 Inference Accuracy")
plt.xlabel("Test Case")
plt.ylabel("Inference Accuracy")
plt.bar(["FP32", "BF16 with AVX512", "BF16 with AMX"], [fp32_inference_accuracy, bf16_noAmx_inference_accuracy, bf16_withAmx_inference_accuracy])

print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')
