
# details about this Python script 
# - Enable auto-mixed precision with few code changes for faster inference.
# - Image Classification task using [TensorFlow Hub's](https://www.tensorflow.org/hub) ResNet50v1.5 pretrained model.
# - Export the optimized model in the [SavedModel](https://www.tensorflow.org/guide/saved_model) format.


# Importing libraries
import os
import sys
import numpy as np
import time
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import requests
from copy import deepcopy

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)

# Check if hardware supports Intel® AMX

from cpuinfo import get_cpu_info
info = get_cpu_info()
flags = info['flags']
amx_supported = False
for flag in flags:
    if "amx" in flag:
        amx_supported = True
        print("Intel® AMX is supported on current hardware. Code sample can be run.\n")
if not amx_supported:
    print("Intel® AMX is not supported on current hardware. Code sample cannot be run.\n")
    sys.exit("Intel® AMX is not supported on current hardware. Code sample cannot be run.\n")

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

IMAGE_SIZE = (224, 224, 3)
model_handle = "https://www.kaggle.com/models/google/resnet-v1/TensorFlow2/50-feature-vector/2"

print("Building model with", model_handle)
fp32_model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE),
    hub.KerasLayer(model_handle, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(class_names),
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
fp32_model.build((None,)+IMAGE_SIZE)
fp32_model.summary()

fp32_model.compile(
  optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

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

for image_batch, labels_batch in val_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

fp32_model.save("models/my_saved_model_fp32")

# Reload the model as the bf16 model with AVX512 to compare inference time
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_BF16"
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
bf16_model_noAmx = tf.keras.models.load_model('models/my_saved_model_fp32')

bf16_model_noAmx_export_path = "models/my_saved_model_bf16_noAmx"
bf16_model_noAmx.save(bf16_model_noAmx_export_path)

start = time.time()
bf16_noAmx_history = bf16_model_noAmx.evaluate(val_ds)
bf16_noAmx_inference_time = time.time() - start

# Reload the model as the bf16 model with Intel® AMX to compare inference time
os.environ["ONEDNN_MAX_CPU_ISA"] = "AMX_BF16"
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
bf16_model_withAmx = tf.keras.models.load_model('models/my_saved_model_fp32')

bf16_model_withAmx_export_path = "models/my_saved_model_bf16_with_amx"
bf16_model_withAmx.save(bf16_model_withAmx_export_path)

start = time.time()
bf16_withAmx_history = bf16_model_withAmx.evaluate(val_ds)
bf16_withAmx_inference_time = time.time() - start

print("Summary")
print("FP32 inference time: %.3f" %fp32_inference_time)
print("BF16 with AVX512 inference time: %.3f" %bf16_noAmx_inference_time)
print("BF16 with Intel® AMX inference time: %.3f" %bf16_withAmx_inference_time)

import matplotlib.pyplot as plt
plt.figure()
plt.title("Resnet50 Inference Time")
plt.xlabel("Test Case")
plt.ylabel("Inference Time (seconds)")
plt.bar(["FP32", "BF16 with AVX512", "BF16 with Intel® AMX"], [fp32_inference_time, bf16_noAmx_inference_time, bf16_withAmx_inference_time])

speedup_bf16_noAMX_from_fp32 = fp32_inference_time / bf16_noAmx_inference_time
print("BF16 with AVX512 is %.2fX faster than FP32" %speedup_bf16_noAMX_from_fp32)
speedup_bf16_withAMX_from_fp32 = fp32_inference_time / bf16_withAmx_inference_time
print("BF16 with Intel® AMX is %.2fX faster than FP32" %speedup_bf16_withAMX_from_fp32)
speedup_bf16_withAMX_from_bf16 = bf16_noAmx_inference_time / bf16_withAmx_inference_time
print("BF16 with Intel® AMX is %.2fX faster than BF16 with AVX512" %speedup_bf16_withAMX_from_bf16)

plt.figure()
plt.title("Intel® AMX Speedup")
plt.xlabel("Test Case")
plt.ylabel("Speedup")
plt.bar(["FP32", "BF16 with AVX512", "BF16 with Intel® AMX"], [1, speedup_bf16_noAMX_from_fp32, speedup_bf16_withAMX_from_fp32])

fp32_inference_accuracy = fp32_history[1]
bf16_noAmx_inference_accuracy = bf16_noAmx_history[1]
bf16_withAmx_inference_accuracy = bf16_withAmx_history[1]
plt.figure()
plt.title("Resnet50 Inference Accuracy")
plt.xlabel("Test Case")
plt.ylabel("Inference Accuracy")
plt.bar(["FP32", "BF16 with AVX512", "BF16 with Intel® AMX"], [fp32_inference_accuracy, bf16_noAmx_inference_accuracy, bf16_withAmx_inference_accuracy])

print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')


