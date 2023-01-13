#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2022 Intel Corporation
 SPDX-License-Identifier: MIT
==============================================================
'''

import torch
import torchvision
import tqdm
from time import time
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert

# Hyperparameters and constants
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'
ITERS = 100


"""
Perform all types of training in main function
"""
def main():
    
    # Load dataset
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=False,
            transform=transform,
            download=DOWNLOAD,
    )
    calibration_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=128
    )

    data = torch.rand(1, 3, 224, 224)
    # Acquire inference times for FP32 model
    model_fp32 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model_fp32.eval()

    start_time = time()
    print("Inference with FP32")
    for i in tqdm.tqdm(range(ITERS)):
        out = model_fp32(data)
    end_time = time()

    fp32_inference_time = (end_time - start_time) / ITERS

    # Acquire inference times for static quantization INT8 model 
    qconfig_static = ipex.quantization.default_static_qconfig
    # Alternatively, define your own qconfig:
    #from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    #qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
    #        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    prepared_model_static = prepare(model_fp32, qconfig_static, example_inputs=data, inplace=False)
    print("Calibration with Static Quantization")
    for (data, _) in tqdm.tqdm(calibration_data_loader):
        prepared_model_static(data)

    converted_model_static = convert(prepared_model_static)
    with torch.no_grad():
        traced_model_static = torch.jit.trace(converted_model_static, data)
        traced_model_static = torch.jit.freeze(traced_model_static)
    
    # save the quantized static model 
    traced_model_static.save("quantized_model_static.pt")

    start_time = time()
    print("Inference with Static Quantization")
    for i in tqdm.tqdm(range(ITERS)):
        out = traced_model_static(data)
    end_time = time()

    int8_inference_time_static = (end_time - start_time) / ITERS

    # Acquire inference times for dynamic quantization INT8 model
    qconfig_dynamic = ipex.quantization.default_dynamic_qconfig
    # Alternatively, define your own qconfig:
    #from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    #qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
    #        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    prepared_model_dynamic = prepare(model_fp32, qconfig_dynamic, example_inputs=data, inplace=False)

    converted_model_dynamic = convert(prepared_model_dynamic)
    with torch.no_grad():
        traced_model_dynamic = torch.jit.trace(converted_model_dynamic, data)
        traced_model_dynamic = torch.jit.freeze(traced_model_dynamic)
    
    # save the quantized static model 
    traced_model_dynamic.save("quantized_model_dynamic.pt")

    start_time = time()
    print("Inference with Dynamic Quantization")
    for i in tqdm.tqdm(range(ITERS)):
        out = traced_model_dynamic(data)
    end_time = time()

    int8_inference_time_dynamic = (end_time - start_time) / ITERS


    # Inference time results
    print("Summary")
    print("FP32 inference time: %.3f" %fp32_inference_time)
    print("INT8 static quantization inference time: %.3f" %int8_inference_time_static)
    print("INT8 dynamic quantization inference time: %.3f" %int8_inference_time_dynamic)

    # Create bar chart with training time results
    plt.figure()
    plt.title("ResNet Inference Time")
    plt.xlabel("Test Case")
    plt.ylabel("Inference Time (seconds)")
    plt.bar(["FP32", "INT8 static Quantization", "INT8 dynamic Quantization"], [fp32_inference_time, int8_inference_time_static, int8_inference_time_dynamic])

    # Calculate speedup when using quantization
    speedup_from_fp32_static = fp32_inference_time / int8_inference_time_static
    print("Staic INT8 %.2fX faster than FP32" %speedup_from_fp32_static)
    speedup_from_fp32_dynamic = fp32_inference_time / int8_inference_time_dynamic
    print("Dynamic INT8 %.2fX faster than FP32" %speedup_from_fp32_dynamic)


    # Create bar chart with speedup results
    plt.figure()
    plt.title("Quantization Speedup")
    plt.xlabel("Test Case")
    plt.ylabel("Speedup")
    plt.bar(["FP32-static INT8", "FP32-dynamic INT8"], [speedup_from_fp32_static, speedup_from_fp32_dynamic])
    


if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')