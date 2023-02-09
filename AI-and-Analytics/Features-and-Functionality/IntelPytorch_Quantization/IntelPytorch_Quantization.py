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
import os
from time import time
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert

# Hyperparameters and constants
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'
WARMUP = 3
ITERS = 100


def inference(model, data):
    # Warmup for several iteration.
    for i in range(WARMUP):
        out = model(data)

    # Benchmark: accumulate inference time for multi iteration and calculate the average inference time.
    print("Inference ...")
    inference_time = 0
    for i in range(ITERS):
        start_time = time()
        _ = model(data)
        end_time = time()
        inference_time = inference_time + (end_time - start_time)



    inference_time = inference_time / ITERS
    print("Inference Time Avg: ", inference_time)
    return inference_time


def staticQuantize(model_fp32, data, calibration_data_loader):
    # Acquire inference times for static quantization INT8 model 
    qconfig_static = ipex.quantization.default_static_qconfig
    # Alternatively, define your own qconfig:
    # from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    # qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
    #        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    prepared_model_static = prepare(model_fp32, qconfig_static, example_inputs=data, inplace=False)
    print("Calibration with Static Quantization ...")
    for batch_idx, (data, target) in enumerate(calibration_data_loader):
        prepared_model_static(data)
        if batch_idx % 10 == 0:
            print("Batch %d/%d complete, continue ..." %(batch_idx+1, len(calibration_data_loader)))
    print("Calibration Done")

    converted_model_static = convert(prepared_model_static)
    with torch.no_grad():
        traced_model_static = torch.jit.trace(converted_model_static, data)
        traced_model_static = torch.jit.freeze(traced_model_static)

    traced_model_static.save("quantized_model_static.pt")
    return traced_model_static

def dynamicQuantize(model_fp32, data):
    # Acquire inference times for dynamic quantization INT8 model
    qconfig_dynamic = ipex.quantization.default_dynamic_qconfig
    print("Quantize Model with Dynamic Quantization ...")

    prepared_model_dynamic = prepare(model_fp32, qconfig_dynamic, example_inputs=data, inplace=False)

    converted_model_dynamic = convert(prepared_model_dynamic)
    with torch.no_grad():
        traced_model_dynamic = torch.jit.trace(converted_model_dynamic, data)
        traced_model_dynamic = torch.jit.freeze(traced_model_dynamic)

    # save the quantized static model 
    traced_model_dynamic.save("quantized_model_dynamic.pt")
    return traced_model_dynamic


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
    model_fp32 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model_fp32.eval()

    if not os.path.exists('quantized_model_static.pt'):
        # Static Quantizaton & Save Model to quantized_model_static.pt
        print('quantize the model with static quantization')
        staticQuantize(model_fp32, data, calibration_data_loader)

    if not os.path.exists('quantized_model_dynamic.pt'):
        # Dynamic Quantization & Save Model to quantized_model_dynamic.pt
        print('quantize the model with dynamic quantization')
        dynamicQuantize(model_fp32, data)

    print("Inference with FP32")
    fp32_inference_time = inference(model_fp32, data)

    traced_model_static = torch.jit.load('quantized_model_static.pt')
    traced_model_static.eval()
    traced_model_static = torch.jit.freeze(traced_model_static)
    print("Inference with Static INT8")
    int8_inference_time_static = inference(traced_model_static, data)

    traced_model_dynamic = torch.jit.load('quantized_model_dynamic.pt')
    traced_model_dynamic.eval()
    traced_model_dynamic = torch.jit.freeze(traced_model_dynamic)
    print("Inference with Dynamic INT8")
    int8_inference_time_dynamic = inference(traced_model_dynamic, data)

    # Inference time results
    print("Summary")
    print("FP32 inference time: %.3f" %fp32_inference_time)
    print("INT8 static quantization inference time: %.3f" %int8_inference_time_static)
    print("INT8 dynamic quantization inference time: %.3f" %int8_inference_time_dynamic)

    # Calculate speedup when using quantization
    speedup_from_fp32_static = fp32_inference_time / int8_inference_time_static
    print("Staic INT8 %.2fX faster than FP32" %speedup_from_fp32_static)
    speedup_from_fp32_dynamic = fp32_inference_time / int8_inference_time_dynamic
    print("Dynamic INT8 %.2fX faster than FP32" %speedup_from_fp32_dynamic)


if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')