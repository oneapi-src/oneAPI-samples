#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2023 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''

import os
from time import time
import matplotlib.pyplot as plt
import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
import torchvision
from torchvision import models
from transformers import BertModel

NUM_SAMPLES = 1000   # number of samples to perform inference on
SUPPORTED_MODELS = ["resnet50", "bert"]   # models supported by this code sample

# BERT sample data parameters
BERT_BATCH_SIZE = 1
BERT_SEQ_LENGTH = 512

"""
Function to perform inference on Resnet50 and BERT
"""
def runInference(model, data, modelName="resnet50", dataType="FP32", amx=True):
    """
    Input parameters
        model: the PyTorch model object used for inference
        data: a sample input into the model
        modelName: str representing the name of the model, supported values - resnet50, bert
        dataType: str representing the data type for model parameters, supported values - FP32, BF16, INT8
        amx: set to False to disable AMX on BF16, Default: True
    Return value
        inference_time: the time in seconds it takes to perform inference with the model
    """
    
    # Display run case
    if amx:
        isa_text = "AVX512_CORE_AMX"
    else:
        isa_text = "AVX512_CORE_VNNI"
    print("%s %s inference with %s" %(modelName, dataType, isa_text))

    # Configure environment variable
    if not amx:
        os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_VNNI"
    else:
        os.environ["ONEDNN_MAX_CPU_ISA"] = "DEFAULT"

    # Special variables for specific models
    if "bert" == modelName:
        d = torch.randint(model.config.vocab_size, size=[BERT_BATCH_SIZE, BERT_SEQ_LENGTH]) # sample data input for torchscript and inference

    # Prepare model for inference based on precision (FP32, BF16, INT8)
    if "INT8" == dataType:
        # Quantize model to INT8 if needed (one time)
        model_filename = "quantized_model_%s.pt" %modelName
        if not os.path.exists(model_filename):
            qconfig = ipex.quantization.default_static_qconfig
            prepared_model = prepare(model, qconfig, example_inputs=data, inplace=False)
            converted_model = convert(prepared_model)
            with torch.no_grad():
                if "resnet50" == modelName:
                    traced_model = torch.jit.trace(converted_model, data)
                elif "bert" == modelName:
                    traced_model = torch.jit.trace(converted_model, (d,), check_trace=False, strict=False)
                else:
                    raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))
            traced_model.save(model_filename)

        # Load INT8 model for inference
        model = torch.jit.load(model_filename)
        model.eval()
        model = torch.jit.freeze(model)
    elif "BF16" == dataType:
        model = ipex.optimize(model, dtype=torch.bfloat16)
        with torch.no_grad():
            with torch.cpu.amp.autocast():
                if "resnet50" == modelName:
                    model = torch.jit.trace(model, data)
                elif "bert" == modelName:
                    model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
                else:
                    raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))
                model = torch.jit.freeze(model)
    else: # FP32
        with torch.no_grad():
            if "resnet50" == modelName:
                model = torch.jit.trace(model, data)
            elif "bert" == modelName:
                model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
            else:
                raise Exception("ERROR: modelName %s is not supported. Choose from %s" %(modelName, SUPPORTED_MODELS))
            model = torch.jit.freeze(model)

    # Run inference
    with torch.no_grad():
        if "BF16" == dataType:
            with torch.cpu.amp.autocast():
                # Warm up
                for i in range(20):
                    model(data)
                
                # Measure latency
                start_time = time()
                for i in range(NUM_SAMPLES):
                    model(data)
                end_time = time()
        else:
            # Warm up
            for i in range(20):
                model(data)
            
            # Measure latency
            start_time = time()
            for i in range(NUM_SAMPLES):
                model(data)
            end_time = time()
    inference_time = end_time - start_time
    print("Inference on %d samples took %.3f seconds" %(NUM_SAMPLES, inference_time))

    return inference_time

"""
Prints out results and displays figures summarizing output.
"""
def summarizeResults(modelName="", results=None):
    """
    Input parameters
        modelName: a str representing the name of the model
        results: a dict with the run case and its corresponding time in seconds
    Return value
        None
    """

    # Inference time results
    print("\nSummary for %s (%d samples)" %(modelName, NUM_SAMPLES))
    for key in results.keys():
        print("%s inference time: %.3f seconds" %(key, results[key]))

    # Create bar chart with inference time results
    plt.figure()
    plt.title("%s Inference Time (%d samples)" %(modelName, NUM_SAMPLES))
    plt.xlabel("Run Case")
    plt.ylabel("Inference Time (seconds)")
    plt.bar(results.keys(), results.values())

    # Calculate speedup when using AMX
    print("\n")
    bf16_with_amx_speedup = results["FP32"] / results["BF16_with_AMX"]
    print("BF16 with AMX is %.2fX faster than FP32" %bf16_with_amx_speedup)
    int8_with_vnni_speedup = results["FP32"] / results["INT8_with_VNNI"]
    print("INT8 with VNNI is %.2fX faster than FP32" %int8_with_vnni_speedup)
    int8_with_amx_speedup = results["FP32"] / results["INT8_with_AMX"]
    print("INT8 with AMX is %.2fX faster than FP32" %int8_with_amx_speedup)
    print("\n\n")

    # Create bar chart with speedup results
    plt.figure()
    plt.title("%s AMX BF16/INT8 Speedup over FP32" %modelName)
    plt.xlabel("Run Case")
    plt.ylabel("Speedup")
    plt.bar(results.keys(), 
        [1, bf16_with_amx_speedup, int8_with_vnni_speedup, int8_with_amx_speedup]
    )

"""
Perform all types of inference in main function

Inference run cases for both Resnet50 and BERT
1) FP32 (baseline)
2) BF16 using AVX512_CORE_AMX
3) INT8 using AVX512_CORE_VNNI
4) INT8 using AVX512_CORE_AMX
"""
def main():
    # Check if hardware supports AMX
    import sys
    sys.path.append('../../')
    import version_check
    from cpuinfo import get_cpu_info
    info = get_cpu_info()
    flags = info['flags']
    amx_supported = False
    for flag in flags:
        if "amx" in flag:
            amx_supported = True
            break
    if not amx_supported:
        print("AMX is not supported on current hardware. Code sample cannot be run.\n")
        return

    # ResNet50
    resnet_model = models.resnet50(pretrained=True)
    resnet_data = torch.rand(1, 3, 224, 224)
    resnet_model.eval()
    fp32_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="FP32", amx=True)
    bf16_amx_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="BF16", amx=True)
    int8_with_vnni_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="INT8", amx=False)
    int8_amx_resnet_inference_time = runInference(resnet_model, resnet_data, modelName="resnet50", dataType="INT8", amx=True)
    results_resnet = {
        "FP32": fp32_resnet_inference_time,
        "BF16_with_AMX": bf16_amx_resnet_inference_time,
        "INT8_with_VNNI": int8_with_vnni_resnet_inference_time,
        "INT8_with_AMX": int8_amx_resnet_inference_time
    }
    summarizeResults("ResNet50", results_resnet)

    # BERT
    bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased') 
    bert_data = torch.randint(bert_model.config.vocab_size, size=[BERT_BATCH_SIZE, BERT_SEQ_LENGTH])
    bert_model.eval()
    fp32_bert_inference_time = runInference(bert_model, bert_data, modelName="bert", dataType="FP32", amx=True)
    bf16_amx_bert_inference_time = runInference(bert_model, bert_data, modelName="bert", dataType="BF16", amx=True)
    int8_with_vnni_bert_inference_time = runInference(bert_model, bert_data, modelName="bert", dataType="INT8", amx=False)
    int8_amx_bert_inference_time = runInference(bert_model, bert_data, modelName="bert", dataType="INT8", amx=True)
    results_bert = {
        "FP32": fp32_bert_inference_time,
        "BF16_with_AMX": bf16_amx_bert_inference_time,
        "INT8_with_VNNI": int8_with_vnni_bert_inference_time,
        "INT8_with_AMX": int8_amx_bert_inference_time
    }
    summarizeResults("BERT", results_bert)

    # Display graphs
    plt.show()

if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')
