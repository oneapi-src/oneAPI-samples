#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2022 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''

import os
from time import time
import matplotlib.pyplot as plt
import torch
import torchvision
import intel_extension_for_pytorch as ipex

# Hyperparameters and constants
LR = 0.001
MOMENTUM = 0.9
DOWNLOAD = True
DATA = 'datasets/cifar10/'

os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"

"""
Function to run a test case
"""
def trainModel(train_loader, modelName="myModel", dataType="fp32"):
    """
    Input parameters
        train_loader: a torch DataLoader object containing the training data
        modelName: a string representing the name of the model
        dataType: the data type for model parameters, supported values - fp32, bf16
    Return value
        training_time: the time in seconds it takes to train the model
    """

    # Initialize the model 
    model = torchvision.models.resnet50()
    model = model.to(memory_format=torch.channels_last)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model.train()
    
    # Optimize with BF16 or FP32 (default)
    if "bf16" == dataType:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
    else:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Train the model
    num_batches = len(train_loader)
    start_time = time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if "bf16" == dataType:
            with torch.cpu.amp.autocast():   # Auto Mixed Precision
                # Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
                data = data.to(memory_format=torch.channels_last)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
        else:
            # Setting memory_format to torch.channels_last could improve performance with 4D input data. This is optional.
            data = data.to(memory_format=torch.channels_last)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
        optimizer.step()
        if 0 == (batch_idx+1) % 50:
            print("Batch %d/%d complete" %(batch_idx+1, num_batches))
    end_time = time()
    training_time = end_time-start_time
    print("Training took %.3f seconds" %(training_time))
    
    # Save a checkpoint of the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint_%s.pth' %modelName)
        
    return training_time

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
    train_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=True,
            transform=transform,
            download=DOWNLOAD,
    )
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=128
    )

    # Train models and acquire training times
    print("Training model with BF16 with AVX512")
    bf16_noAmx_training_time = trainModel(train_loader, modelName="bf16_noAmx", dataType="bf16")

if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')
