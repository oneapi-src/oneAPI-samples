#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import intel_extension_for_pytorch as ipex

'''
BS_TRAIN: Batch size for training data
BS_TEST:  Batch size for testing data
EPOCHNUM: Number of epoch for training
'''
BS_TRAIN = 50
BS_TEST  = 10
EPOCHNUM = 2

'''
TestDataset class is inherited from torch.utils.data.Dataset.
Since data for training involves data and ground truth, a flag "train" is defined in the initialization function. When train is True, instance of TestDataset gets a pair of training data and label data. When it is False, the instance gets data only for inference. Value of the flag "train" is set in __init__ function.
In __getitem__ function, data at index position is supposed to be returned.
__len__ function returns the overall length of the dataset.
'''
class TestDataset(Dataset):
    def __init__(self, train = True):
        super(TestDataset, self).__init__()
        self.train = train

    def __getitem__(self, index):
        if self.train:
            return torch.rand(3, 112, 112), torch.rand(6, 110, 110)
        else:
            return torch.rand(3, 112, 112)

    def __len__(self):
        if self.train:
            return 100
        else:
            return 20

'''
TestModel class is inherited from torch.nn.Module.
Operations that will be used in the topology are defined in __init__ function.
Input data x is supposed to be passed to the forward function. The topology is implemented in the forward function. When perform training/inference, the forward function will be called automatically by passing input data to a model instance.
'''
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 6, 3)
        self.norm = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

'''
Perform training and inference in main function
'''
def main():
    '''
    The following 3 components are required to perform training.
    1. model: Instantiate model class
    2. optim: Optimization function for update topology parameters during training
    3. crite: Criterion function to minimize loss
    '''
    model = TestModel()
    model = model.to(memory_format=torch.channels_last)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    crite = nn.MSELoss(reduction='sum')

    '''
    1. Instantiate the Dataset class defined before
    2. Use torch.utils.data.DataLoader to load data from the Dataset instance
    '''
    train_data  = TestDataset()
    trainLoader = DataLoader(train_data, batch_size=BS_TRAIN)
    test_data   = TestDataset(train=False)
    testLoader  = DataLoader(test_data, batch_size=BS_TEST)

    '''
    Apply Intel Extension for PyTorch optimization against the model object and optimizer object.
    '''
    model, optim = ipex.optimize(model, optimizer=optim)

    '''
    Perform training and inference
    Use model.train() to set the model into train mode. Use model.eval() to set the model into inference mode.
    Use for loop with enumerate(instance of DataLoader) to go through the whole dataset for training/inference.
    '''
    for i in range(0, EPOCHNUM - 1):
        '''
        Iterate dataset for training to train the model
        '''
        model.train()
        for batch_index, (data, y_ans) in enumerate(trainLoader):
            data = data.to(memory_format=torch.channels_last)
            optim.zero_grad()
            y = model(data)
            loss = crite(y, y_ans)
            loss.backward()
            optim.step()

        '''
        Iterate dataset for validation to evaluate the model
        '''
        model.eval()
        for batch_index, data in enumerate(testLoader):
            data = data.to(memory_format=torch.channels_last)
            y = model(data)

if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')
