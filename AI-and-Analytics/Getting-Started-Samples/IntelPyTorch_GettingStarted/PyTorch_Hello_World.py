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
from torch.utils import mkldnn
from torch.utils.data import Dataset, DataLoader

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
    Perform training and inference
    Use model.train() to set the model into train mode. Use model.eval() to set the model into inference mode.
    Use for loop with enumerate(instance of DataLoader) to go through the whole dataset for training/inference.
    '''
    for i in range(0, EPOCHNUM - 1):
        model.train()
        for batch_index, (data, y_ans) in enumerate(trainLoader):
            '''
            1. Clear parameters of optimization function
            2. Do forward-propagation
            3. Calculate loss of the forward-propagation with the criterion function
            4. Calculate gradients with the backward() function
            5. Update parameters of the model with the optimization function
            '''
            optim.zero_grad()
            y = model(data)
            loss = crite(y, y_ans)
            loss.backward()
            optim.step()

        model.eval()
        '''
        1. User is suggested to use JIT mode to get best performance with DNNL with minimum change of Pytorch code. User may need to pass an explicit flag or invoke a specific DNNL optimization pass. The PyTorch DNNL JIT backend is under development (RFC link https://github.com/pytorch/pytorch/issues/23657), so the example below is given in imperative mode.
        2. To have model accelerated by DNNL under imperative mode, user needs to explicitly insert format conversion for DNNL operations using tensor.to_mkldnn() and to_dense(). For best result, user needs to insert the format conversion on the boundary of a sequence of DNNL operations. This could boost performance significantly.
        3. For inference task, user needs to prepack the model’s weight using mkldnn_utils.to_mkldnn(model) to save the weight format conversion overhead. It could bring good performance gain sometime for single batch inference.
        '''
        model_mkldnn = mkldnn.to_mkldnn(model)
        for batch_index, data in enumerate(testLoader):
            y = model_mkldnn(data.to_mkldnn())

if __name__ == '__main__':
    main()
    print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')
