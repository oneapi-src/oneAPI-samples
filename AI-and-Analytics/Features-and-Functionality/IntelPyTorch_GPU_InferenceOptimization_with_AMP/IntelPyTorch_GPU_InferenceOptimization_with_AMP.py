import os
from time import time
from tqdm import tqdm 
import numpy as np
import torch
import torchvision
import intel_extension_for_pytorch as ipex

import matplotlib.pyplot as plt


# Hyperparameters and constants
LR = 0.01
MOMENTUM = 0.9
DATA = 'datasets/cifar10/'
epochs = 1
batch_size=128

#TO check if IPEX_XPU is correctly installed and can be used for PyTorch model 
try:
  device = "xpu" if torch.xpu.is_available() else "cpu"
  
except:
  device="cpu"  

if device == "xpu": # XPU is for Intel dGPU
  print("IPEX_XPU is present and Intel GPU is available to use for PyTorch")
  device = "gpu"
else:
  print("using CPU device for PyTorch")


"""
Function to run a test case
"""
def trainModel(train_loader, modelName="myModel", device="cpu", dataType="fp32"):
    """
    Input parameters
        train_loader: a torch DataLoader object containing the training data with images and labels
        modelName: a string representing the name of the model
        device: the device to use - cpu or gpu
        dataType: the data type for model parameters, supported values - fp32, bf16
    Return value
        training_time: the time in seconds it takes to train the model
    """

    # Initialize the model 
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048,10)
    lin_layer = model.fc
    new_layer = torch.nn.Sequential(
        lin_layer,
        torch.nn.Softmax(dim=1)
    )
    model.fc = new_layer

    #Define loss function and optimization methodology
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    model.train()

    #export model and criterian to XPU device. GPU specific code
    if device == "gpu":
        model = model.to("xpu:0") ## GPU 
        criterion = criterion.to("xpu:0") 

    #Optimize with BF16 or FP32(default) . BF16 specific code
    if "bf16" == dataType:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
    else:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

    #Train the model
    num_batches = len(train_loader) * epochs
    

    for i in range(epochs):
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Export data to XPU device. GPU specific code
            if device == "gpu":
                data = data.to("xpu:0")
                target = target.to("xpu:0")

            # Apply Auto-mixed precision(BF16)  
            if "bf16" == dataType:
                with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


            # Showing Average loss after 50 batches
            if 0 == (batch_idx+1) % 50:
                print("Batch %d/%d complete" %(batch_idx+1, num_batches))
                print(f' average loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    # Save a checkpoint of the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint_%s.pth' %modelName)

    return None



#Dataloader operations
transform = torchvision.transforms.Compose([
torchvision.transforms.Resize((224, 224)),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train = True,
        transform=transform,
        download=True,
)
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size
)

test_dataset = torchvision.datasets.CIFAR10(root=DATA, train = False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size )



#Model Training

if device=='gpu':
  print("Training model with FP32 on GPU, will be saved as checkpoint_gpu_rn50.pth")
  trainModel(train_loader, modelName="gpu_rn50", device="gpu", dataType="fp32")
else:
  print("Training model with FP32 on CPU, will be saved as checkpoint_cpu_rn50.pth")
  trainModel(train_loader, modelName="cpu_rn50", device="cpu", dataType="fp32")  



#Model Evaluation

#Load model from the saved model file
def load_model(cp_file = 'checkpoint_cpu_rn50.pth'):
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048,10)
    lin_layer = model.fc
    new_layer = torch.nn.Sequential(
        lin_layer,
        torch.nn.Softmax(dim=1)
    )
    model.fc = new_layer

    checkpoint = torch.load(cp_file)
    model.load_state_dict(checkpoint['model_state_dict']) 
    return model




#Applying torchscript and IPEX optimizations(Optional)
def ipex_jit_optimize(model, dataType = "fp32" , device="cpu"):
    model.eval()
    
    if device=="gpu": #export model to xpu device
        model = model.to("xpu:0")
    
    if dataType=="bf16": # for bfloat16  
        model = ipex.optimize(model, dtype=torch.bfloat16)
    else:
        model = ipex.optimize(model, dtype=torch.float32)
            
    with torch.no_grad():
        d = torch.rand(1, 3, 224, 224)
        if device=="gpu": 
            d = d.to("xpu:0")
          
        #export model to Torchscript mode    
        if dataType=="bf16": 
          with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16): 
            jit_model = torch.jit.trace(model, d) # JIT trace the optimized model
            jit_model = torch.jit.freeze(jit_model) # JIT freeze the traced model
        else:
          jit_model = torch.jit.trace(model, d) # JIT trace the optimized model
          jit_model = torch.jit.freeze(jit_model) # JIT freeze the traced model              
    return jit_model





def inferModel(model, test_loader, device="cpu" , dataType='fp32'):
    correct = 0
    total = 0
    if device == "gpu":
        model = model.to("xpu:0")
    infer_time = 0

    with torch.no_grad():
        #Warm up rounds of 3 batches
        num_batches = len(test_loader)
        batches=0
                   
        for i, data in tqdm(enumerate(test_loader)):
            
            # Record time for Inference
            if device=='gpu':
              torch.xpu.synchronize()
            start_time = time()
            images, labels = data
            if device =="gpu":
                images = images.to("xpu:0")
                 
            outputs = model(images)
            outputs = outputs.to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()        
            
            # Record time after finishing batch inference
            if device=='gpu':
              torch.xpu.synchronize()
            end_time = time()      

            if i>=3 and i<=num_batches-3:
                infer_time += (end_time-start_time)
                batches += 1
            #Skip last few batches     
            if i == num_batches - 3:
                break    

    accuracy = 100 * correct / total
    return accuracy, infer_time*1000/(batches*batch_size)



#Evaluation of different models
def Eval_model(cp_file = 'checkpoint_model.pth', dataType = "fp32" , device="gpu" ):
    model = load_model(cp_file)
    model = ipex_jit_optimize(model, dataType , device)
    accuracy, bt = inferModel(model, test_loader, device, dataType )
    print(f' Model accuracy: {accuracy} and Average Inference latency: {bt} \n'  )
    return accuracy, bt



#Accuracy and Inference time check

if device == 'cpu': #For FP32 model on CPU
  print("Model evaluation with FP32 on CPU")
  Eval_model(cp_file = 'checkpoint_cpu_rn50.pth', dataType = "fp32" , device=device)
else:    
  #For FP32 model on GPU
  print("Model evaluation with FP32 on GPU")
  acc_fp32, fp32_avg_latency = Eval_model(cp_file = 'checkpoint_gpu_rn50.pth', dataType = "fp32" , device=device)
  
  #For BF16 model on GPU
  print("Model evaluation with BF16 on GPU")
  acc_bf16, bf16_avg_latency = Eval_model(cp_file = 'checkpoint_gpu_rn50.pth', dataType = "bf16" , device=device)
  
  #Summary 
  print("Summary")
  print(f'Inference average latecy for FP32  on GPU is:  {fp32_avg_latency} ')
  print(f'Inference average latency for AMP BF16 on GPU is:  {bf16_avg_latency} ')
  
  speedup_from_amp_bf16 = fp32_avg_latency / bf16_avg_latency
  print("Inference with BF16 is %.2fX faster than FP32 on GPU" %speedup_from_amp_bf16)
  
  
  plt.figure()
  plt.title("ResNet50 Inference Latency Comparison")
  plt.xlabel("Test Case")
  plt.ylabel("Inference Latency per sample(ms)")
  plt.bar(["FP32 on GPU", "AMP BF16 on GPU"], [fp32_avg_latency, bf16_avg_latency])
  plt.savefig('./bf16speedup.png')
  
  plt.figure()
  plt.title("Accuracy Comparison")
  plt.xlabel("Test Case")
  plt.ylabel("Accuracy(%)")
  plt.bar(["FP32 on GPU", "AMP BF16 on GPU"], [acc_fp32, acc_bf16])
  print(f'Accuracy drop with AMP BF16 is: {acc_fp32-acc_bf16}')
  plt.savefig('./accuracy.png')

print('[CODE_SAMPLE_COMPLETED_SUCCESFULLY]')


