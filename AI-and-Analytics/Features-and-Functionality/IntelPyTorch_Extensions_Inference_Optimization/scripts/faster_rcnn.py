import torch
import torchvision
import numpy as np

device="cpu"

def inference(model, data):
  with torch.no_grad():
    warm up
    for _ in range(100):
      model(data)

    # measure
    import time
    start = time.time()
    for _ in range(100):
      output = model(data)
    end = time.time()
    print('Inference took {:.2f} ms in average'.format((end-start)/100*1000))

def main(args):
  data = torch.rand(1, 3, 224, 224)
  
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT, progress=True,
    num_classes=91, weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
  model = model.eval()


  inference(model, data)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'int8'])
  parser.add_argument("--torchscript", default=False, action="store_true")

  main(parser.parse_args())
