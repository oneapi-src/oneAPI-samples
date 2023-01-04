#!/bin/bash

# Activate the oneAPI environment for PyTorch
source activate pytorch

# Install speechbrain
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
cd ..

# Install PyTorch and Intel Extension for PyTorch (IPEX)
pip install torch==1.12.0 torchaudio==0.12.0 torchvision==0.13.0
pip install intel_extension_for_pytorch==1.12.0
pip install neural-compressor==1.14.2

# Update packages
apt-get update && apt-get install libgl1
