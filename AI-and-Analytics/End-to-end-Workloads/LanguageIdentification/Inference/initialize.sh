#!/bin/bash

# Activate the oneAPI environment for PyTorch
source activate pytorch

# Install speechbrain
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
cd ..

# Add speechbrain to environment variable PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/Inference/speechbrain

# Install PyTorch and Intel Extension for PyTorch (IPEX)
pip install torch==1.13.1 torchaudio
pip install --no-deps torchvision==0.14.0
pip install intel_extension_for_pytorch==1.13.100
pip install neural-compressor==2.0

# Update packages
apt-get update && apt-get install libgl1
