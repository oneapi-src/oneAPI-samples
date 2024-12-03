#!/bin/bash

# Install speechbrain
git clone --depth 1 --branch v1.0.2 https://github.com/speechbrain/speechbrain.git
cd speechbrain
python -m pip install -r requirements.txt
python -m pip install --editable .
cd ..

# Add speechbrain to environment variable PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/speechbrain

# Install huggingface datasets and other requirements
conda install -y datasets tqdm librosa jupyter ipykernel ipywidgets

# Install webdataset
python -m pip install webdataset==0.2.100

# Install libraries for MP3 to WAV conversion
python -m pip install pydub

# Install notebook to run Jupyter notebooks
python -m pip install notebook
