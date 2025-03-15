source $HOME/intel/oneapi/intelpython/bin/activate
conda activate jax
git clone https://github.com/google/jax.git 
cp -r jax/examples .
export PYTHONPATH=$PYTHONPATH:$(pwd)
python examples/spmd_mnist_classifier_fromscratch.py 
