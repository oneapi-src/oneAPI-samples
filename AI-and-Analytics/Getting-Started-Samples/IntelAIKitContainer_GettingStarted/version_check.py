#import importlib
from importlib import util
tensorflow_found = util.find_spec("tensorflow") is not None
pytorch_found = util.find_spec("torch") is not None
pytorch_ext_found = util.find_spec("intel_pytorch_extension") is not None

if tensorflow_found == True:

    import tensorflow as tf

    import os

    def get_mkl_enabled_flag():

        mkl_enabled = False
        major_version = int(tf.__version__.split(".")[0])
        minor_version = int(tf.__version__.split(".")[1])
        if major_version >= 2:
            if minor_version < 5:
                from tensorflow.python import _pywrap_util_port
            else:
                from tensorflow.python.util import _pywrap_util_port
                onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
            mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)
        else:
            mkl_enabled = tf.pywrap_tensorflow.IsMklEnabled()
        return mkl_enabled

    print ("We are using Tensorflow version", tf.__version__)
    print("MKL enabled :", get_mkl_enabled_flag())

if pytorch_found == True:
    import torch
    print(torch.__version__)
    mkldnn_enabled = torch.backends.mkldnn.is_available()
    mkl_enabled = torch.backends.mkl.is_available()
    openmp_enabled = torch.backends.openmp.is_available()
    print('mkldnn : {0},  mkl : {1}, openmp : {2}'.format(mkldnn_enabled, mkl_enabled, openmp_enabled))
    print(torch.__config__.show())
    
    if pytorch_ext_found == True:
        import intel_pytorch_extension as ipex
        print("ipex_verion : ",ipex.__version__)
