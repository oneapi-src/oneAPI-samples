import os


def get_device_selector(is_gpu):
    try:    
        from dpctl import device_context, device_type
        with device_context(device_type.gpu, 0):
            is_gpu = True
    except:
        is_gpu = False
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if (
        os.environ.get("SYCL_DEVICE_FILTER") is None
        or os.environ.get("SYCL_DEVICE_FILTER") == "opencl"
    ):
        return "opencl:" + device_selector

    if os.environ.get("SYCL_DEVICE_FILTER") == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get("SYCL_DEVICE_FILTER")