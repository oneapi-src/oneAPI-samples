import os
import torch


# helper function to get model size
def get_model_size_ram(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb


def get_model_size_disk(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    # print('Size (MB):', size_mb)
    os.remove("temp.p")
    return size_mb
