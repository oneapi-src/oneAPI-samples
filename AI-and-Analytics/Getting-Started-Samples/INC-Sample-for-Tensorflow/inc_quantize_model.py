import sys

try:
    import neural_compressor as inc
    print("neural_compressor version {}".format(inc.__version__))
    from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion, TuningCriterion
    from neural_compressor.data import DataLoader
    from neural_compressor.quantization import fit
    from neural_compressor import Metric
except:
    try:
        import lpot as inc
        print("LPOT version {}".format(inc.__version__))
    except:
        import ilit as inc
        print("iLiT version {}".format(inc.__version__))

if inc.__version__ == '1.2':
    print("This script doesn't support LPOT 1.2, please install LPOT 1.1, 1.2.1 or newer")
    sys.exit(1)

import mnist_dataset


class Dataset(object):
    def __init__(self):
        _x_train, _y_train, label_train, x_test, y_test, label_test = mnist_dataset.read_data()

        self.test_images = x_test
        self.labels = label_test

    def __getitem__(self, index):
        return self.test_images[index], self.labels[index]

    def __len__(self):
        return len(self.test_images)

def ver2int(ver):
    s_vers = ver.split(".")
    res = 0
    for i, s in enumerate(s_vers):
        res += int(s)*(100**(2-i))

    return res

def compare_ver(src, dst):
    src_ver = ver2int(src)
    dst_ver = ver2int(dst)
    if src_ver>dst_ver:
        return 1
    if src_ver<dst_ver:
        return -1
    return 0

def auto_tune(input_graph_path, batch_size):
    dataset = Dataset()
    dataloader = DataLoader(framework='tensorflow', dataset=dataset, batch_size=batch_size)
    tuning_criterion = TuningCriterion(max_trials=100)
    config = PostTrainingQuantConfig(approach="static", tuning_criterion=tuning_criterion,
                                     accuracy_criterion = AccuracyCriterion(
                                         higher_is_better=True,
                                         criterion='relative',
                                         tolerable_loss=0.01  )
                                    )
    top1 = Metric(name="topk", k=1)

    q_model = fit(
        model=input_graph_path,
        conf=config,
        calib_dataloader=dataloader,
        eval_dataloader=dataloader,
        eval_metric=top1
        )

    return q_model


batch_size = 200
fp32_frozen_pb_file = "fp32_frozen.pb"
int8_pb_file = "alexnet_int8_model.pb"

q_model = auto_tune(fp32_frozen_pb_file, batch_size)
q_model.save(int8_pb_file)