import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))

import alexnet
import math
import yaml
import mnist_dataset
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion 


def save_int8_frezon_pb(q_model, path):
    from tensorflow.python.platform import gfile
    f = gfile.GFile(path, 'wb')
    f.write(q_model.graph.as_graph_def().SerializeToString())
    print("Save to {}".format(path))


class Dataloader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        x_train, y_train, label_train, x_test, y_test, label_test = mnist_dataset.read_data()
        batch_nums = math.ceil(len(x_test) / self.batch_size)

        for i in range(batch_nums - 1):
            begin = i * self.batch_size
            end = (i + 1) * self.batch_size
            yield x_test[begin: end], label_test[begin: end]

        begin = (batch_nums - 1) * self.batch_size
        yield x_test[begin:], label_test[begin:]


def auto_tune(input_graph_path, config, batch_size):    
    fp32_graph = alexnet.load_pb(input_graph_path)
    dataloader = Dataloader(batch_size)
    assert(dataloader)
    
    tuning_criterion = TuningCriterion(**config["tuning_criterion"])
    accuracy_criterion = AccuracyCriterion(**config["accuracy_criterion"])
    q_model = fit(
            model=input_graph_path,
            conf=PostTrainingQuantConfig(**config["quant_config"],
                        tuning_criterion=tuning_criterion,
                        accuracy_criterion=accuracy_criterion,
                ),
            calib_dataloader=dataloader,
        )
    return q_model


batch_size = 200
fp32_frezon_pb_file = "fp32_frezon.pb"
int8_pb_file = "alexnet_int8_model.pb"

with open("quant_config.yaml") as f:
    config = yaml.safe_load(f.read())
config

q_model = auto_tune(fp32_frezon_pb_file, config, batch_size)
save_int8_frezon_pb(q_model, int8_pb_file)
