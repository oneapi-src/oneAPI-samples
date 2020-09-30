import ilit
import alexnet
import math
import mnist_dataset


def save_int8_frezon_pb(q_model, path):
    from tensorflow.python.platform import gfile
    f = gfile.GFile(path, 'wb')
    f.write(q_model.as_graph_def().SerializeToString())
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


def auto_tune(input_graph_path, yaml_config, batch_size):
    fp32_graph = alexnet.load_pb(input_graph_path)
    tuner = ilit.Tuner(yaml_config)
    dataloader = Dataloader(batch_size)

    q_model = tuner.tune(
        fp32_graph,
        q_dataloader=dataloader,
        eval_func=None,
        eval_dataloader=dataloader)
    return q_model


yaml_file = "alexnet.yaml"
batch_size = 200
fp32_frezon_pb_file = "fp32_frezon.pb"
int8_pb_file = "alexnet_int8_model.pb"

q_model = auto_tune(fp32_frezon_pb_file, yaml_file, batch_size)
save_int8_frezon_pb(q_model, int8_pb_file)