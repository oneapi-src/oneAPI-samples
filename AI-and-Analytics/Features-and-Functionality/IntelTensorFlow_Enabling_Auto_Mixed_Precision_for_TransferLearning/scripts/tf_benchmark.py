import os
import sys
import time
import logging
import argparse
import math
import numpy as np
import logging

from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.protobuf import rewriter_config_pb2

try:
    import tensorflow.compat.v1 as tf_v1
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
except ImportError:
    import tensorflow as tf_v1

logging.basicConfig(level=logging.INFO,
                    datefmt='[%H:%M:%S]',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Benchmark")

unlikely_output_types = ['Const', 'Assign', 'NoOp', 'Parameter', 'Assert', 'Postprocessor', 'Batch', 'Preprocessor', 'save', \
    'global_step', 'Conv2d', 'read', 'switch', 'gradient', 'cond', 'train', 'detection_masks', 'detection_classes', 'xmax', 'xmin', 'ymax', 'ymin', \
        'init_op', 'Merge', 'batch', 'SparseToDense', 'init_ops', 'RMSProp', 'transpose', 'ApplyAdam', 'while', 'ReadVariableOp']

def summarize_graph(graph_def, fix_dynamic_shape):
    placeholders = dict(input_nodes_info={})
    outputs = list()
    graph = tf_v1.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
        tf_v1.import_graph_def(graph_def, name='')
    for node in graph.as_graph_def().node:  # pylint: disable=no-member
        if node.op == 'Placeholder':
            node_dict = dict()
            node_dict['node_name'] = node.name
            node_dict['type'] = tf_v1.DType(node.attr['dtype'].type).name
            is_one_dim = False
            if node_dict['type'] != 'bool':
                # convert shape to list
                try:
                    _shape = list(tf_v1.TensorShape(node.attr['shape'].shape))
                    if tf_v1.__version__ >= '2.0.0':
                        node_dict['shape'] = [item if item != None else fix_dynamic_shape for item in _shape]
                    else:
                        node_dict['shape'] = [item.value if item.value != None else fix_dynamic_shape for item in _shape]
                    # if shape dimension > 1, suppose first dimension is batch-size
                    if len(node_dict['shape']) > 1: 
                        node_dict['shape'] = node_dict['shape'][1:]
                    else:
                        node_dict['shape'] = []
                        is_one_dim = True
                except ValueError as e:
                    print(str(e))
                    _shape = [fix_dynamic_shape, fix_dynamic_shape, 3]
                    node_dict['shape'] = _shape

            else:   # deal with bool dtype inputs, now assign bool dtype input False value
                node_dict['shape'] = None
                node_dict['value'] = False
            node_dict['is_one_dim'] = is_one_dim
            print("Find a possible input node: {}".format(node_dict))
            placeholders['input_nodes_info'].update({node.name: node_dict})
            for tf_op in children(node.name, graph):
                if tf_op.type == 'SparseTensorDenseMatMul':
                    a_shape = tensor_util.MakeNdarray(tf_op.inputs[2].op.node_def.attr['value'].tensor)
                    placeholders['sparse_d_shape'] = {tf_op.name: {tf_op.inputs[0].op.name: a_shape}}
                    placeholders['sparse_d_shape'][tf_op.name][tf_op.inputs[1].op.name] = a_shape

        if len(children(node.name, graph)) == 0:
            is_output = True
            if node.op in unlikely_output_types or node.name.split('/')[-1] in unlikely_output_types:
                is_output = False
            for item in node.name.split('/'):
                for unlikely_pattern in unlikely_output_types:
                    if unlikely_pattern in item:
                        is_output = False
            if is_output:
                print("Find a possible output node: '{}'".format(node.name))
                outputs.append(node.name)

    result = dict()
    result['inputs'] = placeholders
    result['outputs'] = outputs
    return result

def children(op_name: str, graph: tf_v1.Graph):
    op = graph.get_operation_by_name(op_name)
    return set(op for out in op.outputs for op in out.consumers())

def initialize_graph(model_details, args, od_graph_def):
    graph = tf_v1.Graph()
    with graph.as_default():
      input_variables = {
        in_name + ":0": tf_v1.Variable(val)
        for in_name, val in model_details['input'].items()}

      if not od_graph_def.node:
        from neural_compressor.experimental import common
        model = common.Model(os.path.join(os.getcwd(), model_details['model_dir']))
        od_graph_def = model.graph_def

      # optimize for inference
      if not args.disable_optimize:
        # optimize graph for inference
        try:
          input_list = [ in_name for in_name,val in model_details['input'].items() ]
          output_list = [ out_name for out_name in model_details['output'] ]
          input_data_type = [ tf_v1.convert_to_tensor(item).dtype.as_datatype_enum for item in model_details['input'].values() ]

          od_graph_def_tmp = od_graph_def
          od_graph_def = optimize_for_inference_lib.optimize_for_inference(
              od_graph_def,  # inputGraph,
              input_list,  # an array of the input nodes
              output_list,  # an array of output nodes
              input_data_type)
          od_graph_def.library.CopyFrom(od_graph_def_tmp.library)
          print("---- Optimize for inference!")
        except:
          print("---- Optimize for inference failed!")

      tf_v1.import_graph_def(od_graph_def, name='g',
                          input_map=input_variables)

    return graph

def create_tf_config(args):
    if "OMP_NUM_THREADS" in os.environ:
        OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    else:
        OMP_NUM_THREADS = len(os.sched_getaffinity(0))

    config = tf_v1.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = OMP_NUM_THREADS
    config.inter_op_parallelism_threads = 1
    # additional options
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
    config.graph_options.rewrite_options.function_optimization = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
    config.graph_options.rewrite_options.cpu_layout_conversion = rewriter_config_pb2.RewriterConfig.NCHW_TO_NHWC
    if args.precision == 'bfloat16':
        # config.graph_options.rewrite_options.auto_mixed_precision_onednn_bfloat16 = rewriter_config_pb2.RewriterConfig.ON
        config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
        config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON
    return config

def run_benchmark(model_details, args, find_graph_def):
    tf_config = create_tf_config(args)
    graph = initialize_graph(model_details, args, find_graph_def)
    run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
    run_metadata = tf_v1.RunMetadata()

    if args.save_graph:
        # write the real benchmark graph to local
        model_dir = os.path.dirname(os.path.abspath(model_detail['model_dir']))
        out_graph_file = os.path.join(model_dir, 'runtime_graph.pb')
        write_graph(graph.as_graph_def(), out_graph_file)
        print("********** save runtime graph at {}".format(out_graph_file))
 
    with tf_v1.Session(config=tf_config, graph=graph) as sess:
        input_dict = {graph.get_tensor_by_name("g/" + out_name + ":0"): model_details['input'][out_name]
                       for out_name in model_details['input']}
        output_dict = {out_name: graph.get_tensor_by_name("g/" + out_name + ":0")
                       for out_name in model_details['output']}

        sess.run(tf_v1.global_variables_initializer())

        total_time = 0.0
        reps_done = 0
        for rep in range(args.num_iter):
            # sess run
            start = time.time()
            if args.profile:
                _ = sess.run(output_dict, options=run_options, run_metadata=run_metadata)
            else:
                _ = sess.run(output_dict, feed_dict=input_dict)
            end = time.time()
            delta = end - start
            print("Iteration: {}, inference time: {} sec".format(rep, delta), flush=True)

            if rep >= args.num_warmup and rep < (args.num_iter - args.num_warmup):
                total_time += delta
                reps_done += 1

        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * args.batch_size
        print('Batch size = %d' % args.batch_size)
        print("Latency: {:.3f} ms".format(latency))
        print("Throughput: {:.2f} fps".format(throughput))

        # Logging to a file
        log_file = open("log.txt", "a")
        log_file.write("Throughput: " + str(throughput) + "\n")


def get_input_output(graph_path, args):
  # give a fix shape if not get input shape 
  fix_dynamic_shape = 300

  from neural_compressor.experimental import common
  model = common.Model(graph_path)        
  if args.output_name in [[], ['']]:
      raise AttributeError("Empty '--output_name', please specify a valid '--output_name'.")
  elif args.output_name is not None:
      model.output_tensor_names = args.output_name
      graph_def = model.graph_def
      output_nodes = summarize_graph(graph_def, fix_dynamic_shape)
      output_nodes['outputs'] = args.output_name
  else:
      graph_def = model.graph_def
      output_nodes = summarize_graph(graph_def, fix_dynamic_shape)

  return graph_def, output_nodes

def generate_data(input_shape, input_dtype="float32",
                  batch_size=1, max_int_value=35,
                  newaxis=True, is_one_dim=False):
    np.random.seed(1024)
    if input_dtype in ["uint8", "int8", "int32", "int64"]:
        if is_one_dim:
            return np.random.randint(1, max_int_value, batch_size).astype(input_dtype)
        dummy_input = np.random.randint(1, size=input_shape).astype(input_dtype)
    else:
        if is_one_dim:
            return np.random.randn(batch_size).astype(input_dtype)
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)
    # handle the case that the shape of the input is one-dimensional
    if newaxis == False:
        return np.repeat(dummy_input, batch_size, axis=0)
    return np.repeat(dummy_input[np.newaxis, :], batch_size, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-m", "--model_name", help="name of model")
    parser.add_argument("-pb", "--model_path", help="path of model")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--precision", type=str, default='float32', help="float32, int8 or bfloat16")
    parser.add_argument("-i", "-n", "--num_iter", type=int, default=100, help="numbers of inference iteration, default is 500")
    parser.add_argument("-w", "--num_warmup", type=int, default=20, help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_true', help="use this to disable optimize_for_inference")
    parser.add_argument("--profile", action='store_true', help="profile")
    parser.add_argument("--is_meta", action='store_true', help="input a meta file")
    parser.add_argument("--save_graph", action='store_true', help="save_graph")
    parser.add_argument("--output_name", nargs='*', help="Specify output for neural_compressor ckpt.")

    # args
    args = parser.parse_args()

    # benchmark PB model directly
    find_graph_def = tf_v1.GraphDef()
    if args.model_path and not args.model_name:
        # generate model detail
        model_dir = args.model_path
        model_detail = {}
        find_graph_def, model_input_output = get_input_output(model_dir, args)
        output = model_input_output['outputs']
        input_dic = {}
        input_nodes_info = model_input_output['inputs']['input_nodes_info']
        for _input in input_nodes_info:
            # deal with bool dtype input
            if input_nodes_info[_input]['type'] == 'bool':
                input_dic[_input] = input_nodes_info[_input]['value']
            elif _input == 'dropout_keep_prob':
                input_dic[_input] = np.array([0.5,], dtype='float32')
            else:
                dtype = input_nodes_info[_input]['type']
                dshape = input_nodes_info[_input]['shape']
                is_one_dim = input_nodes_info[_input]['is_one_dim']
                dummy_input = generate_data(dshape, dtype, args.batch_size, is_one_dim=is_one_dim)
                input_dic[_input] = dummy_input
        model_detail['model_dir'] = model_dir
        model_detail['input'] = input_dic
        model_detail['output'] = output
        model_detail['ckpt'] = args.is_meta

    inputs_shape = []
    inputs_dtype = []
    for input_tensor in model_detail['input'].values():
        if not isinstance(input_tensor, bool):
            inputs_shape.append(input_tensor.shape)
            inputs_dtype.append(str(input_tensor.dtype))
        else:
            # TODO: wait scalar support in dummy dataset
            inputs_shape.append((1,))
            inputs_dtype.append('bool')
    logger.info("Final benchmark input nodes: name_list={}, shape_list={}, dtype_list={}".format( \
                list(model_detail['input'].keys()), inputs_shape, inputs_dtype))
    logger.info("Final benchmark output nodes: name_list={}".format(model_detail['output']))

    run_benchmark(model_detail, args, find_graph_def)