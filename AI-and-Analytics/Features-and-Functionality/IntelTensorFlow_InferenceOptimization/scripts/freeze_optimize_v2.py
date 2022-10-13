import os
import argparse
import sys

from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import utils as model_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.framework import graph_util
from tensorflow.python.util import nest

from argparse import ArgumentParser

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver as saver_lib

import tensorflow as tf


FLAGS = None
def run_main(unused_args):

  input_model_dir = FLAGS.input_saved_model_dir
  output_model_dir = FLAGS.output_saved_model_dir
  sig_key = FLAGS.signature_key
  inp_tags = FLAGS.saved_model_tags
  if FLAGS.saved_model_tags == "":
    tag_set = []
  else:
    tag_set = [tag for tag in inp_tags.split(",")]
    avail_tags = saved_model_utils.get_saved_model_tag_sets(input_model_dir)
    found = False
    for tag in tag_set:
      if [tag] in avail_tags:
        found = True
      else:
        found = False
        break
    if not found:
      print ("Supplied tags", tag_set, "is not in available tag set,\
                    please use one or more of these", avail_tags, "Using --saved_model_tags")
      exit(1)


  sig_def = saved_model_utils.get_meta_graph_def(input_model_dir, inp_tags)
  pretrained_model = load.load(input_model_dir, tag_set)
  if sig_key not in list(pretrained_model.signatures.keys()):
    print (sig_key, "is not in ", list(pretrained_model.signatures.keys()),
            "provide one of those using --signature_key")
    exit(1)

  infer = pretrained_model.signatures[sig_key]
  frozen_func = convert_to_constants.convert_variables_to_constants_v2(infer,lower_control_flow=True)

  frozen_func.graph.structured_outputs = nest.pack_sequence_as(
        infer.graph.structured_outputs,
        frozen_func.graph.structured_outputs)
  souts = frozen_func.graph.structured_outputs
  inputs = frozen_func.inputs
  input_nodes = [(tensor.name.split(":"))[0] for tensor in inputs]
  output_nodes = [(souts[name].name.split(":"))[0] for name in souts]

  g_def = frozen_func.graph.as_graph_def()
  
  # Add CPU device to the nodes.
  for node in g_def.node:
    node.device = "/device:CPU:0"

  g = tf.Graph()
  with g.as_default():
    importer.import_graph_def(g_def, input_map={}, name="")
    meta_graph = saver_lib.export_meta_graph(graph_def=g_def, graph=g)

    fetch_collection = meta_graph_pb2.CollectionDef()
    for fetch in output_nodes:
      fetch_collection.node_list.value.append(fetch)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  config = config_pb2.ConfigProto()

  # Constant folding grappler pass
  config.graph_options.rewrite_options.CopyFrom(
    rewriter_config_pb2.RewriterConfig(
      remapping=rewriter_config_pb2.RewriterConfig.OFF,
      constant_folding=rewriter_config_pb2.RewriterConfig.AGGRESSIVE))

  optimized_graph = tf_optimizer.OptimizeGraph(
      config, meta_graph)

  opt_graph = optimize_for_inference_lib.optimize_for_inference(optimized_graph, input_nodes, output_nodes,
           [tensor.dtype.as_datatype_enum for tensor in inputs] )

  with session.Session() as sess:
    graph = importer.import_graph_def(opt_graph,name="")

    signature_inputs = {(tensor.name.split(":"))[0]: model_utils.build_tensor_info(tensor)
                        for tensor in inputs}
    signature_outputs = {name: model_utils.build_tensor_info(souts[name])
                         for name in souts}
    signature_def = signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        signature_constants.PREDICT_METHOD_NAME)
    signature_def_map = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        }
    builder = saved_model_builder.SavedModelBuilder(output_model_dir)
    builder.add_meta_graph_and_variables(sess, tags=[tag_constants.SERVING],
            signature_def_map=signature_def_map)
    builder.save()

def parse_args():
  """Main function of freeze_graph."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_saved_model_dir",
      type=str,
      default="",
      help="Path to the dir input TensorFlow \'SavedModel\'.")
  parser.add_argument(
      "--output_saved_model_dir",
      type=str,
      default="",
      help="Path to the dir output TensorFlow \'SavedModel\'.")
  parser.add_argument(
      "--signature_key",
      type=str,
      default="serving_default",
      help="Path to the dir output TensorFlow \'SavedModel\'.")
  parser.add_argument(
      "--saved_model_tags",
      type=str,
      default="serve",
      help="""\
      Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in.\
      """)
  return parser.parse_known_args()


if __name__ == "__main__":
  FLAGS, unparsed = parse_args()
  run_main([sys.argv[0]] + unparsed)
