import tensorflow as tf
import mobilenet_v1 as mn

MOBILENET_MIN_DEPTH = 8
MOBILENET_DEP_MULTI = 0.25
IMAGE_SIZE = 192
IMAGE_CHANNELS = 3

DEF_SHAPE = (None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

def load_graph(name, frozen_graph_filename):
  # We load the protobuf file from the disk and parse it to retrieve the
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we can use again a convenient built-in function to import a graph_def into the
  # current default Graph
  tf.import_graph_def(
    graph_def,
    input_map=None,
    return_elements=None,
    name=name,
    op_dict=None,
    producer_op_list=None
  )

# Loads a mobilenet up to a layer and returns the inputs and outputs
# name: Name you want to give to the graph
# graph_filename: The filepath to the frozen mobilenet
# layer_cutoff: The layer to take the output of the frozen network from
def load_mobilenet(name, graph_filename, layer_cutoff):
  layer = "MobilenetV1/MobilenetV1/Conv2d_%i_pointwise/Relu:0" % layer_cutoff
  load_graph(name, graph_filename)
  scope = tf.contrib.framework.get_name_scope()
  inputs = tf.get_default_graph().get_tensor_by_name("%s/%s/inputs:0" % (scope, name))
  output = tf.get_default_graph().get_tensor_by_name("%s/%s/%s" % (scope, name, layer))
  return inputs, output

# Loads a mobilenet up to a layer and returns the inputs and outputs
# name: Name you want to give to the graph
# graph_filename: The filepath to the frozen mobilenet
# layer_cutoff: The layer to take the output of the frozen network from
# end_num: The last conv layer
def load_mobilenet_and_continue(name, graph_filename, layer_cutoff, end_num=13):
  inputs, output = load_mobilenet(name, graph_filename, layer_cutoff)
  conv_defs = mn._CONV_DEFS[layer_cutoff+1:end_num+1]
  mn_output, _ = mn.mobilenet_v1_base(output,
      final_endpoint='Conv2d_%i_pointwise' % (end_num-layer_cutoff-1),
      min_depth=MOBILENET_MIN_DEPTH, depth_multiplier=MOBILENET_DEP_MULTI,
      conv_defs=conv_defs, output_stride=None, scope=None)
  return inputs, mn_output
