import mobilenet_v1 as mn
import tensorflow as tf
import model_util

def places(graph_path, cutoff):
  layer = "MobilenetV1/MobilenetV1/Conv2d_%i_pointwise/Relu:0" % cutoff
  graph_name = "places"
  model_util.load_graph(graph_name+"0", graph_path)
  model_util.load_graph(graph_name+"1", graph_path)
  with tf.variable_scope("reused", reuse=False) as scope:
    r_inputs = tf.get_default_graph().get_tensor_by_name(graph_name + "0/inputs:0")
    r_output = tf.contrib.layers.flatten(tf.get_default_graph().get_tensor_by_name(graph_name + "0/" + layer))
    r_norm = tf.nn.l2_normalize(r_output, 1)
  with tf.variable_scope("reused", reuse=True) as scope:
    l_inputs = tf.get_default_graph().get_tensor_by_name(graph_name + "1/inputs:0")
    l_output = tf.contrib.layers.flatten(tf.get_default_graph().get_tensor_by_name(graph_name + "1/" + layer))
    l_norm = tf.nn.l2_normalize(l_output, 1)

  d = tf.reduce_sum(tf.square(l_norm - r_norm), 1)
  d_sqrt = tf.sqrt(d)
  l2 = tf.expand_dims(d_sqrt, 1)
  return l_inputs, r_inputs, r_norm, l_norm
