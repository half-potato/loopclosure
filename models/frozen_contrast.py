import tensorflow as tf
import model_util

slim = tf.contrib.slim

# Top half of the network that computes the features of each image
# layer_cutoff: The layer at which to cutoff the frozen mobilenet
# graph_path: The filepath to the frozen mobilenet
# is_training: Whether or not the network is training
# Returns:
# l_output: left output of network
# r_output: right output of network
def top(is_training, graph_path, layer_cutoff, left_input, right_input):
  def branch(name):
    inputs, output = model_util.load_mobilenet_and_continue(name, graph_path, layer_cutoff)
    flat = tf.contrib.layers.flatten(output)
    norm = tf.nn.l2_normalize(flat, 1, name="Norm")
    return inputs, norm

  with slim.arg_scope([slim.batch_norm, slim.dropout],
                       is_training=is_training):
    with tf.variable_scope("mobilenet") as scope:
      l_inputs, l_output = branch("net")
    with tf.variable_scope(scope, reuse=True):
      r_inputs, r_output = branch("net")

  return l_inputs, r_inputs, l_output, r_output


# Bottom half of the network that takes the top half outputs and produces a score
# l_output: left output of top network
# r_output: right output of top network
# is_training: Whether or not the network is training
# Returns:
# logits: The raw output of the network
# result: The true output of the network
def bot(l_output, r_output):
  d = tf.reduce_sum(tf.square(l_output - r_output), 1)
  #d_sqrt = tf.sqrt(d)
  l2 = tf.expand_dims(d, 1)
  out = tf.nn.sigmoid(l2, name="result")

  tf.summary.scalar('mean_output', tf.reduce_mean(out))
  return l2, out
