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
  print(layer_cutoff)
  with slim.arg_scope([slim.batch_norm, slim.dropout],
                       is_training=is_training):
    with tf.variable_scope("mobilenet") as scope:
      l_inputs, l_output = model_util.load_mobilenet_and_continue(
          "left", graph_path, layer_cutoff)
    with tf.variable_scope(scope, reuse=True):
      r_inputs, r_output = model_util.load_mobilenet_and_continue(
          "right", graph_path, layer_cutoff)
  return l_inputs, r_inputs, l_output, r_output

# Bottom half of the network that takes the top half outputs and produces a score
# l_output: left output of top network
# r_output: right output of top network
# is_training: Whether or not the network is training
# Returns:
# logits: The raw output of the network
# result: The true output of the network
def bot(l_output, r_output, is_training):
  flat_left = tf.contrib.layers.flatten(l_output)
  flat_right = tf.contrib.layers.flatten(r_output)

  merged = tf.concat([flat_left, flat_right], 1)
  dense_1 = tf.layers.dense(inputs=merged, units=1024, activation=tf.nn.relu, name="DenseL1")
  #norm_1 = tf.layers.batch_normalization(dense_1, training=is_training)
  dense_2 = tf.layers.dense(inputs=dense_1, units=1024, activation=tf.nn.relu, name="DenseL2")
  #norm_2 = tf.layers.batch_normalization(dense_2, training=is_training)
  dropout = tf.layers.dropout(inputs=dense_2, rate=0.4, training=is_training)

  logits = tf.layers.dense(inputs=dropout, units=1, activation=None, name="Logits")
  result = tf.nn.sigmoid(logits, name="Result")

  tf.summary.scalar('mean_output', tf.reduce_mean(result))
  return logits, result
