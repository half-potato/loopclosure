import mobilenet_v1 as mn
import tensorflow as tf

from collections import namedtuple
import model_util

slim = tf.contrib.slim

# Top half of the network that computes the features of each image
# layer_cutoff: The layer at which to cutoff the frozen mobilenet
# graph_path: The filepath to the frozen mobilenet
# is_training: Whether or not the network is training
# Returns:
# l_output: left output of network
# r_output: right output of network
def top(is_training, graph_path, layer_cutoff):
  def branch(name):
    inputs, output = model_util.load_mobilenet(name, graph_path, layer_cutoff)
    flat = tf.contrib.layers.flatten(output)
    dense = tf.layers.dense(inputs=flat, units=2048, activation=tf.nn.relu, name="Compression")
    return inputs, dense

  with slim.arg_scope([slim.batch_norm, slim.dropout],
                       is_training=is_training):
    with tf.variable_scope("mobilenet", reuse=False) as scope:
      l_inputs, l_output = branch("left")
    with tf.variable_scope(scope, reuse=True):
      r_inputs, r_output = branch("right")
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
  dense_1 = tf.layers.dense(inputs=merged, units=2048, activation=tf.nn.relu, name="DenseL1")
  #dense_1 = tf.layers.dense(inputs=merged, units=512, activation=tf.nn.relu)
  dense_2 = tf.layers.dense(inputs=dense_1, units=1024, activation=tf.nn.relu, name="DenseL2")
  #dense_2 = tf.layers.dense(inputs=merged, units=512, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense_2, rate=0.4, training=is_training)

  logits = tf.layers.dense(inputs=dropout, units=1, activation=None)
  result = tf.nn.sigmoid(logits, name="result")

  tf.summary.scalar('mean_output', tf.reduce_mean(result))
  return logits, result
