import mobilenet_v1 as mn
import tensorflow as tf
import model_util

slim = tf.contrib.slim

# Top half of the network that computes the features of each image
# is_training: Whether or not the network is training
# Returns:
# l_output: left output of network
# r_output: right output of network
def top(is_training, left_input, right_input):
  def branch(name, inputs):
    with tf.variable_scope(name):
      output, endpoints = mn.mobilenet_v1_base(inputs, \
          depth_multiplier=model_util.MOBILENET_DEP_MULTI)
      #kernel = mn.reduced_kernel_size_for_small_input(left_output, KERNEL_SIZE)
      #pool_l = slim.avg_pool2d(left_output, kernel, padding='VALID')
      flat = tf.contrib.layers.flatten(output)
      return inputs, flat

  with slim.arg_scope([slim.batch_norm, slim.dropout],
                       is_training=is_training):
    with tf.variable_scope("mobilenet") as scope:
      l_inputs, l_output = branch("net", left_input)
    with tf.variable_scope(scope, reuse=True):
      r_inputs, r_output = branch("net", right_input)
  return l_inputs, r_inputs, l_output, r_output

# Bottom half of the network that takes the top half outputs and produces a score
# l_output: left output of top network
# r_output: right output of top network
# is_training: Whether or not the network is training
# Returns:
# logits: The raw output of the network
# result: The true output of the network
def bot(l_output, r_output, is_training):
  merged = tf.concat([l_output, r_output], 1)
  dense_1 = tf.layers.dense(inputs=merged, units=1024, activation=tf.nn.relu, name="DenseL1")
  #dense_1 = tf.layers.dense(inputs=merged, units=512, activation=tf.nn.relu)
  dense_2 = tf.layers.dense(inputs=dense_1, units=1024, activation=tf.nn.relu, name="DenseL2")
  #dense_2 = tf.layers.dense(inputs=merged, units=512, activation=tf.nn.relu)

  dropout = tf.layers.dropout(inputs=dense_2, rate=0.4, training=is_training, name="Dropout")

  logits = tf.layers.dense(inputs=dropout, units=1, activation=None, name="Logits")
  result = tf.nn.sigmoid(logits, name="result")

  tf.summary.scalar('mean_output', tf.reduce_mean(result))
  return logits, result
