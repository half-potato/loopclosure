from places_alexnet_model import CaffeNetPlaces365
import tensorflow as tf
import model_util

# Top half of the network that computes the features of each image
# is_training: Whether or not the network is training
# Returns:
# l_output: left output of network
# r_output: right output of network
def top(is_training):
  def branch(name):
    scope = tf.contrib.framework.get_name_scope()
    shape = (None, 227, 227, 3)
    inputs = tf.placeholder(tf.float32, shape)
    net = CaffeNetPlaces365({'data': inputs})
    #net.load('models/places_alexnet_data.npy', scope+"/", sess)
    output = net.layers['pool5']
    flat = tf.contrib.layers.flatten(output)
    norm = tf.nn.l2_normalize(flat, 1, name="Norm")
    return inputs, norm, net, scope

  with tf.variable_scope("mobilenet") as scope:
    l_inputs, l_output, l_net, l_scope = branch("net")
  with tf.variable_scope(scope, reuse=True):
    r_inputs, r_output, r_net, r_scope = branch("net")

  def post(sess):
    with tf.variable_scope("mobilenet") as scope:
      #l_net.load('models/places_alexnet_data.npy', l_scope+"/", sess)
      l_net.load('models/places_alexnet_data.npy', sess)
    with tf.variable_scope(scope, reuse=True):
      #r_net.load('models/places_alexnet_data.npy', r_scope+"/", sess)
      r_net.load('models/places_alexnet_data.npy', sess)

  return l_inputs, r_inputs, l_output, r_output, post

# Bottom half of the network that takes the top half outputs and produces a score
# l_output: left output of top network
# r_output: right output of top network
# is_training: Whether or not the network is training
# Returns:
# logits: The raw output of the network
# result: The true output of the network
def bot(l_output, r_output):
  d = tf.reduce_sum(tf.square(l_output - r_output), 1)
  d_sqrt = tf.sqrt(d)
  l2 = tf.expand_dims(d_sqrt, 1)

  tf.summary.scalar('mean_output', tf.reduce_mean(l2))
  return l2
