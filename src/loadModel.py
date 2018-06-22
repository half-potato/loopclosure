import tensorflow as tf
import constants
import eval_util
import sys
sys.path.insert(0, "models")
import model_util

DEF_THRES = 0.5

def loadForTraining(net_type, layer_cutoff, learning_rate, threshold=DEF_THRES,
                    left_input=tf.placeholder(tf.float32, model_util.DEF_SHAPE),
                    right_input=tf.placeholder(tf.float32, model_util.DEF_SHAPE)):
  net = constants.NET_TYPES[net_type]
  gt = tf.placeholder(tf.float32, [None, 1], name='GroundTruthInput')
  # Frozen
  if constants.is_frozen(net_type):
    # Top
    l_inputs, r_inputs, l_output, r_output = net.top(True, \
        constants.FROZEN_GRAPH, layer_cutoff, left_input, right_input)
  else:
    # Top
    l_inputs, r_inputs, l_output, r_output = net.top(True, left_input, right_input)

  # Contrast
  if constants.is_contrast(net_type):
    # Bot
    logits, result = net.bot(l_output, r_output)
  else:
    # Bot
    logits, result = net.bot(l_output, r_output, True)
    # Loss
  train_step, evaluation_step, prediction, mean_loss, correct_prediction = \
      eval_util.add_sigmoid_cross_entropy(result, gt, logits,\
          learning_rate, threshold)
  precision, recall = eval_util.add_precision_recall(result, gt, DEF_THRES)

  return l_inputs, r_inputs, l_output, r_output, gt, \
         train_step, evaluation_step, prediction, mean_loss, correct_prediction, \
         precision, recall

def loadForTesting(net_type, layer_cutoff,
                   left_input=tf.placeholder(tf.float32, model_util.DEF_SHAPE),
                   right_input=tf.placeholder(tf.float32, model_util.DEF_SHAPE)):
  net = constants.NET_TYPES[net_type]
  def post(sess):
    print("Done")
  if net_type == "places_alexnet":
    # Top
    l_inputs, r_inputs, l_output, r_output, post = net.top(True)
    result = net.bot(l_output, r_output)
    return l_inputs, r_inputs, l_output, r_output, result, post
  # Frozen
  elif constants.is_frozen(net_type):
    # Top
    l_inputs, r_inputs, l_output, r_output = net.top(True,
        constants.FROZEN_GRAPH, layer_cutoff, left_input, right_input)
  else:
    # Top
    l_inputs, r_inputs, l_output, r_output = net.top(True,
        left_input, right_input)

  # Contrast
  if constants.is_contrast(net_type):
    # Bot
    logits, result = net.bot(l_output, r_output)
  else:
    # Bot
    logits, result = net.bot(l_output, r_output, True)
    # Loss

  return l_inputs, r_inputs, l_output, r_output, result, post
