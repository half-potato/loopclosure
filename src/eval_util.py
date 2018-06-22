import tensorflow as tf
import constants

def add_contrastive_loss(left, right, ground_truth, learning_rate, threshold, margin):
  with tf.name_scope('contrastive_loss'):
    d = tf.reduce_sum(tf.square(left - right), 1)
    d_sqrt = tf.sqrt(d)

    loss = (1-ground_truth) * tf.square(tf.maximum(0., margin - d_sqrt)) +\
           ground_truth * d

    loss = 0.5 * tf.reduce_sum(loss)
  tf.summary.scalar('contrastive_loss', loss)
  l2 = tf.expand_dims(d_sqrt, 1)

  with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, constants.DECAY, 
      constants.MOMENTUM, constants.EPSILON)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.less(d_sqrt, threshold)
      correct_prediction = tf.equal(
          prediction, tf.less(ground_truth, threshold))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return train_step, evaluation_step, prediction,\
         loss, correct_prediction

def add_sigmoid_cross_entropy(result, ground_truth, logits, learning_rate, threshold):
  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ground_truth, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, constants.DECAY, 
      constants.MOMENTUM, constants.EPSILON)
    train_step = optimizer.minimize(cross_entropy_mean)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.greater(result, threshold)
      correct_prediction = tf.equal(
          prediction, tf.greater(ground_truth, threshold))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return train_step, evaluation_step, prediction, \
         cross_entropy_mean, correct_prediction

def add_precision_recall(result, ground_truth, threshold):
  with tf.name_scope('precision'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.squeeze(tf.greater(result, threshold))
      gt_cast = tf.squeeze(tf.greater(ground_truth, threshold))

      correct_prediction = tf.equal(prediction, gt_cast)

      true_pos = tf.multiply(tf.cast(gt_cast, tf.float32), \
                             tf.cast(prediction, tf.float32))
      num_true_pos = tf.reduce_sum(tf.cast(true_pos, tf.float32))
      num_prediction = tf.reduce_sum(tf.cast(prediction, tf.float32))
      num_gt = tf.reduce_sum(tf.cast(gt_cast, tf.float32))
    with tf.name_scope('precision'):
      precision_w_nan = tf.div(num_true_pos, num_prediction)
      precision_step = tf.where(tf.is_nan(precision_w_nan), tf.ones_like(precision_w_nan), precision_w_nan)
    with tf.name_scope('recall'):
      recall_w_nan = tf.div(num_true_pos, num_gt)
      recall_step = tf.where(tf.is_nan(recall_w_nan), tf.ones_like(recall_w_nan), recall_w_nan)
  tf.summary.scalar('precision', precision_w_nan)
  tf.summary.scalar('recall', recall_w_nan)
  return precision_w_nan, recall_w_nan
