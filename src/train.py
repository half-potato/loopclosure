import tensorflow as tf

import random
import os
import numpy as np
from datetime import datetime

import util
import math
import constants
import loadModel
import load_dbs
import sys
import names

slim = tf.contrib.slim

def main(
    db_type, # cnn, cnn_nopitts, cnn_rgbd
    net_type, # deep, frozen, ...
    training_steps, # number of training steps
    image_stats, # stats of the training db
    learning_rate, 
    training_batch_size, 
    vary_ratio, # whether to vary the ratio of pos to neg or not
    layer_cutoff = 11, # layer cutoff for frozen networks. Unused if not
    ratio = 1./2, # Ratio of pos to neg
    ratio_range=(5,45), # min, max when varying the ratio
    validation_interval = 100, # moving average size and the intervals at which to do val
    validation_batch_size = 200, # size of validation batch
    ckpt_save_interval = 100): # save interval

  sess = tf.Session()
  # For saving stuff
  run_name = names.get_run_name(db_type, net_type, layer_cutoff)
  ckpt_dir, summaries_dir = names.get_dirs(run_name, net_type)

  # Load model
  l_inputs, r_inputs, l_output, r_output, ground_truth, \
    train_step, evaluation_step, prediction, mean_loss, correct_prediction, \
      precision_step, recall_step = \
        loadModel.loadForTraining(net_type, layer_cutoff, learning_rate)

  # Init summaries
  SUMMARY_NUMBER = util.count_summaries(summaries_dir)
  train_log_dir = "train%i" % SUMMARY_NUMBER
  val_log_dir = "val%i" % SUMMARY_NUMBER
  all_sums = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(summaries_dir + '/' + train_log_dir, sess.graph)
  validation_writer = tf.summary.FileWriter(summaries_dir + '/' + val_log_dir)
  tf.logging.set_verbosity(tf.logging.INFO)

  # Init moving accuracies
  accuracies = []
  recalls = []
  precisions = []
  losses = []

  saver = tf.train.Saver(keep_checkpoint_every_n_hours=3, max_to_keep=5)

  # Init database
  with tf.device("/cpu:0"):
    train_ret, val_ret, test_ret = load_dbs.load_retrievers(db_type)

  init = tf.global_variables_initializer()
  sess.run(init)

  # Load checkpoint
  print(ckpt_dir)
  ckpt = tf.train.latest_checkpoint(ckpt_dir)
  if not ckpt is None:
    rest_var = slim.get_variables_to_restore()
    saver = tf.train.Saver(rest_var)
    saver.restore(sess, ckpt)

  print("Starting training")
  for i in range(training_steps):
    # RANDOMIZE RATIO
    if vary_ratio:
      ratio = 1./(random.random()*ratio_range[1] + ratio_range[0])

    # If for some odd reason you want to use stochastic gradient descent
    if training_batch_size > 1:
      # GET IMAGES
      r_train, l_train, gt_train = train_ret.get_random_pairs(sess, training_batch_size, ratio)
    else:
      # Get single image
      gt_bool = random.random() > ratio
      if gt_bool:
        r_train, l_train = train_ret.positive_ret.get_random_pair(sess)
        gt_train = np.array([[1]])
      else:
        r_train, l_train = train_ret.negative_ret.get_random_pair(sess)
        gt_train = np.array([[0]])

    if constants.is_contrast(net_type):
      gt_train = np.abs(np.ones(gt_train.shape) - gt_train)

    train_summary, _, train_accuracy, loss, \
    precision, recall = sess.run(
        [all_sums, train_step, evaluation_step, mean_loss, \
        precision_step, recall_step],
        feed_dict={l_inputs: l_train,
                   r_inputs: r_train,
                   ground_truth: gt_train})
    train_writer.add_summary(train_summary, i)
    # Moving Average Accuracy
    accuracies.append(train_accuracy)
    losses.append(loss)
    if not math.isnan(recall):
      recalls.append(recall)
    if not math.isnan(precision):
      precisions.append(precision)

    if len(accuracies) > validation_interval:
      accuracies = accuracies[1:]
    if len(losses) > validation_interval:
      losses = losses[1:]
    if len(recalls) > validation_interval:
      recalls = recalls[1:]
    if len(precisions) > validation_interval:
      precisions = precisions[1:]

    # Checkpoint saving
    if (((i % ckpt_save_interval) == 0) or (i + 1 == training_steps)) and i != 0:
      util.save_checkpoint(ckpt_dir, sess, saver)

    # Every so often, print out how well the graph is training.
    if ((i % validation_interval) == 0) or (i + 1 == training_steps):
      #tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                      #(datetime.now(), i, np.mean(accuracies) * 100))
      tf.logging.info('%s: Step %d: Loss = %f' %
                      (datetime.now(), i, np.mean(losses)))
      tf.logging.info('%s: Step %d: Recall = %.1f%%' %
                      (datetime.now(), i, np.mean(recalls) * 100))
      tf.logging.info('%s: Step %d: Precision = %.1f%%' %
                      (datetime.now(), i, np.mean(precisions) * 100))
      # VALIDATION
      r_val, l_val, gt_val = val_ret.get_random_pairs(sess, validation_batch_size, ratio)
      if constants.is_contrast(net_type):
        gt_val = np.abs(np.ones(gt_val.shape) - gt_val)
      validation_summary, validation_accuracy, val_precision, val_recall = sess.run(
          [all_sums, evaluation_step, \
          precision_step, recall_step],
          feed_dict={l_inputs: l_train,
                     r_inputs: r_train,
                     ground_truth: gt_train})

      validation_writer.add_summary(validation_summary, i)
      tf.logging.info('%s: Step %d: Validation recall = %.1f%%' %
                      (datetime.now(), i, val_recall * 100))
      tf.logging.info('%s: Step %d: Validation precision = %.1f%%' %
                      (datetime.now(), i, val_precision * 100))

  util.save_checkpoint(ckpt_dir, sess, saver)
  tf.reset_default_graph()
  sess.close()
  os.system('pkill tensorboard')

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: ./src/train.py SETTING_NUMBER TRAINING_STEPS")
    sys.exit()
  steps = int(sys.argv[2])
  if sys.argv[1] == "all":
    for i in range(len(constants.SETTINGS)):
      # Get configs 
      net_type, layer_cutoff, db_type, learning_rate, batch_size, vary_ratio = \
          constants.SETTINGS[i]
      img_stat = constants.STATS[db_type]
      run_name = names.get_run_name(db_type, net_type, layer_cutoff)
      util.launchTensorBoard("sessions/%s/logs_%s" % (net_type, run_name))
      main(db_type, net_type, steps, img_stat, learning_rate, batch_size, 
           vary_ratio, layer_cutoff)
  else:
    arg = int(sys.argv[1])
    # Get configs 
    net_type, layer_cutoff, db_type, learning_rate, batch_size, vary_ratio = \
        constants.SETTINGS[arg]
    img_stat = constants.STATS[db_type]
    run_name = names.get_run_name(db_type, net_type, layer_cutoff)
    util.launchTensorBoard("sessions/%s/logs_%s" % (net_type, run_name))
    main(db_type, net_type, steps, img_stat, learning_rate, batch_size, 
         vary_ratio, layer_cutoff)
