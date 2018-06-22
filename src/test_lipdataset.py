import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import matplotlib.pyplot as plt

slim = tf.contrib.slim
import random
import numpy as np
import cv2
import glob
import os
import sys
import math
from datetime import datetime

sys.path.insert(0, "model")

import util
import image_util
import loadModel
import constants
import names

SAMPLE_SIZE = 1500

def main(
    dataset_name, # Lip6Indoor or Lip6Outdoor
    net_type, # deep, frozen... declared in constants
    run_name, # the name suffix of the ckpt
    image_stats, # the image stats used to train the net
    layer_cutoff = 11):

  sess = tf.Session()

  # Load net
  l_inputs, r_inputs, l_output, r_output, result, post = \
    loadModel.loadForTesting(net_type, layer_cutoff)

  # Load rest
  data_in, img_out = image_util.add_basic_ops(
                          image_stats.size, image_stats.size, image_stats.depth, 
                          image_stats.mean, image_stats.std)

  init = tf.global_variables_initializer()
  sess.run(init)
  post(sess)
  # Dataset paths
  dataset_path = "raw/"+dataset_name+"/data"
  gt_path = "indexes/"+dataset_name+".png"
  query_path = "indexes/"+dataset_name+"_candidates.txt"

  # Checkpoint dir
  ckpt_dir = "sessions/%s/ckpts_%s/" % (net_type, run_name)
  ckpt = tf.train.latest_checkpoint(ckpt_dir)

  # Save paths for results
  path = os.path.join("graphs", net_type)
  file_path = os.path.join(path, dataset_name+"_"+run_name+".csv")
  graph_path = os.path.join(path, dataset_name+"_"+run_name+".png")

  # Load ground truth image
  gt_index = cv2.imread(gt_path, 0) / 255
  gt_index = gt_index + gt_index.transpose()

  # Walk file directory and make file index
  img_index = np.array([os.path.join(dataset_path, i) for i in \
                          [x for x in gfile.Walk(dataset_path)][0][2]])
  img_index = np.sort(img_index)
  
  with tf.device("/cpu:0"):
    # Create image loader
    data_in, img_out = image_util.add_basic_ops(
                            image_stats.size, image_stats.size, image_stats.depth, 
                            image_stats.mean, image_stats.std)

  # Load rtabmap queries into array
  def process_queries(path):
    queries = np.array([[0,0]]).astype(int)
    with open(path, "r") as f:
      for line in f:
        comma = line.find(",")
        arr = np.array( line[comma+3:-2].split(", ") ).astype(int)[1:]
        if arr.shape[0] == 0:
          continue
        arr = arr - np.ones(arr.shape)
        num = int(line[:comma]) - 1
        nums = np.ones(arr.shape) * num
        pairs = np.vstack((nums, arr)).transpose()
        queries = np.concatenate((queries, pairs), axis=0)
    return queries[1:,:,None]
  pairs = process_queries(query_path)

  # Function to get a pair from the queries
  def get_pair(i):
    l, r = int(np.squeeze(pairs[i,0])), int(np.squeeze(pairs[i,1]))
    gt = int(gt_index[l, r])
    l_img = image_util.load_img(img_index[l], data_in, img_out, sess)
    r_img = image_util.load_img(img_index[r], data_in, img_out, sess)
    return l_img, r_img, gt

  length = pairs.shape[0]
  batches = int(math.floor(float(length)/SAMPLE_SIZE)) + 1

  def get_batch_size(batch_i):
    b = (batch_i+1) * SAMPLE_SIZE
    if b > length:
      return (length % SAMPLE_SIZE)
    return SAMPLE_SIZE

  print("Evaling %i examples in %i batches of size %i to %i" % 
       (length, batches, get_batch_size(0), get_batch_size(batches-1)))
  tf.logging.set_verbosity(tf.logging.INFO)

  img_bins = [np.array([[0,0,0,0]])] * gt_index.shape[0]
  for i in range(batches):
  #for i in range(5):
    # Load images
    size = get_batch_size(i)
    start = i * SAMPLE_SIZE
    l_imgs, r_imgs, gt_arr = [], [], []
    l_inds, r_inds = [], []
    for j in range(size):
      # Load values
      l, r, gt = get_pair(start + j)
      l_imgs.append(l)
      r_imgs.append(r)
      # Remember indices
      l_i, r_i = pairs[start+j, 0], pairs[start+j, 1] 
      l_inds.append(np.squeeze(l_i))
      r_inds.append(np.squeeze(r_i))
      gt_arr.append(gt)
    # Get proper dims
    l_imgs = np.squeeze(l_imgs)
    r_imgs = np.squeeze(r_imgs)
    gt_arr = np.expand_dims(gt_arr, 1)
    r_inds = np.expand_dims(r_inds, 1)
    l_inds = np.expand_dims(l_inds, 1)

    if constants.is_contrast(net_type):
      gt_arr = np.abs(np.ones(gt_arr.shape) - gt_arr)

    count = np.sum(gt_arr)

    # Run net
    output = sess.run(
      [result],
      feed_dict={l_inputs: l_imgs,
                 r_inputs: r_imgs})
    
    # Print entertainment
    print("\nBatch number %i" % i)
    print("Max outputput: %f" % np.max(output))
    print("Min outputput: %f" % np.min(output))
    print("STD outputput: %f" % np.std(output))
    print("Mean outputput: %f" % np.mean(output))
    print("Positive count: %i" % count)

    # Sort results into bins
    for j, v in enumerate(l_inds):
      # Create new result
      gt = int(np.squeeze(gt_index[int(r_inds[j]), int(v)]))
      out = float(np.squeeze(output)[j])
      ind1 = int(r_inds[j])
      ind2 = int(v)
      pair = [[out, gt, ind1, ind2]]
      # Add to corresponding bin
      img_bins[ind2] = np.vstack((img_bins[ind2], pair)) 

  # Calculate precision and recall
  top_n = 0
  precision = []
  thresholds = []
  recall = []
  reverse = -1
  if constants.is_contrast(net_type):
    reverse = reverse * -1

  for i in range(len(img_bins)):
    img_bins[i] = img_bins[i][1:,]
    img_bins[i] = img_bins[i][(img_bins[i][:,0]*reverse).argsort()]

  # Get top for every bin, sort, and divide into thresholds
  top_in_bins = [i[0,0] for i in img_bins if not i is None and i.size > 0]
  top_in_bins.sort()

  def get_threshold(i):
    return top_in_bins[i]

  for i in range(len(top_in_bins)):
    threshold = get_threshold(i)
    correct = []
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    # Count false pos neg and true pos neg for all bins
    for j in range(len(img_bins)):
      if img_bins[j].shape[0] >= top_n+2:
        correct_exists = np.max(img_bins[j][0:,1])
        if constants.is_contrast(net_type):
          if np.min(img_bins[j][0:top_n+1,0]) < threshold:
            if correct_exists == 0:
              false_pos += 1
            else:
              true_pos += 1
          else:
            if correct_exists == 1:
              false_neg += 1
            else:
              true_neg += 1
        else:
          if np.max(img_bins[j][0:top_n+1,0]) > threshold:
            if correct_exists == 0:
              false_pos += 1
            else:
              true_pos += 1
          else:
            if correct_exists == 1:
              false_neg += 1
            else:
              true_neg += 1

    precision.append(float(true_pos) / max(1, true_pos + false_pos))
    recall.append(float(true_pos) / max(1, false_neg + true_pos))
    thresholds.append(threshold)
  if not os.path.isdir(path):
    os.makedirs(path)
  with open(file_path, "w") as f:
    f.write("Precision, recall, threshold\n")
    for i in range(len(top_in_bins)):
      f.write(str(precision[i])+", "+str(recall[i])+", "+str(thresholds[i])+"\n")
  # Display results
  p_line = plt.plot(thresholds, precision)
  r_line = plt.plot(thresholds, recall)
  w_line = plt.plot(recall, precision)
  plt.setp(p_line, color='r')
  plt.setp(r_line, color='b')
  plt.setp(w_line, color='g')
  plt.savefig(graph_path)
  plt.cla()
  #plt.show()
  tf.reset_default_graph()
  sess.close()

if __name__ == "__main__":
  dataset_name = ["Lip6Indoor", "Lip6Outdoor"][int(sys.argv[2])]
  print("Testing on the %s dataset" % dataset_name)
  if sys.argv[1] == "all":
    for i in range(len(constants.SETTINGS)):
      # Get configs 
      net_type, layer_cutoff, db_type, learning_rate, batch_size, vary_ratio = \
          constants.SETTINGS[i]
      image_stats = constants.STATS[db_type]
      run_name = names.get_run_name(db_type, net_type, layer_cutoff)
      main(dataset_name, net_type, run_name, image_stats, layer_cutoff)
  elif sys.argv[1] == "alexnet":
    net_type = "places_alexnet"
    layer_cutoff = 0
    image_stats = constants.STATS["alexnet"]
    main(dataset_name, net_type, "pretrained_places", image_stats, layer_cutoff)
  else:
    arg = int(sys.argv[1])
    # Get configs 
    net_type, layer_cutoff, db_type, learning_rate, batch_size, vary_ratio = \
        constants.SETTINGS[arg]
    image_stats = constants.STATS[db_type]
    run_name = names.get_run_name(db_type, net_type, layer_cutoff)
    main(dataset_name, net_type, run_name, image_stats, layer_cutoff)
