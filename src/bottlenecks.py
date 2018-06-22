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
    image_stats, # the image stats used to train the net
    layer_cutoff = 11):

  sess = tf.Session()

  # Load net
  l_inputs, r_inputs, l_output, r_output, result = \
    loadModel.loadForTesting(net_type, layer_cutoff, learning_rate)

  # Load rest
  data_in, img_out = image_util.add_basic_ops(
                          image_stats.size, image_stats.size, image_stats.depth, 
                          image_stats.mean, image_stats.std)

  init = tf.global_variables_initializer()
  sess.run(init)
  # Dataset paths
  dataset_path = "raw/"+dataset_name+"/data"
  gt_path = "indexes/"+dataset_name+".png"
  query_path = "indexes/"+dataset_name+"_candidates.txt"

  # Checkpoint dir
  run_name = names.get_run_name(db_type, net_type, layer_cutoff)
  ckpt_dir = "sessions/%s/ckpts_%s/" % (net_type, run_name)
  ckpt = tf.train.latest_checkpoint(ckpt_dir)

  # Walk file directory and make file index
  img_index = np.array([os.path.join(dataset_path, i) for i in \
                          [x for x in gfile.Walk(dataset_path)][0][2]])
  img_index = np.sort(img_index)
  
  # Create image loader
  data_in, img_out = image_util.add_basic_ops(
                          image_stats.size, image_stats.size, image_stats.depth, 
                          image_stats.mean, image_stats.std)

  length = img_index.size
  batches = int(math.floor(float(length)/SAMPLE_SIZE)) + 1

  # Convienence function
  def get_batch_size(batch_i):
    b = (batch_i+1) * SAMPLE_SIZE
    if b > length:
      return (length % SAMPLE_SIZE)
    return SAMPLE_SIZE

  print("Evaling %i examples in %i batches of size %i to %i" % 
       (length, batches, get_batch_size(0), get_batch_size(batches-1)))
  tf.logging.set_verbosity(tf.logging.INFO)

  descriptors = None
  for i in range(batches):
  #for i in range(5):
    # Load images
    size = get_batch_size(i)
    start = i * SAMPLE_SIZE
    imgs = []
    for j in range(size):
      img = image_util.load_img(img_index[start+j], data_in, img_out, sess)
      imgs.append(img)
    imgs = np.squeeze(imgs)

    # Run net
    out = sess.run(
      [l_output],
      feed_dict={l_input: imgs})

    # Store descriptors
    if descriptors is None:
      descriptors = np.squeeze(out)
    else:
      descriptors = np.hstack((descriptors, out))

  path = os.path.join("csvs", net_type)
  if not os.path.isdir(path):
    os.makedirs(path)

  # Normalize
  l2 = np.sum(np.square(descriptors), axis=1)
  descriptors = descriptors / l2[:, np.newaxis]
  
  if YAML:
    file_path = os.path.join(path, dataset_name+"_"+run_name+"_descriptors.yaml")
    print(file_path)
    cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("descriptors", descriptors)
    cv_file.release()
  else:
    # CSV
    file_path = os.path.join(path, dataset_name+"_"+run_name+"_descriptors.csv")
    print(file_path)
    with open(file_path, "w") as f:
      for i in descriptors:
        out = ""
        for j in i:
          out += str(np.squeeze(j)) + ", "
        out += "\n"
        f.write(out)

if __name__ == "__main__":
  dataset_name = ["Lip6Indoor", "Lip6Outdoor"][int(sys.argv[2])]
  print("Generating for the %s dataset" % dataset_name)
  if sys.argv[1] == "all":
    for i in range(len(constants.SETTINGS)):
      # Get configs 
      net_type, layer_cutoff, db_type, learning_rate, batch_size, vary_ratio = \
          constants.SETTINGS[i]
      image_stats = constants.STATS[db_type]
      main(dataset_name, net_type, image_stats, layer_cutoff)
  else:
    arg = int(sys.argv[1])
    # Get configs 
    net_type, layer_cutoff, db_type, learning_rate, batch_size, vary_ratio = \
        constants.SETTINGS[arg]
    image_stats = constants.STATS[db_type]
    main(dataset_name, net_type, image_stats, layer_cutoff)

