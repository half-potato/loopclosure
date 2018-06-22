import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

import numpy as np
import cv2
import os

def launchTensorBoard(path):
  def board_thread(path):
    os.system('pkill tensorboard; tensorboard --logdir=' + path)
    return
  import threading
  t = threading.Thread(target=board_thread, args=([path]))
  t.start()

def count_summaries(path):
  try:
    fs = [x[1] for x in gfile.Walk(path)][0]
    return len(fs) / 2
  except Exception as e:
    return 0

def count_ckpts(path):
  fs = [x[2] for x in gfile.Walk(path)][0]
  #count meta
  m = 0
  for i in fs:
    if "meta" in i:
      m += 1
  return m

def save_checkpoint(path, sess, saver):
  #new_ckpt = os.path.join(path, "mobile_places-" + str(TRAINING_BATCH_SIZE) +\
  #           "-" + str(count_ckpts(path)) + ".ckpt")
  #global saver
  if not os.path.isdir(path):
    os.makedirs(path)
  new_ckpt = os.path.join(path, "checkpoint")
  saver.save(sess, new_ckpt, global_step=count_ckpts(path)+1)
  print("Saved: " + new_ckpt)

def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

def disp_img_pair(l, r):
  ss = np.concatenate((l, r), axis=1)
  mn = np.min(ss)
  mx = np.max(ss)
  ss = (ss + mn + 0.2) * (250 /mx)
  ss = np.squeeze(ss.astype(np.uint8))
  ss = cv2.cvtColor(ss, cv2.COLOR_RGB2BGR)
  cv2.imshow("image", ss)
  cv2.waitKey(0)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
