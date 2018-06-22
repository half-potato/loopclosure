import tensorflow as tf
import numpy as np
import random
import math
import os
import image_util
from NamedDB import NamedDB
from DB import DB
from tensorflow.python.platform import gfile

class FlipDB(NamedDB, DB):
  def __init__(self, file_index, path, img_stats, height, width, multi, whitelist_path):
    NamedDB.__init__(self, whitelist_path, "FlipDB")
    DB.__init__(self, "unaltered", img_stats, multi)
    self.image_path = path
    with open(file_index, "r") as f:
      self.index = np.load(f)
    with open(whitelist_path, "r") as f:
      self.whitelist = np.load(f)
    with tf.variable_scope("jpeg_distort"):
      self.input, image = image_util.add_jpeg_decoding(img_stats.depth)
      resized = image_util.resize(image, img_stats.size, img_stats.size)
      norm_l = image_util.normalize(image_util.brighten(resized, 5),
                                    img_stats.mean, img_stats.std)
      norm_r = image_util.normalize(image_util.brighten(resized, 5),
                                    img_stats.mean, img_stats.std)
      self.l_output = tf.expand_dims(tf.squeeze(norm_l), 0)
      self.r_output = tf.expand_dims(tf.image.flip_left_right(tf.squeeze(norm_r)), 0)
    self.image_whitelist = np.unique(self.whitelist)

  # Overwritten
  # There is no implementation for this because the image path is the same for
  # both images
  def get_pair_path(self, i):
    return None, None

  # Overwritten
  # index: The index of the image to retrieve. This corresponds to the whitelist
  # of this db
  # sess: Session to use to load image. 
  # Returns: The left and right images with shape (1, 192, 192, 3)
  def get_image_pair(self, index, sess):
    #i = int(math.floor(self.start_percent * self.index.shape[0])) + i
    i = np.squeeze(self.whitelist[index])
    path = os.path.join(self.image_path, self.index[i,0].strip())
    try:
      im = gfile.FastGFile(path, "rb").read()
      l, r = sess.run([self.l_output, self.r_output], {self.input:im})
      return l, r
    except Exception as e:
      print(e)
      print("\"" + path + "\"")
      return None, None

  # Overwritten
  def get_length(self):
    return self.whitelist.shape[0]

  length = property(get_length)
