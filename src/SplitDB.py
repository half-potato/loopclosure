import tensorflow as tf
import image_util
import numpy as np
import os
from NamedDB import NamedDB
from tensorflow.python.platform import gfile
from DB import DB

class SplitDB(NamedDB, DB):
  # white_list_path: path to the pair indices
  # index_path: path to index with paths of images
  # image_path: relative path to image folder
  # img_stats: stats of dataset
  # multi: chance multiplier
  def __init__(self, file_index, path, img_stats,
               raw_height, raw_width, multi, whitelist_path):
    NamedDB.__init__(self, whitelist_path, "SplitImageDB")
    DB.__init__(self, "unaltered", img_stats, multi)
    self.image_path = path
    with open(file_index, "r") as f:
      self.index = np.load(f)
    with open(whitelist_path, "r") as f:
      self.whitelist = np.load(f)
    with tf.variable_scope("jpeg_distort"):
      self.input, image = image_util.add_jpeg_decoding(img_stats.depth)
      l_output, r_output = image_util.add_cropper(
                                image, raw_height, raw_width,
                                img_stats.size, img_stats.size)
      self.l_output = image_util.normalize(image_util.brighten(l_output, 5),
                                img_stats.mean, img_stats.std)
      self.r_output = image_util.normalize(image_util.brighten(r_output, 5),
                                img_stats.mean, img_stats.std)

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
