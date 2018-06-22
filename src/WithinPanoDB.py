import numpy as np
import math
import image_util
import os
from DB import DB
from NamedDB import NamedDB
from tensorflow.python.platform import gfile

class WithinPanoDB(DB, NamedDB):
  # All dbs are in the format: img_name, lat, lon
  # white_list_path: path to the pair indices
  # index_path: path to index with paths of images
  # image_path: relative path to image folder
  # img_stats: stats of dataset
  # multi: chance multiplier
  def __init__(self, index_path, image_path, white_list_path, img_stats, multi):
    NamedDB.__init__(self, white_list_path, "WithinPanoDB")
    DB.__init__(self, "alter", img_stats, multi)
    with open(index_path, "r") as f:
      self.index = np.load(f)
    with open(white_list_path, "r") as f:
      self.whitelist = np.load(f).transpose()
    self.img_path = image_path
    self.locs, self.reverse_ind = np.unique(self.index[:,1:3], axis=0, return_index=True)
    self.img_per_loc = 12 # self.index.shape[0] / self.locs.shape[0]
    self.pairs_per_loc = self.img_per_loc# * 2 # subtract diagonal
    if len(self.whitelist.shape) != 1 and self.whitelist.shape[0] < self.whitelist.shape[1]:
      self.whitelist = self.whitelist.transpose()
    self.l_input, raw_l = image_util.add_jpeg_decoding(img_stats.depth)
    self.r_input, raw_r = image_util.add_jpeg_decoding(img_stats.depth)

    norm_l = image_util.normalize(raw_l, img_stats.mean, img_stats.std)
    norm_r = image_util.normalize(raw_r, img_stats.mean, img_stats.std)

    self.l_output = image_util.add_edge_crop(norm_l, img_stats.size, img_stats.size,
                                             320, 240, False)
    self.r_output = image_util.add_edge_crop(norm_r, img_stats.size, img_stats.size,
                                             320, 240, True)

  # Overwritten
  # Gets images within the same panoramic that are next to each other
  # Positive pairs has a bunch of pairs of indexes
  # The indexes are retrieved and the corresponding path is looked up in the index
  # Returns: right and left image paths
  def get_pair_path(self, index):
    i = np.squeeze(self.whitelist[index])
    loc_i = math.floor(float(i) / self.pairs_per_loc) * self.pairs_per_loc

    pair_i = i % self.pairs_per_loc

    left_i = (pair_i - 1) % self.img_per_loc
    right_i = pair_i

    left_path = self.index[int(loc_i + left_i), 0]
    right_path = self.index[int(loc_i + right_i), 0]
    p1 = os.path.join(self.img_path, left_path).strip()
    p2 = os.path.join(self.img_path, right_path).strip()
    return p1, p2

  # Overwritten
  # index: The index of the image to retrieve. This corresponds to the whitelist
  # of this db
  # sess: Session to use to load image. 
  # Returns: The left and right images with shape (1, 192, 192, 3)
  def get_image_pair(self, index, sess):
    l_p, r_p = self.get_pair_path(index)
    try:
      l_im = gfile.FastGFile(l_p, "rb").read()
      r_im = gfile.FastGFile(r_p, "rb").read()
      l, r = sess.run([self.l_output, self.r_output], 
          {self.l_input:l_im,
           self.r_input:r_im})
      return l, r
    except Exception as e:
      print(e)
      print("Images: \"%s\" and \"%s\" were not found" % (l_p, r_p))
      return None, None

  # Overwritten
  def get_length(self):
    return self.whitelist.shape[0]

  length = property(get_length)
