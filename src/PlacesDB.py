import os
import random
from tensorflow.python.platform import gfile
import numpy as np
import image_util
from DB import DB
from NamedDB import NamedDB

import csv

# Database that reads from Places
# Crops a single image to generate positive images
# Negatives are from different categories
class PlacesDB(DB, NamedDB):
  # index_path: path to index pairs of image paths
  # image_path: relative path to image folder
  # img_stats: stats of dataset
  # multi: chance multiplier
  def __init__(self, index_path, image_path, img_stats, multi, is_positive=True):
    NamedDB.__init__(self, index_path, "SimpleDB")
    DB.__init__(self, "unaltered", img_stats, multi)

    with open(index_path, "r") as f:
      self.cat_index = []
      for i in f:
        items = i.split(" ")
        cat = int(items[1].strip())
        path = items[0]
        self.cat_index.append([path, cat])
    self.cat_index = np.array(self.cat_index)

    self.is_positive = is_positive
    self.img_path = image_path

    # Image processing
    self.input, out = image_util.add_jpeg_decoding(img_stats.depth)
    norm = image_util.normalize(out, img_stats.mean, img_stats.std)
    self.output = image_util.add_random_crop(out, 0.70, 0.90,
        img_stats.size, img_stats.size, 256, 256, 3)

    # Negative
    self.clusters, self.index, self.counts = np.unique(self.cat_index[:,1],
        return_counts=True, return_index=True)
    self.length = 0
    for i in range(len(self.counts)):
      for j in range(len(self.counts)-1):
        if i == j:
          continue
        self.length += self.counts[i] * self.counts[j]

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
  def get_image_pair(self, i, sess):
    if self.is_positive:
      return self.get_positive_pair(i, sess)
    else:
      l_p, r_p = self.get_negative_pair_path(i)
      #l_p = os.path.join(self.img_path, l_p)
      #r_p = os.path.join(self.img_path, r_p)
      return self.load_image(l_p, sess), self.load_image(l_p, sess)

  # i: the index of the pair
  # Returns: left and right image paths
  def get_negative_pair_path(self, i):
    # always random cause complicated
    q1 = random.randrange(self.clusters.shape[0])
    q2 = random.randrange(self.clusters.shape[0])
    # keep getting a random cluster until they are not equal
    while q1 == q2:
      q2 = random.randrange(self.clusters.shape[0])

    # Get the amount of images in the cluster
    c1 = self.counts[q1]
    c2 = self.counts[q2]

    # Get the start index of the cluster
    i1 = self.index[q1]
    i2 = self.index[q2]

    # Generate a random index within the cluster
    ind1 = random.randrange(int(c1)) + i1
    ind2 = random.randrange(int(c2)) + i2

    # Lookup and paths and return them (paths are already relative from load_mats
    l_p = self.img_path + self.cat_index[ind1, 0]
    r_p = self.img_path + self.cat_index[ind2, 0]
    return l_p, r_p

  # index: The index of the image to retrieve. This corresponds to the whitelist
  # of this db
  # sess: Session to use to load image. 
  # Returns: The left and right images with shape (1, 192, 192, 3)
  def get_positive_pair(self, i, sess):
    path = self.img_path + self.cat_index[i, 0]
    try:
      dat = gfile.FastGFile(path, "rb").read()
      left = sess.run(self.output, {self.input: dat})
      right = sess.run(self.output, {self.input: dat})
      return left, right
    except Exception as e:
      print(e)
      print("\"" + path + "\"")
      return None, None

  # Overwritten
  def get_length(self):
    if self.is_positive:
      return self.cat_index.shape[0]
    else:
      return self.length

  length = property(get_length)

