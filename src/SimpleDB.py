import os
import numpy as np
from DB import DB
from NamedDB import NamedDB

import csv

# A simple DB that simply reads off a whitelist of pairs
# This list can either be in txt or numpy mat form
# Such a list might look like:
# path/to/image1.jpg, path/to/image2.jpg
class SimpleDB(DB, NamedDB):
  # index_path: path to index pairs of image paths
  # image_path: relative path to image folder
  # img_stats: stats of dataset
  # multi: chance multiplier
  def __init__(self, index_path, image_path, img_stats, multi):
    NamedDB.__init__(self, index_path, "SimpleDB")
    DB.__init__(self, "unaltered", img_stats, multi)

    if "txt" in index_path or "csv" in index_path:
      # Load as csv
      self.positive_pairs = []
      with open(index_path, "r") as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|')
        for i in reader:
          self.positive_pairs.append(i)
      self.positive_pairs = np.array(self.positive_pairs)
    else:
      # Load as np array
      with open(index_path, "rb") as f:
        self.positive_pairs = np.load(f)
      # Flip if list is transposed
      if self.positive_pairs.shape[0] < self.positive_pairs.shape[1]:
        self.positive_pairs = self.positive_pairs.transpose()

    self.img_path = image_path

  # Overwritten
  # Returns: left and right image paths
  def get_pair_path(self, i):
    [p1, p2] = np.squeeze(self.positive_pairs[i,:])
    return os.path.join(self.img_path, p1.strip()), \
           os.path.join(self.img_path, p2.strip())

  # Overwritten
  def get_length(self):
    return self.positive_pairs.shape[0]

  length = property(get_length)
