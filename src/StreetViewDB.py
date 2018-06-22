import calc
import math
import os
import numpy as np
from DB import DB
from NamedDB import NamedDB

PAIRS_PER_PANO = 6

# Loads image pairs between locations in StreetView
class StreetViewDB(DB, NamedDB):
  # white_list_path: path to the pair indices
  # index_path: path to index with paths of images
  # image_path: relative path to image folder
  # img_stats: stats of dataset
  # multi: chance multiplier
  def __init__(self, white_list_path, index_path, image_path, img_stats, multi):
    NamedDB.__init__(self, white_list_path, "StreetViewDB")
    DB.__init__(self, "unaltered", img_stats, multi)
    with open(white_list_path, "r") as f:
      self.positive_pairs = np.squeeze(np.load(f))
    if self.positive_pairs.shape[0] < self.positive_pairs.shape[1]:
      self.positive_pairs = self.positive_pairs.transpose()

    # path, lat, lon
    with open(index_path, "r") as f:
      self.index = np.squeeze(np.load(f))

    self.img_path = image_path

  # Positive pairs has a bunch of pairs of indexes
  # The indexes are retrieved and the corresponding path is looked up in the index
  # Returns: right and left image paths
  def get_pair_path(self, i):
    [i1, i2] = self.positive_pairs[i,:]
    p1 = self.index[int(i1), 0]
    p2 = self.index[int(i2), 0]
    p1 = os.path.join(self.img_path, p1).strip()
    p2 = os.path.join(self.img_path, p2).strip()
    return p1, p2

  # Overwritten
  def get_length(self):
    return self.positive_pairs.shape[0]

  length = property(get_length)
