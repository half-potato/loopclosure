import numpy as np
import random
from DB import DB
from NamedDB import NamedDB

import cnn

class CNNDB(DB, NamedDB):
  # path: relative path to CNN data
  # val_or_train: "val" if validation set and "train" if training set
  # img_stats: stats of dataset
  # multi: chance multiplier
  # is_positive: Whether or not this DB gives positive or negative examples
  def __init__(self, path, val_or_train, img_stats, multi, is_positive=False):
    NamedDB.__init__(self, "neg", "SimpleDB")
    DB.__init__(self, "unaltered", img_stats, multi)
    self.is_positive = is_positive
    self.qpid, self.clucid, _ = cnn.load_mats(path, 120, val_or_train)
    self.clusters, self.index, self.counts = np.unique(self.clucid[:,1], return_counts=True, return_index=True)
    self.clusters = self.clusters.astype(float).astype(int)
    self.length = 0
    for i in range(len(self.counts)):
      for j in range(len(self.counts)-1):
        if i == j:
          continue
        self.length += self.counts[i] * self.counts[j]

  # Overwritten
  # i: the index of the pair
  # Returns: left and right image paths
  def get_pair_path(self, i):
    if self.is_positive:
      return self.get_positive_pair_path(i)
    else:
      return self.get_negative_pair_path(i)

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
    return self.clucid[ind1, 0], self.clucid[ind2, 0]

  # i: the index of the pair
  # Returns: left and right image paths
  def get_positive_pair_path(self, i):
      [i1, i2] = self.qpid[i].astype(int)
      p1 = self.clucid[i1, 0]
      p2 = self.clucid[i2, 0]
      return p1, p2

  # Overwritten
  def get_length(self):
    if self.is_positive:
      return self.qpid.shape[0]
    else:
      return self.length

  length = property(get_length)
