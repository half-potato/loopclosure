import numpy as np
import random, os
from DB import DB
from NamedDB import NamedDB

class GTADB(DB, NamedDB):
  # path: relative path to GTA data
  # val_or_train: "val" if validation set and "train" if training set
  # img_stats: stats of dataset
  # multi: chance multiplier
  # is_positive: Whether or not this DB gives positive or negative examples
  def __init__(self, img_path, path, img_stats, multi, is_positive=False):
    NamedDB.__init__(self, "neg", "SimpleDB")
    DB.__init__(self, "unaltered", img_stats, multi)
    self.is_positive = is_positive
    self.data = np.load(path)
    self.img_path = img_path
    print("Calculating length")
    if not is_positive:
      self.clusters, self.cluster_index, self.counts = \
          np.unique(self.data[:,0], return_index=True, return_counts=True)
      cluster_pairs = np.expand_dims(self.counts, 0) * np.expand_dims(self.counts, 1)
      self.l = np.sum(np.triu(cluster_pairs, k=1))
    print("Done loading %s" % self.name)

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
    i1 = self.cluster_index[q1]
    i2 = self.cluster_index[q2]

    # Generate a random index within the cluster
    ind1 = random.randrange(int(c1)) + i1
    ind2 = random.randrange(int(c2)) + i2

    # Lookup and paths and return them (paths are already relative from load_mats
    p1 = os.path.join(self.img_path, self.data[ind1, 1])
    p2 = os.path.join(self.img_path, self.data[ind2, 1])
    return p1, p2

  # i: the index of the pair
  # Returns: left and right image paths
  def get_positive_pair_path(self, i):
    [p1, p2] = self.data[i]
    return os.path.join(self.img_path, p1), os.path.join(self.img_path, p2)

  # Overwritten
  def get_length(self):
    if self.is_positive:
      return self.data.shape[0]
    else:
      x = self.data.shape[0] // 33
      return self.l
  length = property(get_length)
