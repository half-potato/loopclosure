import numpy as np
import math
import random

# Mixes a positive and negative retriever
# The final level of dataset abstraction
class MixedRetriever:
  # positive_ret: A retriever with positive datasets
  # negative_ret: A retriever with negative datasets
  def __init__(self, positive_ret, negative_ret):
    self.positive_ret = positive_ret
    self.negative_ret = negative_ret

  # sess: Session used to load images
  # num: The batch size
  # ratio: The ratio of positive to negative images
  # Returns:
  # The left array with shape (?, 192, 192, 3)
  # The right array with shape (?, 192, 192, 3)
  # The ground truth array with shape (?, 1)
  def get_random_pairs(self, sess, num, ratio):
    pos_n = int(math.floor(ratio * num))
    pos_r, pos_l = self.positive_ret.get_random_pairs(sess, pos_n)
    neg_r, neg_l = self.negative_ret.get_random_pairs(sess, num - pos_n)
    gt = np.concatenate((np.ones((pos_n,1)), np.zeros((num-pos_n,1))), axis=0)
    merge_r = np.vstack((pos_r, neg_r))
    merge_l = np.vstack((pos_l, neg_l))
    rng_state = np.random.get_state()
    np.random.shuffle(gt)
    np.random.set_state(rng_state)
    np.random.shuffle(merge_r)
    np.random.set_state(rng_state)
    np.random.shuffle(merge_l)
    return merge_r, merge_l, gt

  # Get combined length of retrivers
  # Return: length
  def get_length(self):
    return self.positive_ret.get_length() + self.negative_ret.get_length()
    
  length = property(get_length)
