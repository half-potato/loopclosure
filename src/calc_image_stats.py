from Retriever import Retriever
from MixedRetriever import MixedRetriever
from load_dbs import load_dbs
import tensorflow as tf
import numpy as np
import cv2
import random
import sys

def calc_image_stats(pos_db, neg_db, samples=None):
  ret = MixedRetriever(Retriever(pos_db), Retriever(neg_db))
  if samples is None:
    count = ret.get_length()
  else:
    count = samples

  ratio = 1./2
  sess = tf.Session()
  mean = np.zeros((192, 192, 3))
  std = np.zeros((192, 192, 3))
  k = 1
  n = 200
  # Calc mean
  print("Calculating mean")
  for i in xrange(0, ret.get_length(), ret.get_length() // count):
    r, l, g = ret.get_random_pairs(sess, n/2, ratio)
    mean = mean * k/(k+n) + (np.sum(r, axis=0) + np.sum(l, axis=0)) / (k+n)
    k+=1
  #cv2.imshow("mean", mean)
  #cv2.waitKey(0)
  k = 1
  # Calc std
  mean_repeat = np.expand_dims(mean, 0).repeat(n, 0)
  print("Calculating standard deviation")
  for i in xrange(0, ret.get_length(), ret.get_length() // count):
    r, l, g = ret.get_random_pairs(sess, n/2, ratio)
    arr = np.concatenate((r, l), axis=0)
    a = (np.sum(np.square(arr - mean_repeat), axis=0)) / (k+n)
    std = std * k/(k+n) + a / (k+n)
  #cv2.imshow("std", std)
  #cv2.waitKey(0)

  return mean, std


if __name__ == "__main__":
  db = sys.argv[1]
  print("Loading dbs")
  TRAIN_DBS, VAL_DBS, TEST_DBS, TRAIN_N_DBS, VAL_N_DBS, TEST_N_DBS = load_dbs(db, True)
  print("Calculating")
  mean, std = calc_image_stats(TRAIN_DBS, TRAIN_N_DBS, 10)
  with open("stats/%s_mean.npy" % db, "w+") as f:
    np.save(f, mean)
  with open("stats/%s_std.npy" % db, "w+") as f:
    np.save(f, std)
