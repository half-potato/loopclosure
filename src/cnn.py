import scipy.io
import numpy as np
import h5py
import os

# Converts the integer array to a file name
# base: the base path of the image folder
# cid: an array of int32s that represent the name
def cidToFileName(base, cid):
  letters = cid.astype(np.uint32).view("U1").astype(str)
  end = len(cid) - 1
  def arrstr(arr):
    out = ""
    for i in arr:
      out += str(i[0])
    return out
  return os.path.join(base, \
         arrstr(letters[end-1:end+1]), arrstr(letters[end-3:end-1]), \
         arrstr(letters[end-5:end-3]), arrstr(letters))
         
# path: The path to the matrix with name: retrieval-SfM-?k.mat
# img_path: The base path of the images
# val_or_train: "val" if validation set and "train" if training set
# Returns: 
# qpid: aligned image queries. (?, 2) of image index pairs
# clucids: (?, 2) of image paths and their cluster ids, respectively
def processMat(path, img_path, val_or_train):
  mat = h5py.File(path, "r")
  def processSet(dat):
    cids = np.array([cidToFileName(img_path, np.array(dat[cid[0]])) for cid in dat["cids"]]).transpose()
    cids = np.squeeze(cids)
    cluster = np.squeeze(dat["cluster"])
    pidxs = dat["pidxs"]
    qidxs = dat["qidxs"]
    qpid = np.hstack((pidxs, qidxs))
    clucids = np.vstack((np.array(cids).transpose(), cluster)).transpose()
    return qpid, clucids
  return processSet(mat[val_or_train])

# path: The path to the folder than contains the matrixes and "images" folder
# num: 120 or 30 for the different sets that CNN provides. Use 120 unless you
# know what you are doing
# val_or_train: "val" if validation set and "train" if training set
# Returns: 
# qpid: aligned image queries. (?, 2) of image index pairs
# clucids: (?, 2) of image paths and their cluster ids, respectively
def load_mats(path, num, val_or_train):
  cid_120_p = os.path.join(path, "retrieval-SfM-" + str(num) + "k-imagenames-clusterids.mat")
  mat_120_p = os.path.join(path, "retrieval-SfM-" + str(num) + "k.mat")

  cid_120 = scipy.io.loadmat(cid_120_p)
  qpid, clucids = processMat(mat_120_p, os.path.join(path, "images"), val_or_train)

  clusters = np.squeeze(cid_120["cluster"]).transpose()
  cid = np.squeeze(cid_120["cids"][0]).transpose()
  cluster_tab = np.hstack((clusters, cid))

  return qpid, clucids, cluster_tab

# Unused
# Saves the matrix data as a numpy file
def save(path, qpid, clucids, cluster_tab):
  with open(path, "wb") as f:
    np.savez(f, qpid=qpid, clucids=clucids, cluster_tab=cluster_tab)

# Unused
# Loads matrix data from a numpy file
def load(path):
  with open(path, "r") as f:
    mat = np.load(f)
    return mat["qpid"], mat["clucids"], mat["cluster_tab"]

# Extracts the positive pairs 120
# Used in gen.py
# val_or_train: "val" if validation set and "train" if training set
# Returns: (?, 2) array of image file name pairs
def gen_pairs(val_or_train):
  path = "raw/CNNImgRetrieval"
  #qpid, clucids, cluster_tab = load("indexes/cnn_full_120.mat")
  qpid, clucids, cluster_tab = load_mats(path, 120, val_or_train)
  save("indexes/cnn_full_120.mat", qpid, clucids, cluster_tab)
  pairs = []
  for i in qpid:
    [i1, i2] = i.astype(int)
    p1 = clucids[i1, 0]
    p2 = clucids[i2, 0]
    pairs.append([p1, p2])
  return np.array(pairs)
  #with open("indexes/cnn_pos.mat", "wb") as f:
    #np.save(f, np.array(pairs))
  #print(qpid, clucids, cluster_tab)

if __name__ == "__main__":
  pairs = gen_pairs()

