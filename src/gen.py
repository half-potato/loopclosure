import gen_negative_set
import os
import numpy as np
from split import split
import ttm
import t27
import pitts
import calc
import cnn

# All of these functions generate the indexes for the datasets

IMG_FOLDERS = [
  "raw/Pitts250/data",
  "raw/Tokyo247",
  "raw/TokyoTimeMachine/data",
]
  
FOLDERS = {
  "train": "indexes/train",
  "val": "indexes/val",
  "test": "indexes/test",
}

def gen_negatives():
  for i in ["pitts", "t27", "ttm"]:
    gen_opposites(i)
    gen_closepairs(i)
    gen_farpos(i)

def gen_opposites(set_name):
  name = "%s_neg_opposites.mat" % set_name
  dat = np.array(gen_negative_set.opposites_diff_pos("indexes/%s_calc.mat" % set_name))
  train_file = os.path.join(FOLDERS["train"], name)
  val_file = os.path.join(FOLDERS["val"], name)
  test_file = os.path.join(FOLDERS["test"], name)
  split(dat, train_file, val_file, test_file, .05, .10)

def gen_closepairs(set_name):
  name = "%s_neg_closepairs.mat" % set_name
  dat = np.array(gen_negative_set.closepairs("indexes/%s_file_index.mat" % set_name))
  train_file = os.path.join(FOLDERS["train"], name)
  val_file = os.path.join(FOLDERS["val"], name)
  test_file = os.path.join(FOLDERS["test"], name)
  split(dat, train_file, val_file, test_file, .05, .10)

def gen_farpos(set_name):
  name = "%s_neg_farpos.mat" % set_name
  dat = np.array(gen_negative_set.far_pos("indexes/%s_calc.mat" % set_name))
  train_file = os.path.join(FOLDERS["train"], name)
  val_file = os.path.join(FOLDERS["val"], name)
  test_file = os.path.join(FOLDERS["test"], name)
  split(dat, train_file, val_file, test_file, .05, .10)

def gen_positive(index, ind, dist, nz, af):
  print("Reverse lookup")
  pairs = []
  for i in nz:
    pairs.extend(calc.reverse_lookup_index(index, ind, dist, i[0], i[1], af))
  pairs = np.array(pairs)
  return pairs

def gen_split():
  def gen(index_path, name):
    with open(index_path, "r") as f:
      leng = np.load(f).shape[0]
    split(np.expand_dims(np.array(range(leng)), 1),
          "indexes/train/"+name+"_neg_split.mat",
          "indexes/val/"+name+"_neg_split.mat",
          "indexes/test/"+name+"_neg_split.mat",
          .05, .10)

  gen("indexes/pitts_file_index.mat", "pitts")
  gen("indexes/t27_big_file_index.mat", "t27")
  gen("indexes/ttm_file_index.mat", "ttm")

def gen_flips():
  def gen(index_path, name):
    with open(index_path, "r") as f:
      leng = np.load(f).shape[0]
    split(np.expand_dims(np.array(range(leng)), 1),
          "indexes/train/"+name+"_neg_flip.mat",
          "indexes/val/"+name+"_neg_flip.mat",
          "indexes/test/"+name+"_neg_flip.mat",
          .05, .10)

  gen("indexes/pitts_file_index.mat", "pitts")
  gen("indexes/t27_file_index.mat", "t27")
  gen("indexes/ttm_file_index.mat", "ttm")

def gen_within_pano():
  def gen(index_path, name):
    with open(index_path, "r") as f:
      leng = np.load(f).shape[0]
    split(np.expand_dims(np.array(range(leng)), 1),
          "indexes/train/"+name+"_pos_prox.mat",
          "indexes/val/"+name+"_pos_prox.mat",
          "indexes/test/"+name+"_pos_prox.mat",
          .05, .10)

  gen("indexes/pitts_file_index.mat", "pitts")
  gen("indexes/t27_file_index.mat", "t27")
  gen("indexes/ttm_file_index.mat", "ttm")

def gen_positives():
  print("Generating TTM")
  split(gen_positive(*ttm.generate_scratch()),
                     "indexes/train/ttm_pos.mat",
                     "indexes/val/ttm_pos.mat",
                     "indexes/test/ttm_pos.mat",
                     .05, .10)
  print("Generating T27")
  split(gen_positive(*t27.generate_scratch()),
                     "indexes/train/t27_pos.mat",
                     "indexes/val/t27_pos.mat",
                     "indexes/test/t27_pos.mat",
                     .05, .10)
  print("Generating Pitts")
  split(gen_positive(*pitts.generate_scratch()),
                     "indexes/train/pitts_pos.mat",
                     "indexes/val/pitts_pos.mat",
                     "indexes/test/pitts_pos.mat",
                     .05, .10)

def gen_cnn():
  l = cnn.gen_pairs("train")
  split(l,
        "indexes/train/cnn_pos.mat",
        "indexes/val/cnn_pos.mat",
        "indexes/test/cnn_pos.mat",
        .00, .10)
  split(cnn.gen_pairs("val"),
        "indexes/train/cnn_pos.mat",
        "indexes/val/cnn_pos.mat",
        "indexes/test/cnn_pos.mat",
        1.00, 0.0)

def gen_pos_gta():
  l = np.load("indexes/gta_pairs.npy")
  split(l,
        "indexes/train/gta_pos.npy",
        "indexes/val/gta_pos.npy",
        "indexes/test/gta_pos.npy",
        .05, .10)
<<<<<<< HEAD:scripts/gen.py

def gen_neg_cnn():
  for folder in ("train", "test", "val"):
    arr = gen_negative_set.inverse_set("indexes/"+folder+"/cnn_pos.mat")
    gen_negative_set.save("indexes/"+folder+"/cnn_pos.mat", arr)

=======
      
>>>>>>> 4573a363735de00c737f4d29dc492f124d993fcc:src/gen.py

if __name__ == "__main__":
  # Gen streetview
  #gen_positives()
  # Gen within pano
  #gen_within_pano()
  # Gen all negatives except for split (generated on the fly)
  #gen_negatives()
  #gen_flips()
  #gen_split()
<<<<<<< HEAD:scripts/gen.py
  gen_cnn()
  gen_neg_cnn()
=======
  #gen_cnn()
  gen_pos_gta()

>>>>>>> 4573a363735de00c737f4d29dc492f124d993fcc:src/gen.py
