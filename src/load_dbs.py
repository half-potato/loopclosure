import tensorflow as tf
from CNNDB import CNNDB
from Retriever import Retriever
from MixedRetriever import MixedRetriever

import constants
import positive_dbs
import negative_dbs

def load_dbs(dbtype, is_viewing=False):
  print("Loading db %s" % dbtype)

  # Saves your life
  if is_viewing:
    stats = constants.DEFAULT_STATS
    # Creates a block of equal signs
    block = "\n".join(["".join(["="*40])]*10)
    print("%s\nYOU ARE VIEWING THE DATASET. YOU ARE \nUSING A ZERO MEAN STAT.\n%s" %
          (block, block))
  else:
    stats = constants.STATS[dbtype]

  with tf.variable_scope("img_processing"):
    # Create positive database
    # adds each type of dataset depending on whether 
    def make_db(folder):
      db = []
      if "gta" in dbtype:
        db.extend(positive_dbs.make_gtav(folder, stats))
      if "cnn" in dbtype:
        db.extend(positive_dbs.make_cnn(folder, stats))
      if "nopitts" in dbtype:
        db.extend(positive_dbs.make_ttm(folder, stats))
        db.extend(positive_dbs.make_t27(folder, stats))
      if "rgbd" in dbtype:
        db.extend(positive_dbs.make_rgbd(folder, stats))
        db.extend(positive_dbs.make_scenenn(folder, stats))
      if dbtype == "simple":
        db = []
        db.extend(positive_dbs.make_ttm(folder, stats, "wp"))
        db.extend(positive_dbs.make_pitts(folder, stats, "wp"))
        db.extend(positive_dbs.make_t27(folder, stats, "wp"))
      if dbtype == "places":
        db = []
        db.extend(positive_dbs.make_places(folder, stats))
      return db

    # Generate positives
    TRAIN_DBS = make_db("train")
    VAL_DBS = make_db("val")
    TEST_DBS = make_db("test")

    def make_n_db(folder):
      db = []
      if "gta" in dbtype:
        db.extend(negative_dbs.make_gtav(folder, stats))
      if "nopitts" in dbtype:
        db.extend(negative_dbs.make_n_split(folder, stats))
        db.extend(negative_dbs.make_n_flip(folder, stats))
        db.extend(negative_dbs.farpos(folder, stats))
        db.extend(negative_dbs.make_n_opposites(folder, stats))
        db.extend(negative_dbs.closepairs(folder, stats))
      if "rgbd" in dbtype:
        db.extend(negative_dbs.make_n_rgbd(folder, stats))
        db.extend(negative_dbs.make_n_scenenn(folder, stats))
      if folder == "test":
        return db
      if "cnn" in dbtype:
        db.extend(negative_dbs.make_n_cnn(folder, stats))
      if dbtype == "simple":
        db = []
        db.extend(negative_dbs.farpos(folder, stats))
      if dbtype == "places":
        db = []
        db.extend(negative_dbs.make_n_places(folder, stats))
      return db

    TRAIN_N_DBS = make_n_db("train")
    VAL_N_DBS = make_n_db("val")
    TEST_N_DBS = make_n_db("test")

    print("Positives")
    l = 0
    for i in TRAIN_DBS:
      print(i.name)
      print(i.length)
      l += i.length
    print("Negatives")
    for i in TRAIN_N_DBS:
      print(i.name)
      print(i.length)
    print(stats)
    print("Training with %i examples" % l)
    return TRAIN_DBS, VAL_DBS, TEST_DBS, TRAIN_N_DBS, VAL_N_DBS, TEST_N_DBS

def load_retrievers(dbtype, viewing=False):
  # INIT DATABASE RETRIEVERS
  TRAIN_DBS, VAL_DBS, TEST_DBS, TRAIN_N_DBS, VAL_N_DBS, TEST_N_DBS = \
            load_dbs(dbtype, viewing)
  train_pos = Retriever(TRAIN_DBS)
  train_neg = Retriever(TRAIN_N_DBS)
  train_ret = MixedRetriever(train_pos, train_neg)

  val_pos = Retriever(VAL_DBS)
  val_neg = Retriever(VAL_N_DBS)
  val_ret = MixedRetriever(val_pos, val_neg)

  test_pos = Retriever(TEST_DBS)
  test_neg = Retriever(TEST_N_DBS)
  test_ret = MixedRetriever(test_pos, test_neg)
  return train_ret, val_ret, test_ret
