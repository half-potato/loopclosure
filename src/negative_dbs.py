from SimpleDB import SimpleDB
from PlacesDB import PlacesDB
from SplitDB import SplitDB
from FlipDB import FlipDB
from CNNDB import CNNDB
from GTADB import GTADB
import scenenn

def make_n_cnn(folder, stats):
  return [CNNDB("raw/CNNImgRetrieval", folder, stats, .000001)]

def make_n_places(folder, stats):
  print("Places")
  if folder == "test":
    return []
  if folder == "train" or folder == "val":
    return [PlacesDB("../places/data/places365_train_standard.txt",
      "../places/data/train_256", stats, 0.5, False)]
  #if folder == "val":
    #return [PlacesDB("../places/data/places365_val.txt",
      #"../places/data/val_256", stats, 0.5, False)]

def make_n_split(set_name, stats):
  SPLIT_PROB = 1
  return [
    SplitDB("indexes/t27_big_file_index.mat", "raw/Tokyo247",
                 stats, 480, 640, SPLIT_PROB, 
                 "indexes/"+set_name+"/t27_neg_split.mat"),
    #SplitDB("indexes/pitts_file_index.mat", "raw/Pitts250/data_large",
    #             stats, 480, 640, SPLIT_PROB,
    #             "indexes/"+set_name+"/pitts_neg_split.mat"),
    SplitDB("indexes/ttm_file_index.mat", "raw/TokyoTimeMachine/data_large",
                 stats, 480, 640, SPLIT_PROB,
                 "indexes/"+set_name+"/ttm_neg_split.mat"),
  ]

def make_n_flip(set_name, stats):
  FLIP_PROB = 0.5
  return [
    FlipDB("indexes/t27_file_index.mat", "raw/Tokyo247",
           stats, 480, 640, FLIP_PROB,
           "indexes/"+set_name+"/t27_neg_flip.mat"),
    #FlipDB("indexes/pitts_file_index.mat", "raw/Pitts250/data",
    #       stats, 480, 640, FLIP_PROB,
    #       "indexes/"+set_name+"/pitts_neg_flip.mat"),
    FlipDB("indexes/ttm_file_index.mat", "raw/TokyoTimeMachine/data",
           stats, 480, 640, FLIP_PROB,
           "indexes/"+set_name+"/ttm_neg_flip.mat"),
  ]

def make_n_rgbd(folder, stats):
  return [SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg1_room_neg.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_large_with_loop_neg.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_pioneer_slam2_neg.txt",
            "raw/RGBD/data/", stats, 1),
          #SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg3_long_office_household_neg.txt",
            #"raw/RGBD/data/"), stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_large_no_loop_neg.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_pioneer_slam_neg.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_pioneer_slam3_neg.txt",
            "raw/RGBD/data/", stats, 1)]

def farpos(folder, stats):
  prob = 0.1
  return [SimpleDB("indexes/%s/pitts_neg_farpos.mat" % folder,
            "raw/Pitts250/data", stats, prob),
          SimpleDB("indexes/%s/t27_neg_farpos.mat" % folder, 
            "raw/Tokyo247", stats, prob),
          SimpleDB("indexes/%s/ttm_neg_farpos.mat" % folder, 
            "raw/TokyoTimeMachine/data", stats, prob)]

def make_n_opposites(folder, stats):
  prob = 0.005
  return [#SimpleDB(("indexes/%s/pitts_neg_opposites.mat" % folder, 
          #  "raw/Pitts250/data"), stats, prob),
          SimpleDB("indexes/%s/t27_neg_opposites.mat" % folder, 
            "raw/Tokyo247", stats, prob),
          SimpleDB("indexes/%s/ttm_neg_opposites.mat" % folder, 
            "raw/TokyoTimeMachine/data", stats, prob)]

def closepairs(folder, stats):
  prob = 1
  return [#SimpleDB("indexes/%s/pitts_neg_closepairs.mat" % folder, 
          #  "raw/Pitts250/data", stats, prob),
          SimpleDB("indexes/%s/t27_neg_closepairs.mat" % folder, 
            "raw/Tokyo247", stats, prob),
          SimpleDB("indexes/%s/ttm_neg_closepairs.mat" % folder, 
            "raw/TokyoTimeMachine/data", stats, prob)]

def make_n_scenenn(folder, stats):
  ret = []
  for i in scenenn.SCENENN_NUMS:
    i_p = scenenn.SCENENN_INDEX_PATH % (folder, i, "neg")
    db = SimpleDB(i_p, scenenn.SCENENN_IMAGE_PATH, stats, 1)
    ret.append(db)
  return ret

def make_gtav(folder, stats):
  return [
      GTADB("raw/GTAV", "indexes/gta_clusters.npy", stats, 1, False)]
