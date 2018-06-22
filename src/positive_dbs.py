from SimpleDB import SimpleDB
from StreetViewDB import StreetViewDB
from WithinPanoDB import WithinPanoDB
from PlacesDB import PlacesDB
from GTADB import GTADB
import scenenn

def make_ttm(folder, stats, setting="sv_wp"):
  print("Streetview")
  ret = []
  if "sv" in setting:
    ret.append(
        StreetViewDB("indexes/"+folder+"/ttm_pos.mat",
                     "indexes/ttm_file_index.mat", "raw/TokyoTimeMachine/data",
                     stats, 1))
  else:
    ret.append(
        WithinPanoDB("indexes/ttm_file_index.mat","raw/TokyoTimeMachine/data",
                     "indexes/"+folder+"/ttm_pos_prox.mat",
                     stats, 0.75))
  return ret

def make_t27(folder, stats, setting="sv_wp"):
  print("Streetview")
  ret = []
  if "sv" in setting:
    ret.append(
        StreetViewDB("indexes/"+folder+"/t27_pos.mat",
                     "indexes/t27_file_index.mat", "raw/Tokyo247",
                     stats, 1))
  else:
    ret.append(
        WithinPanoDB("indexes/t27_file_index_alt.mat", "raw/Tokyo247",
                     "indexes/"+folder+"/t27_pos_prox.mat",
                     stats, 0.75))
  return ret

def make_pitts(folder, stats, setting="sv_wp"):
  print("Streetview")
  ret = []
  if "sv" in setting:
    ret.append(
        StreetViewDB("indexes/"+folder+"/pitts_pos.mat",
                     "indexes/pitts_file_index.mat", "raw/Pitts250/data",
                     stats, 1))
  else:
    ret.append(
      WithinPanoDB("indexes/pitts_file_index.mat", "raw/Pitts250/data",
                   "indexes/"+folder+"/pitts_pos_prox.mat",
                   stats, 0.75))
  return ret

def make_cnn(folder, stats):
  print("CNN")
  return [SimpleDB("indexes/"+folder+"/cnn_pos.mat", "", stats, 0.5)]

def make_places(folder, stats):
  print("Places")
  if folder == "test":
    return []
  if folder == "train" or folder == "val":
    return [PlacesDB("../places/data/places365_train_standard.txt",
      "../places/data/train_256", stats, 0.5, True)]
  #if folder == "val":
    #return [PlacesDB("../places/data/places365_val.txt",
      #"../places/data/val_256", stats, 0.5, True)]

def make_rgbd(folder, stats):
  print("RGBD")
  return [SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg1_room_pos.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_large_with_loop_pos.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_pioneer_slam2_pos.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg3_long_office_household_pos.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_large_no_loop_pos.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_pioneer_slam_pos.txt",
            "raw/RGBD/data/", stats, 1),
          SimpleDB("indexes/"+folder+"/rgbd_dataset_freiburg2_pioneer_slam3_pos.txt",
            "raw/RGBD/data/", stats, 1)]

def make_scenenn(folder, stats):
  ret = []
  for i in scenenn.SCENENN_NUMS:
    i_p = scenenn.SCENENN_INDEX_PATH % (folder, i, "pos")
    db = SimpleDB(i_p, scenenn.SCENENN_IMAGE_PATH, stats, 1)
    ret.append(db)
  return ret

def make_gtav(folder, stats):
  return [
      GTADB("raw/GTAV", "indexes/%s/gta_pos.npy" % folder, stats, 1, True)]
