import image_util
import sys
import numpy as np
sys.path.insert(0, "models")
import frozenv2
import frozen
import frozen_contrast
import deep
import contrast
import places_alexnet

DECAY = 0.9999
MOMENTUM = 0.0
EPSILON = 1e-10
MARGIN = 0.7

FROZEN_GRAPH = "pretrained_places_2.pb"

# CONSTANT IMAGE STATS
#STREETVIEW = image_util.ImageStats(size=192, depth=3,
#    mean=np.load("stats/nopitts_mean.npy"),
#    std=np.load("stats/nopitts_std.npy"))
STREETVIEW = image_util.ImageStats(
    size=192, depth=3, mean=105.403474097, std=71.1522369385)
STREETVIEW_CNN = image_util.ImageStats(
    size=192, depth=3, mean=105.403474097, std=71.1522369385)
CNN = image_util.ImageStats(size=192, depth=3,
    mean=np.load("stats/cnn_mean.npy"),
    std=np.load("stats/cnn_std.npy"))
GTA_CNN = image_util.ImageStats(size=192, depth=3,
    mean=np.load("stats/gta_cnn_mean.npy"),
    std=np.load("stats/gta_cnn_std.npy"))
CNN_RGBD = image_util.ImageStats(
    size=192, depth=3, mean=117.183452,    std=56.817754623)
ALEXNET_PLACES = image_util.ImageStats(
    size=227, depth=3, mean=116.28,        std=57.375)
PLACES = image_util.ImageStats(
    size=192, depth=3, mean=116.28,        std=57.375)
SIMPLE = image_util.ImageStats(size=192, depth=3,
    mean=np.load("stats/simple_mean.npy"),
    std=np.load("stats/simple_std.npy"))
GTA = image_util.ImageStats(size=192, depth=3,
    mean=np.load("stats/gta_mean.npy"),
    std=np.load("stats/gta_std.npy"))

DEFAULT_STATS = image_util.ImageStats(size=192, depth=3, mean=0, std=1)

# db name: stat
STATS = {
    "cnn": CNN,
    "cnn_nopitts": STREETVIEW_CNN,
    "nopitts": STREETVIEW,
    "cnnrgbd": CNN,
    "alexnet": ALEXNET_PLACES,
    "simple": SIMPLE,
    "places": PLACES,
    "gta": GTA,
    "gta_cnn": GTA_CNN,
    "cnn_gta": GTA_CNN,
}

#net_type, cutoff_layer, db_type, learning_rate, batch_size, vary_ratio
SETTINGS = [
  #("deep",            0, "cnn",         0.01, 400, False), # 0
  #("deep",            0, "cnn_nopitts", 0.01, 100, False), # 1
  #("frozen",         11, "cnn",         0.01, 100, False), # 2
  #("frozen",         11, "cnn_nopitts", 0.01, 100, False), # 3
  #("frozenv2",       11, "cnn",         0.01, 100, False), # 4
  #("frozenv2",       11, "cnn_nopitts", 0.01, 100, False), # 5
  #("contrast",        0, "cnn",         0.01, 100,   False), # 6
  #("contrast",        0, "cnn_nopitts", 0.01, 100,   False), # 7
  #("frozen_contrast",11, "cnn",         0.01, 100,   False), # 8
  #("frozen_contrast",11, "cnn_nopitts", 0.01, 100,   False), # 9
  #("deep",            0, "simple",      0.01, 100, False), # 10
  #("frozen",         11, "simple",      0.01, 100, False), # 11
  #("frozenv2",       11, "simple",      0.01, 100, False), # 12
  #("contrast",        0, "simple",      0.01, 100,   False), # 13
  #("frozen_contrast",11, "simple",      0.01, 100,   False), # 14
  #("frozen",         11, "places",      0.01, 100,   False), # 15
  #("frozen_euclid",11, "places",      0.01, 100,   False), # 16
  #("frozen_euclid",11, "simple",      0.01, 100,   False), # 16
  #("frozen_euclid",11, "places",      0.01, 100,   False), # 16
  #("frozen_euclid",11, "cnn",      0.01, 100,   False), # 16
  #("frozen_euclid",11, "nopitts",      0.01, 100,   False), # 16
  ("frozen_euclid",11, "gta_cnn",      0.01, 100,   False), # 16
  ("frozen_euclid",11, "gta",      0.01, 100,   False), # 16
  #("frozen_euclid",11, "rgbd_scenenn",      0.01, 100,   False), # 16
  #("places_alexnet",  5, "alexnet",     0.01, 100,   False), # 17
]

NET_TYPES = {
    "deep": deep,
    "frozen": frozen,
    "frozenv2": frozenv2,
    "contrast": contrast,
    #"frozen_contrast": frozen_contrast,
    "frozen_euclid": frozen_contrast,
    "places_alexnet": places_alexnet,
}

FILTERING_STREETVIEW_PARAMETERS = {
  # darkness threshold, standard dev threshold
  "ttm": (45, 20),
  "t27": (45, 10),
  "pitts": (37, 14),
}

def is_contrast(net_type):
  return net_type in ["contrast", "frozen_contrast", "places", "places_alexnet", "frozen_euclid"]

def is_frozen(net_type):
  return net_type in ["frozen", "frozenv2", "frozen_contrast", "frozen_euclid"]
