import scipy.io
import numpy as np
import utm
import os
import math
import calc
import constants

UTM_ZONE = 17
UTM_LETTER = "T"

IMGS_PER_LOC = 24

INDEX_PATH = "indexes"

# Load mat and convert coords from utm
# Just open the mat if you want to understand why
def load(path):
  dat = scipy.io.loadmat(os.path.join(path, "pittsburgh_database_10586_utm.mat"))
  coords = np.array(dat["Cdb"].transpose())
  def convert(a):
    return utm.to_latlon(a[0], a[1], UTM_ZONE, UTM_LETTER)
  coords = np.apply_along_axis(convert, axis=1, arr=coords)
  print(coords)

  with open(os.path.join(path, "imagelist.txt")) as f:
    img_list = np.array(f.readlines())

  # 24 images per coord
  coord_r = np.repeat(coords, 24, axis=0)[0:len(img_list), :]
  t = (img_list, coord_r[:,0], coord_r[:,1])
  dat = np.vstack(t).transpose()

  return dat

# Used in gen.py
# Generates the streetview pairs between locations
def generate_scratch():
  #af = calc.load_index(os.path.join(INDEX_PATH, "pitts_file_index.mat"))
  #"""
  af = load("raw/Pitts250/groundtruth")
  # remove 12 rows every other 12 rows (removing pitch2)
  alt = np.empty(af.shape[0] / 12)
  alt[::2] = 0
  alt[1::2] = 1
  row12 = np.repeat(alt, 12, axis=0) * (np.array(range(af.shape[0])) + 1)
  af = np.delete(af, np.nonzero(row12), axis=0)
  # Filter
  dark_thres, std_thres = constants.FILTERING_STREETVIEW_PARAMETERS["pitts"]
  af = calc.filter_darks(af, "raw/Pitts250/data", dark_thres, std_thres)
  calc.save_index(af, os.path.join(INDEX_PATH, "pitts_file_index.mat"))

  index, ind, dist, nz = calc.calculate(af[:,1:3].astype(float))
  return index, ind, dist, nz, af[:,0]

def main():
  af = load("raw/Pitts250/groundtruth")
  # remove 12 rows every other 12 rows (removing pitch2)
  alt = np.empty(af.shape[0] / 12)
  alt[::2] = 0
  alt[1::2] = 1
  row12 = np.repeat(alt, 12, axis=0) * (np.array(range(af.shape[0])) + 1)
  af = np.delete(af, np.nonzero(row12), axis=0)

  af = calc.filter_darks(af, "raw/Pitts250/data", 17, 12)

  calc.save_index(af, os.path.join(INDEX_PATH, "pitts_file_index.mat"))
  #af = calc.load_index(os.path.join(INDEX_PATH, "pitts_file_index.mat"))

  #load or do calculations
  index, ind, dist, nz = calc.calculate(af[:,1:3].astype(float))
  calc.save(index, ind, dist, nz, af[:,0], os.path.join(INDEX_PATH, "pitts_calc.mat"))
  index, ind, dist, nz, af = calc.load_calculations(os.path.join(INDEX_PATH, "pitts_calc.mat"))

  calc.test(index, ind, dist, af, nz, "raw/Pitts250/data")

if __name__ == "__main__":
  main()

