import utm
import numpy as np
import scipy.io
import calc
import os
import constants

UTM_ZONE = 54
UTM_LETTER = "N"

INDEX_PATH = "indexes"

# first 3 db, second 3 query
# returns these as an array of mats
# img dir, utm, timestamp
# string, array of x, y, int
# 12 of each to every location
# Just open the mat if you want to understand why
def process_mat(mat_name):
  mat = scipy.io.loadmat(mat_name)["dbStruct"][0,0]
  return tuple([mat[i + 1].squeeze() for i in range(6)])

# Converts utm coords to normal coords
def get_locations(db_utm):
  coords, ind, cnt = np.unique(db_utm.transpose(), axis=0, return_counts=True, return_index=True)
  print(coords.shape)
  # Convert UTM to Lat and Long
  def convert(a):
    return utm.to_latlon(a[0], a[1], UTM_ZONE, UTM_LETTER)
  coords = np.apply_along_axis(convert, axis=1, arr=coords)
  return coords, ind, cnt

# Used in gen.py
# Generates the streetview pairs between locations
def generate_scratch():
  val = os.path.join("raw/TokyoTimeMachine/", "tokyoTM_val.mat")
  dat_db_img, dat_db_utm, dat_db_stp, dat_qy_img, dat_qy_utm, dat_qy_stp \
      = process_mat(val)
  coords, ind, cnt = get_locations(dat_db_utm)
  db_img = np.array([str(i[0]) for i in dat_db_img])

  # convert coords again. I don't know anymore
  def convert(a):
    return utm.to_latlon(a[0], a[1], UTM_ZONE, UTM_LETTER)
  coords = np.apply_along_axis(convert, axis=1, arr=dat_db_utm.transpose())
  both = np.hstack((np.expand_dims(db_img, 1), coords[:,0,None], coords[:,1,None]))
  dark_thres, std_thres = constants.FILTERING_STREETVIEW_PARAMETERS["ttm"]
  both = calc.filter_darks(both, "raw/TokyoTimeMachine/images", dark_thres, std_thres)
  print("Done filtering")
  calc.save_index(both, os.path.join(INDEX_PATH, "ttm_file_index.mat"))

  filt_coords = np.squeeze(both[:,1:,None]).astype(float)
  u_coords, ind = np.unique(filt_coords, axis=0, return_index=True)
  db_img = both[:,0,None]
  dist, index, nz = calc.process(u_coords)
  return index, ind, dist, nz, db_img

def main():
  base_path = "raw/TokyoTimeMachine/"
  train = os.path.join(base_path, "ttm_train_index.mat")
  val = os.path.join(base_path, "ttm_val_index.mat")

  index, ind, dist, nz, db_img = gen_calcs(val, "indexes/ttm_calc.mat")
  calc.test(index, ind, dist, db_img, nz, "raw/TokyoTimeMachine/images")

if __name__ == "__main__":
  main()

