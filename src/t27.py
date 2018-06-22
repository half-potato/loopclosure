import numpy as np
import calc
import os
import constants

INDEX_PATH = "indexes"

# Walks through images to produce file index
def create_index(path):
  all_files = []
  for root, dirs, files in os.walk(path):
    if len(files) == 24:
      zipper = np.array(np.sort(files)).reshape((12,2))
      for row in zipper:
        with open(os.path.join(root, row[0])) as f:
          l = f.readline().split(",")
          lat = float(l[0])
          lon = float(l[1])
          alt = float(l[2])
          all_files.append([os.path.join(root, row[1]), lat, lon, alt])
  af = np.array(all_files)
  names = af[:,0]
  names = [i[13:].strip() for i in names]
  af[:,0] = names
  return af

# Used in gen.py
# Generates the streetview pairs between locations
def generate_scratch():
  af = create_index("raw/Tokyo247/data")
  #af = load_index(os.path.join(INDEX_PATH, "t27_file_index_alt.mat"))
  #print((af))
  dark_thres, std_thres = constants.FILTERING_STREETVIEW_PARAMETERS["ttm"]
  af = calc.filter_darks(af, "raw/Tokyo247", dark_thres, std_thres)
  calc.save_index(af[:,0:3], os.path.join(INDEX_PATH, "t27_file_index.mat"))
  calc.save_index(af, os.path.join(INDEX_PATH, "t27_file_index_alt.mat"))
  coords, ind = np.unique(af[:,1:4].astype(float), return_index=True, axis=0)
  dist, index, nz = calc.process(coords)
  return index, ind, dist, nz, af[:,0]

def main():
  af = create_index("raw/Tokyo247/data")
  af = calc.filter_darks(af, "raw/Tokyo247", 25, 0, 12)
  #calc.save_index(af[:,0:3], os.path.join(INDEX_PATH, "t27_file_index.mat"))
  #calc.save_index(af, os.path.join(INDEX_PATH, "t27_file_index_alt.mat"))
  af = calc.load_index(os.path.join(INDEX_PATH, "t27_file_index_alt.mat"))
  coords, ind = np.unique(af[:,1:4].astype(float), return_index=True, axis=0)
  dist, index, nz = calc.process(coords)

  calc.save(index, ind, dist, nz, af[:,0], os.path.join(INDEX_PATH, "t27_calc.mat"))
  index, ind, dist, nz, af = calc.load_calculations(os.path.join(INDEX_PATH, "t27_calc.mat"))
  calc.test(index, ind, dist, af, nz, "raw/Tokyo247")

if __name__ == "__main__":
  main()

