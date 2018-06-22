import calc
import numpy as np
import sys

DIST_THRES = 0.070

# Gets all points that are within the distance threshold and returns pairs of
# images that don't face each other
def opposite_diff_pos(calc_path):
  # Get pairs facing opposite ways
  print(calc_path)
  index, ind, dist, nz, db_img = calc.load_calculations(calc_path)
  pairs = []
  for i in nz:
    pairs.extend(calc.reverse_lookup(index, ind, dist, i[0], i[1], db_img, True))
  return pairs

# Get pairs at the same location that aren't facing each other
def closepairs(image_index_path):
  with open(image_index_path, "r") as f:
    image_index = np.load(f)
  img_num = (image_index.shape[0] // 12 - 1) * 12
  pairs = []
  for i in range(img_num):
    pair_i = i % 12
    loc_i = i // 12 * 12

    left_i = (pair_i - 3) % 12
    right_i = pair_i
    pairs.append([image_index[left_i + loc_i,0], image_index[right_i + loc_i,0]])
  return pairs

# Gives all points that are further than the distance threshold
def far_pos(calc_path, samples=1000000):
  index, ind, dist, nz, db_img = calc.load_calculations(calc_path)
  pairs = []
  # Get pairs too far apart
  xy = np.transpose(np.nonzero(dist > DIST_THRES))
  for row in xy:
    a = ind[row[0]]
    b = ind[row[1]]
    pairs.append([db_img[a], db_img[b]])
  if samples >= len(pairs):
    return pairs
  pairs = np.array(pairs)
  ind = np.random.randint(pairs.shape[0], size=samples)
  return pairs[ind,:]

def inverse_set(index, samples=1000000):
  with open(index, "rb") as f:
    positive_pairs = np.load(f)
  if positive_pairs.shape[0] < positive_pairs.shape[1]:
    positive_pairs = positive_pairs.transpose()
  images = np.expand_dims(np.unique(positive_pairs), 1)
  print(images)
  all_pairs = np.hstack((np.repeat(images, images.shape[0], axis=0),
                         np.tile(images.transpose(), images.shape[0]).transpose()))
  print(all_pairs)

  nrows, ncols = all_pairs.shape
  dtype={'names':['f{}'.format(i) for i in range(ncols)],
         'formats':ncols * [all_pairs.dtype]}

  C = np.intersect1d(all_pairs.view(dtype), positive_pairs.view(dtype))

  # This last bit is optional if you're okay with "C" being a structured array...
  C = C.view(all_pairs.dtype).reshape(-1, ncols)
  if samples >= C.shape[0]:
    return C
  ind = np.random.randint(C.shape[0], size=samples)
  return pairs[ind,:]
