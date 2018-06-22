import numpy as np
import math
import cv2
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

MAX_DIST = 0.010 #kilometers
MIN_DIST = 0.001
MAX_ALT_DIST = 0.002
EST_DIST_TO_OBJECT = .002 #kilometers
SLICES = 12
SLICE_WIDTH = (2*math.pi)/SLICES
NEGATIVE_SLICES = 8

UTM_ZONE = 54
UTM_LETTER = "N"

# The corresponding images for 1-12
# The base array is rotated to match the angle between the two panoramics
# Details in the paper
# Returns an array of size 12 to match range(12)
def get_pano_mapping(dist, angle):
  #       0  1  2       3       4       5  6  7       8       9       10 11
  #base = [1, np.nan, np.nan, np.nan, np.nan, 6, 7, np.nan, np.nan, np.nan, np.nan, 12]
  base = [1, 2, np.nan, np.nan, 5, 6, 7, 8, np.nan, np.nan, 11, 12]
  #base = [1, np.nan, np.nan, np.nan, 2, 6, 7, 11, np.nan, np.nan, np.nan, 12]
  # add angle
  shift = int(math.ceil((math.pi - angle) / SLICE_WIDTH)) % SLICES
  shifted = np.add(np.add(base, shift-1) % SLICES, 1)
  na = np.nan_to_num(shifted)
  shifted = np.roll(na, shift)
  # replace nan with zero and add 1
  return shifted.tolist()
  #return np.nan_to_num(base)

# Like pano mappings but gives the full inverse set of images that don't overlap
# Returns: An array of pano mappings
def get_negative_mappings(dist, angle):
  shift = int(math.ceil((math.pi-angle) / SLICE_WIDTH)) % SLICES
  def shif(base, shift):
    shifted = np.add(np.add(base, shift-1) % SLICES, 1)
    na = np.nan_to_num(shifted)
    shifted = np.roll(na, shift)
    # replace nan with zero and add 1
    # add angle
    return shifted.tolist()
  mappings = []
  mappings.append(shif([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  1,  2], shift))
  mappings.append(shif([ 4,  5,  6,  7,  8,  9, 10, 11, 12,  1,  2,  3], shift))
  mappings.append(shif([ 5,  6,  7,  8,  9, 10, 11, 12,  1,  2,  3,  4], shift))
  mappings.append(shif([ 6,  7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5], shift))
  mappings.append(shif([ 7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6], shift))
  mappings.append(shif([ 8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7], shift))
  mappings.append(shif([ 9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8], shift))
  mappings.append(shif([10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8,  9], shift))
  mappings.append(shif([11, 12,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10], shift))
  return mappings

# Returns:
# dx: a matrix with shape (coords.shape[0], coords.shape[0]) with the difference
# between the positions along the x axis
# dy: a matrix with shape (coords.shape[0], coords.shape[0]) with the difference
# between the positions along the y axis
# dist: a matrix with shape (coords.shape[0], coords.shape[0]) with the distance
# between the positions
# diff_alt: a matrix with shape (coords.shape[0], coords.shape[0]) with the altitude
# distance between the positions
# Coords is a ?x2 mat of the coords
def get_diff_and_distance(coords):
  lat = np.radians(coords[:,0,None])
  lon = np.radians(coords[:,1,None])

  # Diff between all values
  lat_lr = np.ones(lat.shape).dot(lat.transpose()) 
  lat_td = lat.dot(np.ones(lat.shape).transpose())
  diff_lat = lat_lr - lat_td
  lon_lr = np.ones(lon.shape).dot(lon.transpose())
  lon_td = lon.dot(np.ones(lon.shape).transpose())
  diff_lon = lon_lr - lon_td

  ## Haversine to get dist in km
  r = 6367 # Earth r in km
  a = np.sin(diff_lat/2.0)**2 + np.cos(lat_lr) * np.cos(lat_td) * np.sin(diff_lon/2.0)**2
  dist = r * 2 * np.arcsin(np.sqrt(a))
  diff_alt = None
  if coords.shape[1] == 3:
    print("Using altitude")
    alt = coords[:,2,None]
    alt_lr = np.ones(alt.shape).dot(alt.transpose()) 
    alt_td = alt.dot(np.ones(alt.shape).transpose())
    diff_alt = np.abs(alt_lr - alt_td)
  # Convert long and lat diff to km
  dx = r * (np.cos(lat_lr) * np.cos(lon_lr) - np.cos(lat_td) * np.cos(lon_td))
  dy = r * (np.cos(lat_lr) * np.sin(lon_lr) - np.cos(lat_td) * np.sin(lon_td))
  return dx, dy, dist, diff_alt

# Creates a matrix of the angle between all coordinates and assigns 0 to points
# that are too far apart
# These inputs can be calculated by using get_diff_and_distance
# dx: a matrix with shape (coords.shape[0], coords.shape[0]) with the difference
# between the positions along the x axis
# dy: a matrix with shape (coords.shape[0], coords.shape[0]) with the difference
# between the positions along the y axis
# dist: a matrix with shape (coords.shape[0], coords.shape[0]) with the distance
# between the positions
# alt_dist: The difference in altitude between points
# max_dist: The max distance to use
# min_dist: The min distance to use
def loop_closure_img_index(dx, dy, dist, alt_dist, max_dist, min_dist):
  # ind starts at 0
  def slice_unit_circle(ind, slices):
    sw = (2 * math.pi) / slices
    start = ind * sw
    end = ind * sw + sw
    return start, end

  # Start and end of each slice of the panoramic
  mat = np.zeros(dist.shape)
  angles = np.arctan2(dy, dx) + math.pi
  close_points = (dist > min_dist) * (dist < max_dist) * \
                 (np.ones(mat.shape) - np.eye(mat.shape[0]))
  if alt_dist is None:
    alt_cap = (alt_dist < MAX_ALT_DIST)
    close_points = close_points * alt_cap
  return close_points * angles

# Index is matrix of the angles between points
# Indices is the matrix to look up the index of the image path using x and y
# Dist is the matrix to look up the distance between x and y
# db is the array of paths indexed by indices
def reverse_n_lookup_index(index, indices, dist, x, y, db):
  d = dist[x, y]
  # pick pano offsets at discrete distances
  angle1 = index[x,y]
  circle1s = get_negative_mappings(d, angle1)
  start_ind1 = indices[x]
  start_ind2 = indices[y]
  pairs = []
  for circle1 in circle1s:
    # remove zero entries from lists
    match1 = np.transpose(np.nonzero(circle1)).transpose()
    u_ind1 = [circle1[int(i)]-1 for i in match1.transpose()]

    ind1 = np.add(u_ind1, start_ind2)
    m_ind1 = np.add(match1, start_ind1)

    pairs1 = np.vstack([ind1, m_ind1]).transpose()
    pairs.extend(pairs1)
  return pairs
  
# Index is matrix of the angles between points
# Indices is the matrix to look up the index of the image path using x and y
# Dist is the matrix to look up the distance between x and y
# db is the array of paths indexed by indices
def reverse_lookup_index(index, indices, dist, x, y, db):
  d = dist[x, y]
  # pick pano offsets at discrete distances
  angle1 = index[x,y]
  circle1 = get_pano_mapping(d, angle1)
  start_ind1 = indices[x]
  start_ind2 = indices[y]

  # remove zero entries from lists
  match1 = np.transpose(np.nonzero(circle1)).transpose()
  u_ind1 = [circle1[int(i)]-1 for i in match1.transpose()]

  ind1 = np.add(u_ind1, start_ind2)
  m_ind1 = np.add(match1, start_ind1)

  pairs1 = np.vstack([ind1, m_ind1]).transpose()
  return pairs1

def process(coords):
  dx, dy, dist, alt_dist = get_diff_and_distance(coords)
  index = loop_closure_img_index(dx, dy, dist, alt_dist, MAX_DIST, MIN_DIST)
  nz = np.transpose(np.nonzero(index))
  return dist, index, nz

# Filters images that are too dark or uninteresting
# l: The array of image paths
# path: The base dir of the images
# thres: The threhold to filter dark images. 0-255
# std_thres: The threshold to filter uninteresting images.
# slices: The amount of images to take out when once of the points of that pano
# is removed. This should match the amount of images in a pano
def filter_darks(l, path, thres, std_thres, slices=SLICES):
  with tf.variable_scope("jpeg_decode"):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([256, 256])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
  sess = tf.Session()
  delete_i = []
  for i, row in enumerate(l):
    if type(row) == type(np.array([])):
      row = row[0]
    i_path = os.path.join(path, row).strip()
    im = gfile.FastGFile(i_path, "rb").read()
    img = sess.run(resized_image, {jpeg_data:im})
    if np.count_nonzero(img) <= 0:
      print(i_path + " is null")
      continue
    if np.mean(img) < thres or np.std(img) < std_thres:
      start = i - (i % slices)
      print("deleting")
      print(start)
      if not start in delete_i:
        for j in range(slices):
          delete_i.append(start + j)
  print("Removing " + str(len(delete_i)) + " images")
  print(l.shape)
  return np.delete(l, delete_i, axis=0)

# Calculates the distance between a set of coords
# Coordinates come in sets of 12 because of the number of images per a pano, so
# the coords are filtered using np unique and the index back to the image index
# from the new arrays is given
# ind: The array from the unique coords back to the start of their image index
# index: A matrix of the angle between coords within the distance threshold
# dist: A matrix of the distance between coords
# nz: The points that are actually close to each other. Those that are too far
# have a value of 0 in the index so these are the non zero elements of the index
def calculate(coords):
  crds, ind = np.unique(coords, return_index=True, axis=0)
  dist, index, nz = process(crds)
  return index, ind, dist, nz

# Saves the calculations
# ind: The array from the unique coords back to the start of their image index
# index: A matrix of the angle between coords within the distance threshold
# dist: A matrix of the distance between coords
# nz: The points that are actually close to each other. Those that are too far
# have a value of 0 in the index so these are the non zero elements of the index
def save(index, ind, dist, nz, af, path):
  with open(path, "wb") as f:
    np.savez(f, index=index, ind=ind, dist=dist, nz=nz, af=af)

# Saves the calculations
# ind: The array from the unique coords back to the start of their image index
# index: A matrix of the angle between coords within the distance threshold
# dist: A matrix of the distance between coords
# nz: The points that are actually close to each other. Those that are too far
# have a value of 0 in the index so these are the non zero elements of the index
# af: the array of image paths
def load_calculations(path):
  with open(path, "r") as f:
    arr = np.load(f)
    return arr["index"], arr["ind"], arr["dist"], arr["nz"], arr["af"]

# Quick wrapper to save a matrix
def save_index(index, path):
  with open(path, "wb") as f:
    np.save(f, index)

# Quick wrapper to load a matrix
def load_index(path):
  with open(path, "r") as f:
    return np.load(f)
