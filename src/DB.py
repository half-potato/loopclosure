import numpy as np
import random
import os
import image_util
from tensorflow.python.platform import gfile

# A DB is a comparmentalized way of accessing a dataset
# Often DBs also subclass Named DB to allow for easy identification
# It is accessed by the Retriever class to make a positive dataset, which is then
# used to created a Mixed Retriever, which has both positive and negative
# examples
class DB:
  # Chance_multi is a variable stored in each database as a helper variable for
  # the retriever class when sampling the dataset. The chance multi is multiplied
  # with the size of the dataset to adjust the chance of it being picked
  # Probability of this database being picked is:
  # chance_multi*self.get_length() / sum(all other database lengths * their chance_multi)
  # Image stats are the stats used to normalize the data
  # Image option is either alter, unaltered, or distort
  # Alter crops the image by cropping the image by up to 10%, scaling the image 
  # by up to 5%, and brightening the image by up to 5%
  # Unaltered does not make changes to the image
  # Distort will make distortions that might make the images no longer correspond
  # It might flip the image, crop by up to 20%, scale by up to 20%, and brighten
  # by up to 5%
  def __init__(self, image_option, image_stats, chance_multi):
    self.chance_multi = chance_multi
    def alter():
      self.input, self.output = \
                image_util.add_input_distortions(False, 10, 5, 5, image_stats.size, \
                                           image_stats.size, image_stats.depth, \
                                           image_stats.mean, image_stats.std)
    def unaltered():
      self.input, self.output = \
                image_util.add_basic_ops(image_stats.size, image_stats.size, \
                                         image_stats.depth, image_stats.mean, \
                                         image_stats.std)
    def distort():
      self.input, self.output = \
                image_util.add_input_distortions(True, 20, 20, 5, image_stats.size, \
                                           image_stats.size, image_stats.depth, \
                                           image_stats.mean, image_stats.std)
    options = {
      "alter": alter,
      "unaltered": unaltered,
      "distort": distort,
    }
    options[image_option]()

  # MUST BE OVERRIDDEN
  # Example: raw/Tokyo247/data/image1.jpg, raw/Tokyo247/data/image1.jpg
  # Returns the paths to a pair's images
  # Indexes to pairs should be laid out so that all pairs are covered and no pairs
  # are skipped
  def get_pair_path(self, i):
    return None, None
  
  # MUST BE OVERRIDDEN
  # Returns the length of the database
  # Can be accessed using db.length
  def get_length(self):
    return 0

  length = property(get_length)

  # Given a path and a session, load the image
  # The path is relative to the base dir of the repo
  # The session is for loading the image using tensorflow. OpenCV loads jpegs
  # differently
  # Return a numpy array of dims (1, 192, 192, 3) (192 is the default images size)
  # 1, Width, Height, Depth
  # Requires self.output and self.input to be declared, which are the tensor input and outputs
  # To jpeg image loading. Any sort of alteration should be done here
  def load_image(self, path, sess):
    if os.path.isfile(path):
      im = gfile.FastGFile(path, "rb").read()
      try:
        out = sess.run(self.output, {self.input: im})
      except Exception as e:
        print(e)
        print(path)
        out = None
      return out
    else:
      print("Image: %s does not exist" % path)
      return None

  # Takes the index of the image pairs in the dataset
  # The session is for loading the image using tensorflow. OpenCV loads jpegs
  # differently
  # Return two images of shape (1, 192, 192, 3)
  def get_image_pair(self, i, sess):
    p1, p2 = self.get_pair_path(i)
    p1 = str(np.squeeze(p1)).strip()
    p2 = str(np.squeeze(p2)).strip()
    return self.load_image(p1, sess), self.load_image(p2, sess)

  # Retrieves a random pair in the dataset
  # Recursively checks for a non None pair up to 5 times
  # Return two images of shape (1, 192, 192, 3)
  def get_random_pair(self, sess, level=0):
    if level > 5:
      print("Failed to retrieve pair")
      return 1, 1
    i = random.randrange(self.get_length())
    pair1, pair2 = self.get_image_pair(i, sess)
    if type(pair1) == type(None) or type(pair2) == type(None):
      return self.get_random_pair(sess, level+1)
    return pair1, pair2

  # Gets a batch of random images in the dataset
  # Returns two array: the left images and the right images
  # The shapes are (?, 192, 192, 3) where the ? is the batch size
  def batch(self, sess, batch_i, batch_size):
    start_i = batch_i * batch_size
    size = min(self.get_length() - start_i, batch_size)
    r_bat = []
    l_bat = []
    for i in range(size):
      r, l = self.get_random_pair(i, sess)
      r_bat.append(np.squeeze(r))
      l_bat.append(np.squeeze(l))
    r_bat = np.array(r_bat)
    l_bat = np.array(l_bat)
    return r_bat, l_bat
