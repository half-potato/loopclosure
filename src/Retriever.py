import numpy as np
import random

# Used in MixedRetriever to retriver pairs from a list of databases
class Retriever:
  # dbs: An array of objects that implement DB.py
  def __init__(self, dbs):
    self.dbs = dbs 
    # Array of unnormalized probabilities for each dataset
    self.chance_array = [(i.get_length()) * i.chance_multi for i in dbs]

  # index: index of image pair to retrieve
  # sess: Session to use to load image. 
  # Treats datasets as if they were concatenated to a single array
  # Returns: A pair of images with shape (1, 192, 192, 3)
  def get_image_pair(self, index, sess):
    for i in self.dbs:
      if index >= i.get_length():
        index -= i.get_length()
      else:
        return i.get_image_pair(index, sess)
    return None, None

  # Returns: The length of all of the datasets combined
  def get_length(self):
    length = 0
    for i in self.dbs:
      length += i.get_length()
    return length

  length = property(get_length)

  # Returns a random pair with each dataset getting a probabilty from the chance
  # array
  def get_random_pair(self, sess):
    index = random.randrange(int(sum(self.chance_array)))
    for i,v in enumerate(self.chance_array):
      if index > v:
        index -= v
      else:
        #print(self.dbs[i].name)
        multi = self.dbs[i].chance_multi
        index = int(multi * index)
        return self.dbs[i].get_random_pair(sess)

  # sess: The session used to load the images
  # num: The batch size
  # Returns: the right and left array of images with shape (?, 192, 192, 3)
  def get_random_pairs(self, sess, num):
    r_imgs = []
    l_imgs = []
    for i in range(num):
      r, l = self.get_random_pair(sess)
      r_imgs.append(np.squeeze(r))
      l_imgs.append(np.squeeze(l))
    r_imgs = np.array(r_imgs)
    l_imgs = np.array(l_imgs)
    return r_imgs, l_imgs
