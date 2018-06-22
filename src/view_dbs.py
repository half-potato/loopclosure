from load_dbs import load_dbs
from Retriever import Retriever
import tensorflow as tf
import cv2
import sys
import numpy as np

def main():
  TRAIN_DBS, VAL_DBS, TEST_DBS, TRAIN_N_DBS, VAL_N_DBS, TEST_N_DBS = \
      load_dbs(sys.argv[1], False)
  ret = Retriever(TRAIN_DBS)
  sess = tf.Session()
  cv2.namedWindow("image", cv2.WINDOW_NORMAL)
  cv2.resizeWindow("image", 600, 1200)
  l = 1000
  div = ret.get_length() / l
  pairs = []
  for i in range(l):
    img1, img2 = ret.get_image_pair(i*div + 1, sess)
    #img1, img2 = ret.get_random_pair(sess)
    img2 = np.squeeze(img2.astype(np.uint8))
    img1 = np.squeeze(img1.astype(np.uint8))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    #cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    #cv2.imwrite("image1.png", img1)
    #cv2.imwrite("image2.png", img2)

    ss = np.concatenate((img1, img2), axis=1)
    cv2.imshow("image", ss)
    val = cv2.waitKey(0)
    if val == 27:
      break
  cv2.destroyWindow("image")
  sess.close()
    #pairs.append(ss)
  #img = pairs[0]
  #for j in range(len(pairs)-1):
    #i = j+1
    #img = np.hstack((img, pairs[i]))
  #plt.imshow(img)
  #plt.show()

if __name__ == "__main__":
  main()

