import numpy as np
import sys

# Splits a row element array into train, test, and val sets
# target: The source file with a row array
# train_file: The destination file for training
# val_file: The destination file for val
# test_file: The destination file for test
# val_percent: The percent of the target to use for validation
# test_percent: The percent of the target to use for testing
# Saves the files for you
def split_file(target, train_file, val_file, test_file, val_percent, test_percent):
  with open(target, "r") as f:
    dat = np.load(f)
    split(dat, train_file, val_file, test_file, val_percent, test_percent)

# Splits a row element array into train, test, and val sets
# target: The row wise array of elements to split
# train_file: The destination file for training
# val_file: The destination file for val
# test_file: The destination file for test
# val_percent: The percent of the target to use for validation
# test_percent: The percent of the target to use for testing
# Saves the files for you
def split(target, train_file, val_file, test_file, val_percent, test_percent):
  print("Spliting")
  dat = target
  length = dat.shape[0]
  print(length)
  test_count = int(test_percent * length)
  val_count = int(val_percent * length)
  train_count = length - val_count - test_count

  # Split into train and val+test
  # Get indices for split
  ind = range(length)
  both = np.hstack((np.ones((1, val_count + test_count)), np.zeros((1, train_count))))
  both = np.squeeze(both)
  both = np.random.permutation(both)
  valtest_ind = np.transpose(np.nonzero(ind * both))

  # Perform split
  train = np.delete(dat, valtest_ind, axis=0)
  valtest = np.squeeze(np.take(dat, valtest_ind, axis=0))

  # Split into val and test
  # Get indices for split
  test_ind = np.hstack((np.ones((1, test_count - 1)), np.zeros((1, val_count))))
  test_ind = np.squeeze(test_ind)
  test_ind = np.random.permutation(test_ind)
  test_ind = np.transpose(np.nonzero(range(val_count+test_count-1) * test_ind))

  # Perform split
  val = np.delete(valtest, test_ind, axis=0)
  test = np.squeeze(np.transpose(np.take(valtest, test_ind, axis=0)))

  # Save
  with open(train_file, "wb") as f:
    np.save(f, train)
  with open(val_file, "wb") as f:
    np.save(f, val)
  with open(test_file, "wb") as f:
    np.save(f, test)

# Usable as a script, but used in gen.py
if __name__ == "__main__":
  target = sys.argv[1]
  train_file = sys.argv[2]
  val_file = sys.argv[3]
  test_file = sys.argv[4]
  val_percent = float(sys.argv[5])
  test_percent = float(sys.argv[6])

  split_file(target, train_file, val_file, test_file, val_percent, test_percent)

