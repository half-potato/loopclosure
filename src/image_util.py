from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from collections import namedtuple
import tensorflow as tf
import numpy as np
import os

# Used all over to store stats for each dataset
ImageStats = namedtuple("ImageTuple", ["size", "depth", "mean", "std"])

# Convienence function to load images
# path: path to image
# data_in: The tensor input for the raw jpeg data
# img_out: The tensor output for the image
# sess: session to use to load
def load_img(path, data_in, img_out, sess):
  if type(path) == np.ndarray:
    path = str(np.squeeze(path))
  path = path.strip()
  if os.path.isfile(path):
    im = gfile.FastGFile(path, "rb").read()
    try:
      out = sess.run(img_out, {data_in: im})
    except Exception as e:
      print(e)
      print(path)
      out = None
    return out
  else:
    print(path + " not found")
    return None

# Normalizing image tensor
# img: Image tensor. Shape doesn't matter
# Returns: output image tensor
def normalize(img, mean, std):
  offset_image = tf.subtract(img, mean)
  mul_image = tf.multiply(offset_image, 1.0 / std)
  return mul_image

# Randomly brightens an image
# img: Image tensor. Shape doesn't matter
# random_brightness: integer between 0 and 100
# Returns: output image tensor
def brighten(img, random_brightness):
  """
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.
  """
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(img, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return distort_result

# Cuts an image in half, randomizing the margin between the top of the result 
# and the original image
# img: Image tensor (must be wide enough)
# input_width: The width of the input image (ie: 640)
# input_height: The height of the input image (ie: 480)
# input_depth: The depth of the input image. Usually 3 for RGB
# final_image_width: The output image width (192)
# final_image_height: The output image height (192)
# Returns: left image tensor, right image tensor with shapes (1, 192, 192, 3)
def add_edge_crop(img, width, height, original_width, original_height, left_edge):

  with tf.variable_scope("cropper"):
    if left_edge:
      w_offset = 0
    else:
      w_offset = tf.random_uniform(tensor_shape.scalar(),
                                   minval=0,
                                   maxval=original_width - width)
      w_offset = tf.cast(w_offset, dtype=tf.int32)
    # Crop twice for each half

    # Generate vertical offsets for left and right hand images
    h_offset = tf.random_uniform(tensor_shape.scalar(),
                                 minval=0,
                                 maxval=original_height - height)
    h_offset = tf.cast(h_offset, dtype=tf.int32)

    # Prep for cropping
    squeeze_image = tf.squeeze(img, squeeze_dims=[0])
    # Crop
    image = tf.image.crop_to_bounding_box(squeeze_image, 
                    h_offset, w_offset, width, height)
    # Resize
    f_shape = tf.stack([width, height])
    f_shape = tf.cast(f_shape, dtype=tf.int32)
    image_f = tf.image.resize_bilinear(tf.expand_dims(image,0), f_shape)
  return image_f

# Cuts an image in half, randomizing the margin between the top of the result 
# and the original image
# img: Image tensor (must be wide enough)
# input_width: The width of the input image (ie: 640)
# input_height: The height of the input image (ie: 480)
# input_depth: The depth of the input image. Usually 3 for RGB
# final_image_width: The output image width (192)
# final_image_height: The output image height (192)
# Returns: left image tensor, right image tensor with shapes (1, 192, 192, 3)
def add_cropper(img, width, height, final_image_width, final_image_height):

  with tf.variable_scope("cropper"):
    # Crop twice for each half

    # Generate vertical offsets for left and right hand images
    l_h_offset = tf.random_uniform(tensor_shape.scalar(),
                                   minval=0,
                                   maxval=height - width)
    l_h_offset = tf.cast(l_h_offset, dtype=tf.int32)
    r_h_offset = tf.random_uniform(tensor_shape.scalar(),
                                   minval=0,
                                   maxval=height - width)
    r_h_offset = tf.cast(r_h_offset, dtype=tf.int32)

    # Prep for cropping
    squeeze_image = tf.squeeze(img, squeeze_dims=[0])
    # Crop
    r_image = tf.image.crop_to_bounding_box(squeeze_image, 
                    r_h_offset, width/2, width/2, width/2)
    l_image = tf.image.crop_to_bounding_box(squeeze_image, 
                    l_h_offset, 0, width/2, width/2)
    # Resize
    f_shape = tf.stack([final_image_width, final_image_height])
    f_shape = tf.cast(f_shape, dtype=tf.int32)
    r_image_f = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(r_image,0),
      f_shape), squeeze_dims=[0])
    l_image_f = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(l_image,0),
      f_shape), squeeze_dims=[0])
  return l_image_f, r_image_f

# Takes in jpeg data, outputs image
# input_depth: The depth of the input image. Usually 3 for RGB
# Returns: jpeg data tensor, output image tensor
def add_jpeg_decoding(input_depth):
  with tf.variable_scope("jpeg_decode"):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    return jpeg_data, decoded_image_4d

# Resize operation tensor
# img: Image tensor (must be wide enough)
# output_width: The width of the output image (192)
# output_height: The height of the output image (192)
# Returns: sized image tensor
def resize(img, output_width, output_height):
  resize_shape = tf.stack([output_height, output_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(img, resize_shape_as_int)
  return resized_image

# Decodes the image, resizes it, and normalizes it
# output_width: The width of the output image (192)
# output_height: The height of the output image (192)
# output_depth: The depth of the output image. Usually 3 for RGB
# input_mean: The mean of the dataset
# input_std: The std of the dataset
# Returns: image tensor with shape (1, 192, 192, 3)
def add_basic_ops(output_width, output_height, output_depth,
                  input_mean, input_std):
    jpeg_data, decoded_image_4d = add_jpeg_decoding(output_depth)
    resized = resize(decoded_image_4d, output_width, output_height)
    norm = normalize(resized, input_mean, input_std)
    return jpeg_data, norm

# Randomly crops an image by a random amount
# image: image to crop
# min_crop: Min crop amount as a percentage float
# max_crop: Max crop amount as a percentage float
# output_width: The width of the output image (192)
# output_height: The height of the output image (192)
# input_width: Horizontal size of expected input image to model.
# input_height: Vertical size of expected input image to model.
# input_depth: How many channels the expected input image should have.
# Returns: image tensor with shape (1, 192, 192, 3)
def add_random_crop(image, min_crop, max_crop, output_width, output_height,
    input_width, input_height, input_depth):
  # Crop
  size = tf.random_uniform(tensor_shape.scalar(),
                           minval=min_crop*min(input_width, input_height),
                           maxval=max_crop*min(input_width, input_height))
  size = tf.cast(size, dtype=tf.int32)

  precrop = tf.squeeze(image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precrop, [size, size, input_depth])
  post_crop = tf.expand_dims(cropped_image, 0)
  resized = resize(post_crop, output_width, output_height)
  return resized

# Decodes an image and distorts it
# flip_left_right: Boolean whether to randomly mirror images horizontally.
# random_crop: Integer percentage setting the total margin used around the
# crop box.
# random_scale: Integer percentage of how much to vary the scale by.
# random_brightness: Integer range to randomly multiply the pixel values by.
# graph.
# input_width: Horizontal size of expected input image to model.
# input_height: Vertical size of expected input image to model.
# input_depth: How many channels the expected input image should have.
# input_mean: Pixel value that should be zero in the image for the graph.
# input_std: How much to divide the pixel values by before recognition.
# Returns: jpeg input layer and the distorted result tensor.
def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~
  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.
  """

  with tf.variable_scope("jpeg_distort"):
    jpeg_data, decoded_image_4d = add_jpeg_decoding(input_depth)
    # Scale
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    # Crop
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [input_height, input_width, input_depth])
    if flip_left_right:
      flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
      flipped_image = cropped_image
    brightened_image = brighten(flipped_image, random_brightness)
    norm = normalize(brightened_image, input_mean, input_std)
    distort_result = tf.expand_dims(norm, 0, name='DistortResult')
  return jpeg_data, distort_result
