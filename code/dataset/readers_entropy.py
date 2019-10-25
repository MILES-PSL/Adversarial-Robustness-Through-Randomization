
import random
from os.path import join
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
from tensorflow import flags

from config import hparams as FLAGS


class BaseReader:

  def _get_tfrecords(self, name):
    data_pattern = 'test-00001*,test-00002*'
    paths = list(map(lambda x: join(FLAGS.data_dir, name, x),
                     data_pattern.split(',')))
    files = gfile.Glob(paths)
    if not files:
      raise IOError("Unable to find files. data_pattern='{}'.".format(
        data_pattern))
    logging.info("Number of TFRecord files: {}.".format(
      len(files)))
    return files

  def _maybe_one_hot_encode(self, labels):
    """One hot encode the labels"""
    if FLAGS.one_hot_labels:
      labels = tf.one_hot(labels, self.n_classes)
      labels = tf.squeeze(labels)
      return labels
    return labels

  def _parse_and_processed(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    """
    image, label = self._parse_fn(example_serialized)
    image = self._image_preprocessing(image)
    return image, label

  def _parse_fn(self, example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers.
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
    """
    feature_map = {
      'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/label'], dtype=tf.int32)
    return features['image'], label

  def input_fn(self):

    with tf.device('/cpu:0'):
      files = tf.constant(self.files, name="tfrecord_files")
      with tf.name_scope('batch_processing'):
        dataset = tf.data.TFRecordDataset(files,
                        num_parallel_reads=self.num_parallel_readers)
        dataset = dataset.map(self._parse_and_processed,
                          num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
    label_batch = self._maybe_one_hot_encode(label_batch)
    return image_batch, label_batch


class CIFAR10Reader(BaseReader):

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = FLAGS.data_augmentation
    self.use_gray_scale = FLAGS.grayscale

    # use grey scale
    if self.use_gray_scale:
      self.batch_shape = (None, 32, 32, 1)

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = self._get_tfrecords('cifar10')

  def _data_augmentation(self, image):
    image = tf.image.resize_image_with_crop_or_pad(
                        image, self.height+4, self.width+4)
    image = tf.image.random_crop(image, [self.height, self.width, 3])
    image = tf.image.random_flip_left_right(image)
    return image

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    # Decode the string as an RGB JPEG.
    image = tf.decode_raw(image_buffer, tf.uint8)
    image = tf.reshape(image, (self.height, self.width, 3))
    if self.use_gray_scale:
      image = tf.image.rgb_to_grayscale(image)
      image = tf.reshape(image, (1, self.height * self.width))
    image = tf.cast(image, dtype=tf.float32)
    if self.use_data_augmentation and self.is_training:
      image = self._data_augmentation(image)
    if FLAGS.per_image_standardization:
      image = tf.image.per_image_standardization(image)
    elif FLAGS.dataset_standardization:
      mean = [125.3, 123.0, 113.9]
      std  = [63.0,  62.1,  66.7]
      image = (image - mean) / std
    else:
      image = (image / 255 - 0.5) * 2
    return image



class CIFAR100Reader(BaseReader):

  def __init__(self, batch_size, is_training, *args, **kwargs):

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.n_classes = 100
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = FLAGS.data_augmentation
    self.use_gray_scale = FLAGS.grayscale

    if self.use_gray_scale:
      self.batch_shape = (None, 32, 32, 1)

    readers_params = FLAGS.readers_params
    self.num_parallel_calls = readers_params['num_parallel_calls']
    self.num_parallel_readers = readers_params['num_parallel_readers']
    self.prefetch_buffer_size = readers_params['prefetch_buffer_size']

    self.files = self._get_tfrecords('cifar100')

  def _data_augmentation(self, image):
    image = tf.image.resize_image_with_crop_or_pad(
                        image, self.height+4, self.width+4)
    image = tf.image.random_crop(image, [self.height, self.width, 3])
    image = tf.image.random_flip_left_right(image)
    return image

  def _image_preprocessing(self, image_buffer):
    """Decode and preprocess one image for evaluation or training.
    Args:
      image: image as numpy array
    Returns:
      3D Tensor containing an appropriately scaled image
    """
    # Decode the string as an RGB JPEG.
    image = tf.decode_raw(image_buffer, tf.uint8)
    if self.use_gray_scale:
      image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (self.height, self.width, 3))
    if self.use_data_augmentation and self.is_training:
      image = self._data_augmentation(image)
    if FLAGS.per_image_standardization:
      image = tf.image.per_image_standardization(image)
    else:
      image = (image / 255 - 0.5) * 2
    return image
