
import os, sys, six
import random
import tarfile
import pickle
from os.path import join, exists
from six.moves import urllib
from datetime import datetime
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python_io import TFRecordWriter
from tensorflow.keras import datasets

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "/tmp/",
                    "Output data directory")

flags.DEFINE_integer("train_split", 10,
                     "Number of TFRecords to split the train dataset")

flags.DEFINE_integer("test_split", 1,
                     "Number of TFRecords to split the test dataset")


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



class ConvertDataset:

  def reshape(self):
    if self.x_train.ndim == 3:
      _, self.height, self.width = self.x_train.shape
      dim = self.height * self.width
    else:
      _, self.height, self.width, self.channels = self.x_train.shape
      dim = self.height * self.width * self.channels
    self.x_train = self.x_train.reshape(-1, dim)
    self.x_test = self.x_test.reshape(-1, dim)
    self.y_train = self.y_train.reshape(-1, 1)
    self.y_test = self.y_test.reshape(-1, 1)

  def _convert_to_example(self, image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(image),
      'image/height': _int64_feature(self.height),
      'image/width': _int64_feature(self.width),
      'image/label': _int64_feature(label)}))
    return example

  def _process_images(self, name, images, labels, id_file, n_files):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      name: string, unique identifier specifying the data set
      images: array of images
      labels: array of labels
    """
    output_filename = '{}-{:05d}-of-{:05d}'.format(name, id_file, n_files)
    output_file = os.path.join(FLAGS.output_dir, self.name, output_filename)
    with TFRecordWriter(output_file) as writer:
      for image, label in zip(images, labels):
        example = self._convert_to_example(image.tobytes(), label)
        writer.write(example.SerializeToString())
    print('{}: Wrote {} images to {}'.format(
        datetime.now(), len(images), output_file), flush=True)

  def split_data(self, x, y, n_split):
    x_split = np.array_split(x, n_split)
    y_split = np.array_split(y, n_split)
    return x_split, y_split

  def convert(self):
    """Main method to convert mnist images to TFRecords
    """
    data_folder = join(FLAGS.output_dir, self.name)
    if exists(data_folder):
      logging.info('{} data already converted to TFRecords.'.format(self.name))
      return
    os.makedirs(data_folder)
    x_train_split, y_train_split = self.split_data(
      self.x_train, self.y_train, FLAGS.train_split)
    x_test_split, y_test_split = self.split_data(
      self.x_test, self.y_test, FLAGS.test_split)
    for i, (x, y) in enumerate(zip(x_train_split, y_train_split), 1):
      self._process_images("train", x, y, i, FLAGS.train_split)
    for i, (x, y) in enumerate(zip(x_test_split, y_test_split), 1):
      self._process_images("test", x, y, i , FLAGS.test_split)


class ConvertMNIST(ConvertDataset):
  def __init__(self):
    self.name = "mnist"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.mnist.load_data()
    self.reshape()

class ConvertFashionMNIST(ConvertDataset):
  def __init__(self):
    self.name = "fashion_mnist"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.fashion_mnist.load_data()
    self.reshape()

class ConvertCIFAR10(ConvertDataset):
  def __init__(self):
    self.name = "cifar10"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.cifar10.load_data()
    self.reshape()

class ConvertCIFAR100(ConvertDataset):
  def __init__(self):
    self.name = "cifar100"
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        datasets.cifar100.load_data()
    self.reshape()



















class ConvertDataset:

  def reshape(self):
    if self.x_train.ndim == 3:
      _, self.height, self.width = self.x_train.shape
      dim = self.height * self.width
    else:
      _, self.height, self.width, self.channels = self.x_train.shape
      dim = self.height * self.width * self.channels
    self.x_train = self.x_train.reshape(-1, dim)
    self.x_test = self.x_test.reshape(-1, dim)
    self.y_train = self.y_train.reshape(-1, 1)
    self.y_test = self.y_test.reshape(-1, 1)

  def _convert_to_example(self, image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(image),
      'image/label': _bytes_feature(label)}))
    return example

  def _process_images(self, name, images, labels, id_file, n_files):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      name: string, unique identifier specifying the data set
      images: array of images
      labels: array of labels
    """
    output_filename = '{}-{:05d}-of-{:05d}'.format(name, id_file, n_files)
    output_file = os.path.join(FLAGS.output_dir, self.name, output_filename)
    with TFRecordWriter(output_file) as writer:
      for image, label in zip(images, labels):
        example = self._convert_to_example(image.tobytes(), label.tobytes())
        writer.write(example.SerializeToString())
    print('{}: Wrote {} images to {}'.format(
        datetime.now(), len(images), output_file), flush=True)

  def convert(self):
    """Main method to convert mnist images to TFRecords
    """
    data_folder = join(FLAGS.output_dir, self.name)
    if exists(data_folder):
      logging.info('{} data already converted to TFRecords.'.format(self.name))
      return
    os.makedirs(data_folder)
    self._process_images("train", self.x_train, self.y_train, 1, 1)
    self._process_images("test", self.x_train, self.y_train, 1, 1)




class ConvertRandom(ConvertDataset):
  def __init__(self):
    self.name = "random"
    X = np.random.uniform(0, 1, size=(10000, 32))
    W = np.random.uniform(0, 1, size=(32, 32))
    eps = np.random.normal(0, 10e-4, size=(10000, 32))
    Y = X @ W + eps
    (self.x_train, self.y_train), (self.x_test, self.y_test) = \
        (X, Y), (X, Y)





def main(_):
  # ConvertMNIST().convert()
  # ConvertFashionMNIST().convert()
  # ConvertCIFAR10().convert()
  # ConvertCIFAR100().convert()
  ConvertRandom().convert()

if __name__ == '__main__':
  tf.app.run()
