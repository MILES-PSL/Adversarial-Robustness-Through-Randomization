
import json
import time
import os
import re
import socket
import pprint
from collections import OrderedDict
from os.path import join, basename, exists

import models
import attacks
from train_utils import losses
from dataset import readers_entropy
from utils import make_summary
from dump_files import DumpFiles
from utils import MessageBuilder

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
from tensorflow.python.lib.io import file_io

from config import YParams
from config import hparams as FLAGS

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def tf_get(name):
  collection = tf.get_collection(name)
  if len(collection) == 0:
    return []
  elif len(collection) > 1:
    raise ValueError("Mulitple values in collection {}".format(name))
  return collection[0]




class Evaluate:

  def __init__(self):
   self.wait = 20

  def _predict(self, images, labels, num_towers, device_string, n_classes,
               is_training=False, compute_loss=False):

    tower_inputs = tf.split(images, num_towers)
    tower_logits = []
    if compute_loss:
      tower_labels = tf.split(labels, num_towers)
      tower_losses = []

    for i in range(num_towers):
      with tf.device(device_string.format(i)):
        with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
          logits = self.model.create_model(
            tower_inputs[i], n_classes, is_training)
          tower_logits.append(logits)
          if compute_loss:
            losses = self.loss_fn.calculate_loss(
              logits=logits, labels=tower_labels[i])
            tower_losses.append(losses)

    logits_batch = tf.concat(tower_logits, 0)
    if compute_loss:
      losses_batch = tf.reduce_mean(tower_losses)
      return logits_batch, losses_batch
    return logits_batch

  def build_graph(self):
    """Creates the Tensorflow graph for evaluation."""

    num_towers = self.num_towers
    device_string = self.device_string
    n_classes = self.reader.n_classes

    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope("train_input"):
      images_batch, labels_batch = self.reader.input_fn()
      tf.summary.histogram("model/input_raw", images_batch)
    tf.add_to_collection('images', images_batch)
    tf.add_to_collection('labels',
      tf.one_hot(tf.reshape(labels_batch, [-1]), self.reader.n_classes))

    # get loss and logits from real examples
    logits_batch, loss_batch = self._predict(
      images_batch, labels_batch, num_towers, device_string,
      n_classes, is_training=False, compute_loss=True)
    tf.add_to_collection('logits', logits_batch)

    preds_batch = tf.argmax(logits_batch, axis=1, output_type=tf.int32)
    preds_batch = tf.one_hot(preds_batch, self.reader.n_classes)
    tf.add_to_collection('predictions', preds_batch)

  def _get_global_step_from_ckpt(self, filename):
    regex = "(?<=ckpt-)[0-9]+"
    return int(re.findall(regex, filename)[-1])

  def get_checkpoint(self, last_global_step):
    if not exists(self.train_dir):
      return None, None

    if FLAGS.start_eval_from_ckpt == 'first':
      files = file_io.get_matching_files(
        join(self.train_dir, 'model.ckpt-*.index'))
      # No files
      if not files:
        return None, None
      sort_fn = lambda x: int(re.findall("(?<=ckpt-)[0-9]+", x)[-1])
      files = sorted(files, key=self._get_global_step_from_ckpt)
      for filename in files:
        filname_global_step = self._get_global_step_from_ckpt(filename)
        if last_global_step < filname_global_step:
          return filename[:-6], filname_global_step
      return None, None
    else:
      latest_checkpoint = tf.train.latest_checkpoint(self.train_dir)
      if latest_checkpoint is None:
        return None, None
      global_step = self._get_global_step_from_ckpt(latest_checkpoint)
      return latest_checkpoint, global_step

  def get_best_checkpoint(self):
    best_acc_file = join(self.logs_dir, "best_accuracy.txt")
    if not exists(best_acc_file):
      raise ValueError("Could not find best_accuracy.txt in {}".format(
              self.logs_dir))
    with open(best_acc_file) as f:
      content = f.readline().split('\t')
      best_ckpt = content[0]
    best_ckpt_path = file_io.get_matching_files(
        join(self.train_dir, 'model.ckpt-{}.index'.format(best_ckpt)))
    return best_ckpt_path[-1][:-6], int(best_ckpt)

  def eval(self):
    """Compute entropy."""

    best_checkpoint, global_step = self.get_best_checkpoint()

    with tf.Session(config=self.config) as sess:
      logging.info("Compute Entropy")

      # Restores from checkpoint
      self.saver.restore(sess, best_checkpoint)
      sess.run(tf.local_variables_initializer())

      cumul = np.zeros((self.batch_size, self.reader.n_classes))
      entropy_sample = 10000
      images = tf_get('images')
      predictions = tf_get('predictions')
      prev_images_val, prev_labels_val = None, None
      for i in range(entropy_sample):
        images_val, predictions_val = \
            sess.run([images, predictions])
        cumul += predictions_val
        # if prev_images_val is not None:
        #   np.testing.assert_array_equal(prev_images_val, images_val)
        #   np.testing.assert_array_equal(prev_labels_val, labels_val)
        # prev_images_val = images_val
        # prev_labels_val = labels_val

        if i % 10 == 0 and i != 0:

          proba = cumul / (i+1)
          cross_entropy = proba * np.log(proba + 1e-8)
          entropy = np.mean(-np.sum(cross_entropy, axis=1))
          exp_entropy = np.mean(np.exp(np.sum(cross_entropy, axis=1)))

          msg = "iter: {: 7}, entropy: {:.4f}, exp entropy: {:.4f}".format(
            i, entropy, exp_entropy)
          logging.info(msg)

    return

  def run(self):

    tf.logging.set_verbosity(logging.INFO)

    # Setup logging & log the version.
    logging.info("Tensorflow version: {}.".format(tf.__version__))
    logging.info("hostname: {}.".format(
      socket.gethostname()))

    self.train_dir = FLAGS.train_dir
    self.logs_dir = "{}_logs".format(self.train_dir)

    self.num_gpu = 2
    self.batch_size = 1000
    self.num_towers = self.num_gpu
    self.device_string = '/gpu:{}'

    self.config = tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True)

    with tf.Graph().as_default():

      self.reader = find_class_by_name(FLAGS.reader, [readers_entropy])(
        self.batch_size, is_training=False)

      self.model = find_class_by_name(FLAGS.model, [models])()
      self.loss_fn = find_class_by_name(FLAGS.loss, [losses])()

      self.build_graph()
      logging.info("Built evaluation graph")

      self.saver = tf.train.Saver(tf.global_variables(scope="tower"))
      self.eval()

if __name__ == '__main__':
  evaluate = Evaluate()
  evaluate.run()
