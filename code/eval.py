
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
from dataset import readers
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


def get_epoch(step, n_gpu, batch_size, n_files):
  if n_gpu:
    return (step * batch_size * n_gpu) / n_files
  return (step * batch_size) / n_files


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
    tf.add_to_collection('processed_img', images_batch)
    tf.add_to_collection('labels', labels_batch)

    # get loss and logits from real examples
    logits_batch, loss_batch = self._predict(
      images_batch, labels_batch, num_towers, device_string,
      n_classes, is_training=False, compute_loss=True)
    tf.add_to_collection('logits', logits_batch)

    preds_batch = tf.argmax(logits_batch, axis=1, output_type=tf.int32)
    loss, loss_update_op = tf.metrics.mean(loss_batch)
    accuracy, acc_update_op = tf.metrics.accuracy(
      tf.cast(labels_batch, tf.float32), preds_batch)

    tf.add_to_collection('images_batch', images_batch)
    tf.add_to_collection('labels_batch', labels_batch)
    tf.add_to_collection('predictions', preds_batch)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('loss_update_op', loss_update_op)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('acc_update_op', acc_update_op)

    if FLAGS.eval_under_attack:

      # get loss and logits from adv examples
      def fn_logits(x):
        return self._predict(x, None, num_towers, device_string,
                      n_classes, is_training=False, compute_loss=False)

      images_adv_batch = self.attack.generate(images_batch, fn_logits)
      tf.add_to_collection('processed_img_adv', images_adv_batch)

      logits_adv_batch, losses_adv_batch = self._predict(
        images_adv_batch, labels_batch, num_towers, device_string,
        n_classes, is_training=False, compute_loss=True)

      preds_adv_batch = tf.argmax(
        logits_adv_batch, axis=1, output_type=tf.int32)

      loss_adv, loss_adv_update_op = tf.metrics.mean(losses_adv_batch)
      accuracy_adv, acc_adv_update_op = tf.metrics.accuracy(
        tf.cast(labels_batch, tf.float32), preds_adv_batch)

      perturbation = images_adv_batch - images_batch
      perturbation = tf.layers.flatten(perturbation)
      for name, p in [('l1', 1), ('l2', 2), ('linf', np.inf)]:
        value, update = tf.metrics.mean(
          tf.norm(perturbation, ord=p, axis=1))
        tf.add_to_collection('mean_norm_{}'.format(name), value)
        tf.add_to_collection('mean_norm_{}_update_op'.format(name), update)

      tf.add_to_collection('images_adv_batch', images_adv_batch)
      tf.add_to_collection('predictions_adv', preds_adv_batch)
      tf.add_to_collection('loss_adv', loss_adv)
      tf.add_to_collection('loss_adv_update_op', loss_adv_update_op)
      tf.add_to_collection('accuracy_adv', accuracy_adv)
      tf.add_to_collection('acc_adv_update_op', acc_adv_update_op)


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

  def eval_attack(self):
    """Run the evaluation under attack."""

    best_checkpoint, global_step = self.get_best_checkpoint()

    epoch = get_epoch(
             global_step,
             FLAGS.train_num_gpu,
             FLAGS.train_batch_size,
             self.reader.n_train_files)

    with tf.Session(config=self.config) as sess:
      logging.info("Evaluation under attack:")

      # Restores from checkpoint
      self.saver.restore(sess, best_checkpoint)
      sess.run(tf.local_variables_initializer())

      # pass session to attack class for Carlini Attack
      self.attack.sess = sess

      fetches = OrderedDict(
         loss_update_op=tf_get('loss_update_op'),
         acc_update_op=tf_get('acc_update_op'),
         loss_adv_update_op=tf_get('loss_adv_update_op'),
         acc_adv_update_op=tf_get('acc_adv_update_op'),
         mean_norm_l1_update_op=tf_get('mean_norm_l1_update_op'),
         mean_norm_l2_update_op=tf_get('mean_norm_l2_update_op'),
         mean_norm_linf_update_op=tf_get('mean_norm_linf_update_op'),
         images=tf_get('images_batch'),
         images_adv=tf_get('images_adv_batch'),
         loss=tf_get('loss'),
         accuracy=tf_get('accuracy'),
         predictions=tf_get('predictions'),
         predictions_adv=tf_get('predictions_adv'),
         loss_adv=tf_get('loss_adv'),
         accuracy_adv=tf_get('accuracy_adv'),
         labels_batch=tf_get('labels_batch'),
         mean_l1=tf_get('mean_norm_l1'),
         mean_l2=tf_get('mean_norm_l2'),
         mean_linf=tf_get('mean_norm_linf'))

      count = 0
      dump = DumpFiles(self.train_dir)
      while True:
        try:

          batch_start_time = time.time()
          values = sess.run(list(fetches.values()))
          values = dict(zip(fetches.keys(), values))
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch
          count += self.batch_size

          # dump images and images_adv
          if FLAGS.dump_files:
            dump.files(values)

          message = MessageBuilder()
          message.add('', [count, self.reader.n_test_files])
          message.add('acc img/adv',
                      [values['accuracy'], values['accuracy_adv']], format='.5f')
          message.add('avg loss', [values['loss'], values['loss_adv']], format='.5f')
          message.add('imgs/sec', examples_per_second, format='.3f')
          if FLAGS.eval_under_attack:
            norms_mean = [values['mean_l1'], values['mean_l2'], values['mean_linf']]
            message.add('l1/l2/linf mean', norms_mean, format='.2f')
          logging.info(message.get_message())

        except tf.errors.OutOfRangeError:

          message = MessageBuilder()
          message.add('Final: images/adv',
                      [values['accuracy'], values['accuracy_adv']], format='.5f')
          message.add('avg loss', [values['loss'], values['loss_adv']], format='.5f')
          logging.info(message.get_message())
          logging.info("Done evaluation of adversarial examples.")
          break

    return values['accuracy'], values['accuracy_adv']


  def eval_loop(self, last_global_step):
    """Run the evaluation loop once."""

    latest_checkpoint, global_step = self.get_checkpoint(
      last_global_step)
    logging.info("latest_checkpoint: {}".format(latest_checkpoint))

    if latest_checkpoint is None or global_step == last_global_step:
      time.sleep(self.wait)
      return last_global_step

    with tf.Session(config=self.config) as sess:
      logging.info("Loading checkpoint for eval: {}".format(latest_checkpoint))

      # Restores from checkpoint
      self.saver.restore(sess, latest_checkpoint)
      sess.run(tf.local_variables_initializer())

      epoch = get_epoch(
                global_step,
                FLAGS.train_num_gpu,
                FLAGS.train_batch_size,
                self.reader.n_train_files)

      fetches = OrderedDict(
         loss_update_op=tf_get('loss_update_op'),
         acc_update_op=tf_get('acc_update_op'),
         loss=tf_get('loss'),
         accuracy=tf_get('accuracy'))

      while True:
        try:

          batch_start_time = time.time()
          values = sess.run(list(fetches.values()))
          values = dict(zip(fetches.keys(), values))
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch

          message = MessageBuilder()
          message.add('epoch', epoch, format='.2f')
          message.add('step', global_step)
          message.add('accuracy', values['accuracy'], format='.5f')
          message.add('avg loss', values['loss'], format='.5f')
          message.add('imgs/sec', examples_per_second, format='.0f')
          logging.info(message.get_message())

        except tf.errors.OutOfRangeError:

          if self.best_accuracy is None or self.best_accuracy < values['accuracy']:
            self.best_global_step = global_step
            self.best_accuracy = values['accuracy']

          make_summary("accuracy", values['accuracy'], self.summary_writer, global_step)
          make_summary("loss", values['loss'], self.summary_writer, global_step)
          make_summary("epoch", epoch, self.summary_writer, global_step)
          self.summary_writer.flush()

          message = MessageBuilder()
          message.add('final: epoch', epoch, format='.2f')
          message.add('step', global_step)
          message.add('accuracy', values['accuracy'], format='.5f')
          message.add('avg loss', values['loss'], format='.5f')
          logging.info(message.get_message())
          logging.info("Done with batched inference.")

          if self.stopped_at_n:
           self.counter += 1

          break

      return global_step


  def run(self):

    # Setup logging & log the version.
    if not FLAGS.debug:
      tf.logging.set_verbosity(logging.INFO)
    else:
      tf.logging.set_verbosity(logging.DEBUG)
    logging.info("Tensorflow version: {}.".format(tf.__version__))
    logging.info("hostname: {}.".format(
      socket.gethostname()))

    self.train_dir = FLAGS.train_dir
    self.logs_dir = "{}_logs".format(self.train_dir)

    if FLAGS.eval_num_gpu:
      self.batch_size = \
          FLAGS.eval_batch_size * FLAGS.eval_num_gpu
    else:
      self.batch_size = FLAGS.eval_batch_size

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.eval_num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
      logging.info("Using the {} GPUs".format(num_gpus))
      self.num_towers = num_gpus
      self.device_string = '/gpu:{}'
      logging.info("Using total batch size of {} for evaluation "
        "over {} GPUs: batch size of {} per GPUs.".format(
          self.batch_size, self.num_towers,
            self.batch_size // self.num_towers))
    else:
      logging.info("No GPUs found. Eval on CPU.")
      self.num_towers = 1
      self.device_string = '/cpu:{}'
      logging.info("Using total batch size of {} for evalauton "
        "on CPU.".format(self.batch_size))

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(FLAGS.values()))

    self.config = tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement,
      allow_soft_placement=True)

    with tf.Graph().as_default():

      self.reader = find_class_by_name(FLAGS.reader, [readers])(
        self.batch_size, is_training=False)

      if FLAGS.eval_under_attack:
        attack_method = FLAGS.attack_method
        attack_cls = getattr(attacks, attack_method, None)
        if attack_cls is None:
          raise ValueError("Attack is not recognized.")
        attack_config = getattr(FLAGS, attack_method)
        self.attack = attack_cls(
          batch_size=self.batch_size, sample=FLAGS.attack_sample, **attack_config)


      data_pattern = FLAGS.data_pattern
      self.dataset = re.findall("[a-z0-9]+", data_pattern.lower())[0]
      if data_pattern is "":
        raise IOError("'data_pattern' was not specified. "
          "Nothing to evaluate.")

      self.model = find_class_by_name(FLAGS.model, [models])()
      self.loss_fn = find_class_by_name(FLAGS.loss, [losses])()

      self.build_graph()
      logging.info("Built evaluation graph")

      if FLAGS.eval_under_attack:
        self.saver = tf.train.Saver(tf.global_variables(scope="tower"))
        acc_val, acc_adv_val = self.eval_attack()
        # filename = "score_{}.txt".format(self.attack.get_name())
        path = join(self.logs_dir, "attacks_score.txt")
        with open(path, 'a') as f:
          f.write("{}\n".format(FLAGS.attack_method))
          f.write("sample {}, {}\n".format(FLAGS.attack_sample,
                                         json.dumps(attack_config)))
          f.write("{:.5f}\t{:.5f}\n\n".format(acc_val, acc_adv_val))

      else:

        self.saver = tf.train.Saver(tf.global_variables())
        filename_suffix= "_{}_{}".format("eval",
                    re.findall("[a-z0-9]+", data_pattern.lower())[0])
        self.summary_writer = tf.summary.FileWriter(
          self.train_dir,
          filename_suffix=filename_suffix,
          graph=tf.get_default_graph())

        if FLAGS.stopped_at_n == "auto":
          one_epoch = self.reader.n_train_files / \
              (FLAGS.train_batch_size * FLAGS.train_num_gpu)
          self.stopped_at_n = (FLAGS.num_epochs * one_epoch) // FLAGS.save_checkpoint_steps
        else:
          self.stopped_at_n = FLAGS.stopped_at_n
        logging.info("Making evaluation for {} ckpts.".format(
          int(self.stopped_at_n)))

        self.best_global_step = None
        self.best_accuracy = None
        self.counter = 0
        last_global_step_val = 0
        while self.counter < self.stopped_at_n:
          last_global_step_val = self.eval_loop(last_global_step_val)
        path = join(self.logs_dir, "best_accuracy.txt")
        with open(path, 'w') as f:
          f.write("{}\t{:.4f}\n".format(self.best_global_step, self.best_accuracy))

      logging.info("Done evaluation -- number of eval reached.")

if __name__ == '__main__':
  evaluate = Evaluate()
  evaluate.run()
