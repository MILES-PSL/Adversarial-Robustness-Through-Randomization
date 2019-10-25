
import os, sys
import json
import time
import pprint
import socket
from collections import OrderedDict
from os.path import join, exists
from datetime import datetime

import models
from dataset import readers
from train_utils import losses
from train_utils.learning_rate import LearningRate
from train_utils.optimizer import Optimizer
from train_utils.gradients import ComputeAndProcessGradients
from train_utils.gradients import compute_hessian_and_summary
from train_utils.gradients import combine_gradients
from train_utils.update_ops import UpdateOps
from eval_utils import eval_util
from utils import MessageBuilder

import tensorflow as tf
from tensorflow import app
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib

from config import hparams as FLAGS


def task_as_string(task):
  return "/job:{}/task:{}".format(task.type, task.index)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def build_graph(reader, model, label_loss_fn, batch_size, regularization_penalty):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: the input class.
    model: The core model.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
  """
  global_step = tf.train.get_or_create_global_step()

  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  gpus = gpus[:FLAGS.train_num_gpu]
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:{}'
    logging.info("Using total batch size of {} for training "
      "over {} GPUs: batch size of {} per GPUs.".format(
        batch_size, num_towers, batch_size // num_towers))
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:{}'
    logging.info("Using total batch size of {} for training. ".format(
      batch_size))

  learning_rate = LearningRate(global_step, batch_size).get_learning_rate()
  opt = Optimizer(learning_rate).get_optimizer()

  with tf.name_scope("input"):
    images_batch, labels_batch = reader.input_fn()
  tf.summary.histogram("model/input_raw", images_batch)

  gradients_cls = ComputeAndProcessGradients()

  tower_inputs = tf.split(images_batch, num_towers)
  tower_labels = tf.split(labels_batch, num_towers)
  tower_gradients = []
  tower_logits = []
  tower_final_losses = []
  for i in range(num_towers):
    reuse = tf.AUTO_REUSE
    reuse = False if i == 0 else True
    with tf.device(device_string.format(i)):
      with tf.variable_scope("tower", reuse=reuse):

          logits = model.create_model(tower_inputs[i],
            n_classes=reader.n_classes, is_training=True)
          tower_logits.append(logits)

          label_loss = label_loss_fn.calculate_loss(
            logits=logits, labels=tower_labels[i])
          reg_losses = tf.losses.get_regularization_losses()
          if reg_losses:
            reg_loss = tf.add_n(reg_losses)
          else:
            reg_loss = tf.constant(0.)

          # Adds update_ops (e.g., moving average updates in batch norm) as
          # a dependency to the train_op.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          if update_ops:
            with tf.control_dependencies(update_ops):
              barrier = tf.no_op(name="gradient_barrier")
              with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)

          # Incorporate the L2 weight penalties etc.
          final_loss = regularization_penalty * reg_loss + label_loss
          gradients = gradients_cls.get_gradients(opt, final_loss)
          tower_gradients.append(gradients)
          tower_final_losses.append(final_loss)

  total_loss = tf.stack(tower_final_losses)
  full_gradients = combine_gradients(tower_gradients)

  # make summary
  tf.summary.scalar("loss", tf.reduce_mean(total_loss))
  for variable in tf.trainable_variables():
    tf.summary.histogram(variable.op.name, variable)

  # apply gradients
  # gradients = gradients_cls.get_gradients(opt, total_loss)
  train_op_cls = UpdateOps(opt)

  summary_op = tf.summary.merge_all()
  with tf.control_dependencies([summary_op]):
    train_op = train_op_cls.make_update(full_gradients, global_step)

  logits = tf.concat(tower_logits, 0)

  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("logits", logits)
  tf.add_to_collection("labels", labels_batch)
  tf.add_to_collection("learning_rate", learning_rate)
  tf.add_to_collection("summary_op", summary_op)
  tf.add_to_collection("train_op", train_op)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, batch_size):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """
    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.model = model
    self.reader = reader
    self.batch_size = batch_size

    self.config = tf.ConfigProto(allow_soft_placement=True,
      log_device_placement=FLAGS.log_device_placement)
    jit_level = 0
    if FLAGS.compile:
      # Turns on XLA JIT compilation.
      jit_level = tf.OptimizerOptions.ON_2
    self.config.graph_options.optimizer_options.global_jit_level = jit_level

    # define the number of epochs
    num_steps_by_epochs = reader.n_train_files / batch_size
    self.max_steps = FLAGS.num_epochs * num_steps_by_epochs

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model and exists(self.train_dir):
      self.remove_training_directory(self.train_dir)

    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(FLAGS.values()))

    model_flags_dict = FLAGS.to_json()
    log_folder = '{}_logs'.format(self.train_dir)
    flags_json_path = join(log_folder, "model_flags.json")
    if not exists(flags_json_path):
      # Write the file.
      with open(flags_json_path, "w") as fout:
        fout.write(model_flags_dict)

    target, device_fn = self.start_server_if_distributed()
    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:
      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.train.get_global_step()
        loss = tf.get_collection("loss")[0]
        logits = tf.get_collection("logits")[0]
        labels = tf.get_collection("labels")[0]
        learning_rate = tf.get_collection("learning_rate")[0]
        train_op = tf.get_collection("train_op")[0]
        summary_op = tf.get_collection("summary_op")[0]
        init_op = tf.global_variables_initializer()

        gradients_norm = tf.get_collection("gradients_norm")[0]


      scaffold = tf.train.Scaffold(
        saver=saver,
        init_op=init_op,
        summary_op=summary_op,
      )

      hooks = [
        tf.train.NanTensorHook(loss),
        tf.train.StopAtStepHook(num_steps=self.max_steps),
      ]

      session_args = dict(
        is_chief=self.is_master,
        scaffold=scaffold,
        checkpoint_dir=FLAGS.train_dir,
        hooks=hooks,
        save_checkpoint_steps=FLAGS.save_checkpoint_steps,
        save_summaries_steps=10,
        save_summaries_secs=None,
        log_step_count_steps=0,
        config=self.config,
      )

      logging.info("Start training")
      with tf.train.MonitoredTrainingSession(**session_args) as sess:

        summary_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)

        if FLAGS.profiler:
          profiler = tf.profiler.Profiler(sess.graph)

        global_step_val = 0
        while not sess.should_stop():

          make_profile = False
          profile_args = {}

          if global_step_val % 1000 == 0 and FLAGS.profiler:
            make_profile = True
            run_meta = tf.RunMetadata()
            profile_args = {
              'options': tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE),
              'run_metadata': run_meta
            }

          fetches = OrderedDict(
             train_op=train_op,
             global_step=global_step,
             loss=loss,
             learning_rate=learning_rate,
             logits=logits,
             labels=labels
          )

          if gradients_norm != 0:
            fetches['gradients_norm'] = gradients_norm
          else:
            grad_norm_val = 0

          batch_start_time = time.time()
          values = sess.run(list(fetches.values()), **profile_args)
          fetches_values = OrderedDict(zip(fetches.keys(), values))
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = self.batch_size / seconds_per_batch

          global_step_val = fetches_values['global_step']
          loss_val = fetches_values['loss']
          learning_rate_val = fetches_values['learning_rate']
          predictions_val = fetches_values['logits']
          labels_val = fetches_values['labels']

          if gradients_norm != 0:
            grad_norm_val = fetches_values['gradients_norm']

          if FLAGS.gradients['compute_hessian'] and global_step_val != 0 and \
             global_step_val % FLAGS.gradients['hessian_every_n_step'] == 0:
            compute_hessian_and_summary(sess, summary_writer, global_step_val)

          if make_profile and FLAGS.profiler:
            profiler.add_step(global_step_val, run_meta)

            # Profile the parameters of your model.
            profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder
                .trainable_variables_parameter()))

            # Or profile the timing of your model operations.
            opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
            profiler.profile_operations(options=opts)

            # Or you can generate a timeline:
            opts = (tf.profiler.ProfileOptionBuilder(
                    tf.profiler.ProfileOptionBuilder.time_and_memory())
                    .with_step(global_step_val)
                    .with_timeline_output('~/profile.logs').build())
            profiler.profile_graph(options=opts)


          to_print = global_step_val % FLAGS.frequency_log_steps == 0
          if (self.is_master and to_print) or global_step_val == 1:
            epoch = ((global_step_val * self.batch_size)
              / self.reader.n_train_files)

            message = MessageBuilder()
            message.add("epoch", epoch, format="4.2f")
            message.add("step", global_step_val, width=5, format=".0f")
            message.add("lr", learning_rate_val, format=".6f")
            message.add("loss", loss_val, format=".4f")
            if "YT8M" in self.reader.__class__.__name__:
              gap = eval_util.calculate_gap(predictions_val, labels_val)
              message.add("gap", gap, format=".3f")
            message.add("imgs/sec", examples_per_second, width=5, format=".0f")
            if FLAGS.gradients['perturbed_gradients']:
              message.add("grad norm", grad_norm_val, format=".4f")
            logging.info(message.get_message())

        # End training
        logging.info("{}: Done training -- epoch limit reached.".format(
          task_as_string(self.task)))
        if FLAGS.profiler:
          profiler.advise()
    logging.info("{}: Exited training loop.".format(task_as_string(self.task)))


  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("{}: Starting trainer within cluster {}.".format(
                   task_as_string(self.task), self.cluster.as_dict()))
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device=task_as_string(self.task),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(("{}: Train dir already exist and start_new_model "
                    "set to True. To restart model from scratch, "
                    "delete the directory.").format(task_as_string(self.task)))
      # gfile.DeleteRecursively(train_dir)
      sys.exit()
    except:
      logging.error("{}: Failed to delete directory {} when starting a new "
        "model. Please delete it manually and try again.".format(
          task_as_string(self.task), train_dir))
      sys.exit()


  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("{}: Flag 'start_new_model' is set. Building a new "
        "model.".format(task_as_string(self.task)))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("{}: No checkpoint file found. Building a new model.".format(
                   task_as_string(self.task)))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("{}: No meta graph file found. Building a new model.".format(
                     task_as_string(self.task)))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("{}: Restoring from meta graph file {}".format(
      task_as_string(self.task), meta_filename))
    return tf.train.import_meta_graph(meta_filename,
      clear_devices=FLAGS.clear_devices)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.loss, [losses, tf.nn])()

    build_graph(reader=reader,
                model=model,
                label_loss_fn=label_loss_fn,
                batch_size=self.batch_size,
                regularization_penalty=FLAGS.regularization_penalty)

    #TODO: make max_to_keep a FLAGS argument
    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("{}: Starting parameter server within cluster {}.".format(
      task_as_string(self.task), self.cluster.as_dict()))
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """
  if not task.type:
    raise ValueError("{}: The task type must be specified.".format(
      task_as_string(task)))
  if task.index is None:
    raise ValueError("{}: The task index must be specified.".format(
      task_as_string(task)))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)


def main():

  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Setup logging & log the version.
  if not FLAGS.debug:
    logging.set_verbosity(logging.INFO)
    # export TF_CPP_MIN_LOG_LEVEL=3
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
  else:
    logging.set_verbosity(logging.DEBUG)
  logging.info("{}: Tensorflow version: {}.".format(
    task_as_string(task), tf.__version__))
  logging.info("hostname: {}.".format(
    socket.gethostname()))

  if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    if FLAGS.train_num_gpu == 0:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, range(FLAGS.train_num_gpu)))

  # Define batch size
  if FLAGS.train_num_gpu:
    batch_size = FLAGS.train_batch_size * FLAGS.train_num_gpu
  else:
    batch_size = FLAGS.train_batch_size

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    reader = find_class_by_name(FLAGS.reader, [readers])(
      batch_size, is_training=True)

    model = find_class_by_name(FLAGS.model, [models])()
    logging.info("Using {} as model".format(FLAGS.model))

    trainer = Trainer(cluster, task, FLAGS.train_dir, model, reader, batch_size)
    trainer.run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("{}: Invalid task_type: {}.".format(
      task_as_string(task), task.type))


if __name__ == "__main__":
  main()
