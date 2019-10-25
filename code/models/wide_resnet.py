
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from .base import BaseModel
from config import hparams as FLAGS

class WideResnetModel(BaseModel):
  """Wide ResNet model.
     https://arxiv.org/abs/1605.07146
  """

  def l1_normalize(self, x, dim, epsilon=1e-12, name=None):
    """Normalizes along dimension `dim` using an L1 norm.
    For a 1-D tensor with `dim = 0`, computes
        output = x / max(sum(abs(x)), epsilon)
    For `x` with more dimensions, independently normalizes each 1-D slice along
    dimension `dim`.
    Args:
      x: A `Tensor`.
      dim: Dimension along which to normalize.  A scalar or a vector of
        integers.
      epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
        divisor if `norm < sqrt(epsilon)`.
      name: A name for this operation (optional).
    Returns:
      A `Tensor` with the same shape as `x`.
    """
    with tf.name_scope(name, "l1_normalize", [x]) as name:
      abs_sum = tf.reduce_sum(tf.abs(x), dim, keep_dims = True)
      x_inv_norm = tf.reciprocal(tf.maximum(abs_sum, epsilon))
      return tf.multiply(x, x_inv_norm, name=name)

  def _get_noise(self, x):
    """Pixeldp noise layer."""

    train_with_noise = self.is_training and self.train_with_noise
    if train_with_noise or (self.train_with_noise and FLAGS.noise_in_eval):
      noise_activate = tf.constant(1.0)
      logging.info(
        "train/eval with noise - noise sd {:.2f}".format(self.scale_noise))
    else:
      noise_activate = tf.constant(0.0)

    loc = tf.zeros(tf.shape(x), dtype=tf.float32)
    scale = tf.ones(tf.shape(x), dtype=tf.float32)

    if self.distributions == 'l1':
      noise = tf.distributions.Laplace(loc, scale).sample()
      noise = noise_activate * (self.scale_noise / tf.sqrt(2.)) * noise

    elif self.distributions == 'l2':
      noise = tf.distributions.Normal(loc, scale).sample()
      noise = noise_activate * self.scale_noise * noise

    elif self.distributions == 'exp':
      noise = tf.distributions.Exponential(rate=scale).sample()
      noise = noise_activate * self.scale_noise * noise

    elif self.distributions == 'weibull':
      k = 3
      eps = 10e-8
      alpha = ((k - 1) / k)**(1 / k)
      U = tf.random_uniform(tf.shape(x), minval=eps, maxval=1)
      X = (-tf.log(U) + alpha**k)**(1 / k) - alpha
      tensor = tf.zeros_like(x) + 0.5
      bernoulli = tf.distributions.Bernoulli(probs=tensor).sample()
      B = tf.cast(2 * bernoulli - 1, tf.float32)
      noise = B * X / 0.3425929
      noise = noise_activate * self.scale_noise * noise

    else:
      raise ValueError("Distributions is not recognised.")

    return noise

  def _conv_with_noise(self, name, x, filter_size, in_filters, out_filters,
                       strides):

    assert(strides[1] == strides[2])
    stride = strides[1]

    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      shape = [filter_size, filter_size, in_filters, out_filters]
      kernel = tf.get_variable('kernel', shape,
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)),
        regularizer=self.regularizer)

      # layer_sensivity == 'l2_l2'
      if self.distributions in ['l2', 'weibull']:
        sensitivity_rescaling = np.ceil(filter_size / stride)
        k = kernel / sensitivity_rescaling
        x = tf.nn.conv2d(x, k, strides, padding='SAME')

      # layer_sensivity == 'l1_l1'
      elif self.distributions in ['l1', 'exp']:
        k = self.l1_normalize(kernel, dim=[0, 1, 3])
        x = tf.nn.conv2d(x, k, strides, padding='SAME')

      else:
         raise ValueError('sensitivity_norm is not recognised')

    noise = self._get_noise(x)
    if self.config['learn_noise_defense']:
      logging.info('parameterized noise defense activated')
      shape = noise.shape.as_list()[1:]
      noise = tf.layers.flatten(noise)
      feature_size = noise.shape.as_list()[-1]
      weights = tf.get_variable('weights_noise', (feature_size, ),
        initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/feature_size)),
        regularizer=self.regularizer)
      biases = tf.get_variable('biases_noise', (feature_size, ),
        regularizer=self.regularizer)
      noise = tf.multiply(noise, weights) + biases
      logging.info('get learned noise summary')
      tf.summary.histogram('weights_noise', weights)
      tf.summary.histogram('biases_noise', biases)
      tf.summary.histogram('noise', noise)
      noise = tf.reshape(noise, (-1, *shape))
      tf.add_to_collection('learned_noise', noise)

    return x + noise

  def _conv(self, name,  x, filter_size, in_filters, out_filters, strides):

    assert(strides[1] == strides[2])
    stride = strides[1]

    n = filter_size * filter_size * out_filters
    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
    kernel_size = (filter_size, filter_size)
    x = tf.layers.conv2d(x, out_filters, kernel_size,
                         strides=(stride, stride),
                         padding='same',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=self.regularizer,
                         use_bias=False,
                         name=name)
    return x

  def _batch_normalization(self, x):
     return tf.layers.batch_normalization(x,
               training=self.is_training,
               beta_regularizer=self.regularizer,
               gamma_regularizer=self.regularizer)

  def _residual(self, x, in_filter, out_filter, stride):
    """Residual unit with 2 sub layers."""

    x_orig = x
    strides = [1, stride, stride, 1]

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, strides)
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)
    with tf.variable_scope('sub2'):
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)
      if self.dropout > 0:
        x = tf.layers.dropout(x, rate=self.dropout, training=self.is_training)

    if in_filter == out_filter:
      x = x + x_orig
    else:
      x = x + self._conv('conv_orig', x_orig, 1, in_filter, out_filter, strides)

    x = self._batch_normalization(x)
    x = tf.nn.leaky_relu(x, self.leaky_slope)
    return x

  def _unit(self, x, in_filter, out_filter, n, stride, unit_id):
    for i in range(n):
      with tf.variable_scope('group{}_block_{}'.format(unit_id, i)):
        x = self._residual(x, in_filter, out_filter, stride if i == 0 else 1)
    return x

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.config = config = FLAGS.wide_resnet
    self.n_classes = n_classes
    self.is_training = is_training
    self.k = config['widen_factor']
    self.depth = config['depth']
    self.leaky_slope = config['leaky_slope']
    self.dropout = config['dropout']
    self.train_with_noise = config['train_with_noise']
    self.distributions = config['distributions']
    self.scale_noise = config['scale_noise']

    assert(self.depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (self.depth - 4) // 6
    filters = [16, 16*self.k, 32*self.k, 64*self.k]

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      self.regularizer = None
    else:
      self.regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    x = model_input

    with tf.variable_scope('init'):
      filter_size = 3
      in_filters  = x.get_shape()[-1]
      out_filters = 16
      if self.train_with_noise:
        x = self._conv_with_noise(
          "init_conv", x, filter_size, in_filters, out_filters, [1, 1, 1, 1])
      else:
        x = self._conv(
          "init_conv", x, filter_size, in_filters, out_filters, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)

    x = self._unit(x, filters[0], filters[1], n, 1, 1)
    x = self._unit(x, filters[1], filters[2], n, 2, 2)
    x = self._unit(x, filters[2], filters[3], n, 2, 3)

    with tf.variable_scope('unit_last'):
      x = tf.layers.average_pooling2d(x, [8, 8], [1, 1])
      x = tf.layers.flatten(x)

    with tf.variable_scope('logits') as scope:
      feature_size = x.get_shape().as_list()[-1]
      stddev = 1/np.sqrt(feature_size)
      kernel_initializer = tf.random_normal_initializer(stddev=stddev)
      logits = tf.layers.dense(x, n_classes, use_bias=True,
         kernel_initializer=kernel_initializer,
         kernel_regularizer=self.regularizer,
         bias_regularizer=self.regularizer,
         activation=None)

    return logits


class WideResnetModelNoiseImage(BaseModel):
  """Wide ResNet model.
     https://arxiv.org/abs/1605.07146
  """

  def _get_noise(self, x):
    """Pixeldp noise layer."""

    train_with_noise = self.is_training and self.train_with_noise
    if train_with_noise or (self.train_with_noise and FLAGS.noise_in_eval):
      noise_activate = tf.constant(1.0)
      logging.info(
        "train/eval with noise in img - noise sd {:.2f}".format(self.scale_noise))
    else:
      noise_activate = tf.constant(0.0)

    loc = tf.zeros(tf.shape(x), dtype=tf.float32)
    scale = tf.ones(tf.shape(x), dtype=tf.float32)

    if self.distributions == 'l1':
      noise = tf.distributions.Laplace(loc, scale).sample()
      noise = noise_activate * (self.scale_noise / tf.sqrt(2.)) * noise

    elif self.distributions == 'l2':
      noise = tf.distributions.Normal(loc, scale).sample()
      noise = noise_activate * self.scale_noise * noise

    elif self.distributions == 'exp':
      noise = tf.distributions.Exponential(rate=scale).sample()
      noise = noise_activate * self.scale_noise * noise

    elif self.distributions == 'weibull':
      k = 3
      eps = 10e-8
      alpha = ((k - 1) / k)**(1 / k)
      U = tf.random_uniform(tf.shape(x), minval=eps, maxval=1)
      X = (-tf.log(U) + alpha**k)**(1 / k) - alpha
      tensor = tf.zeros_like(x) + 0.5
      bernoulli = tf.distributions.Bernoulli(probs=tensor).sample()
      B = tf.cast(2 * bernoulli - 1, tf.float32)
      noise = B * X / 0.3425929
      noise = noise_activate * self.scale_noise * noise

    else:
      raise ValueError("Distributions is not recognised.")

    return noise

  def _conv(self, name,  x, filter_size, in_filters, out_filters, strides):

    assert(strides[1] == strides[2])
    stride = strides[1]

    n = filter_size * filter_size * out_filters
    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
    kernel_size = (filter_size, filter_size)
    x = tf.layers.conv2d(x, out_filters, kernel_size,
                         strides=(stride, stride),
                         padding='same',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=self.regularizer,
                         use_bias=False,
                         name=name)
    return x

  def _batch_normalization(self, x):
     return tf.layers.batch_normalization(x,
               training=self.is_training,
               beta_regularizer=self.regularizer,
               gamma_regularizer=self.regularizer)

  def _residual(self, x, in_filter, out_filter, stride):
    """Residual unit with 2 sub layers."""

    x_orig = x
    strides = [1, stride, stride, 1]

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, strides)
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)
    with tf.variable_scope('sub2'):
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)
      if self.dropout > 0:
        x = tf.layers.dropout(x, rate=self.dropout, training=self.is_training)

    if in_filter == out_filter:
      x = x + x_orig
    else:
      x = x + self._conv('conv_orig', x_orig, 1, in_filter, out_filter, strides)

    x = self._batch_normalization(x)
    x = tf.nn.leaky_relu(x, self.leaky_slope)
    return x

  def _unit(self, x, in_filter, out_filter, n, stride, unit_id):
    for i in range(n):
      with tf.variable_scope('group{}_block_{}'.format(unit_id, i)):
        x = self._residual(x, in_filter, out_filter, stride if i == 0 else 1)
    return x

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    """Build the core model within the graph."""

    self.config = config = FLAGS.wide_resnet
    self.n_classes = n_classes
    self.is_training = is_training
    self.k = config['widen_factor']
    self.depth = config['depth']
    self.leaky_slope = config['leaky_slope']
    self.dropout = config['dropout']
    self.train_with_noise = config['train_with_noise']
    self.distributions = config['distributions']
    self.scale_noise = config['scale_noise']

    assert(self.depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (self.depth - 4) // 6
    filters = [16, 16*self.k, 32*self.k, 64*self.k]

    reg_fn = getattr(tf.keras.regularizers, FLAGS.reg_norm, None)
    if reg_fn is None:
      self.regularizer = None
    else:
      self.regularizer = reg_fn(l=FLAGS.weight_decay_rate)

    x = model_input

    with tf.variable_scope('init'):
      filter_size = 3
      in_filters  = x.get_shape()[-1]
      out_filters = 16

      x = x + self._get_noise(x)
      x = self._conv(
        "init_conv", x, filter_size, in_filters, out_filters, [1, 1, 1, 1])
      x = self._batch_normalization(x)
      x = tf.nn.leaky_relu(x, self.leaky_slope)

    x = self._unit(x, filters[0], filters[1], n, 1, 1)
    x = self._unit(x, filters[1], filters[2], n, 2, 2)
    x = self._unit(x, filters[2], filters[3], n, 2, 3)

    with tf.variable_scope('unit_last'):
      x = tf.layers.average_pooling2d(x, [8, 8], [1, 1])
      x = tf.layers.flatten(x)

    with tf.variable_scope('logits') as scope:
      feature_size = x.get_shape().as_list()[-1]
      stddev = 1/np.sqrt(feature_size)
      kernel_initializer = tf.random_normal_initializer(stddev=stddev)
      logits = tf.layers.dense(x, n_classes, use_bias=True,
         kernel_initializer=kernel_initializer,
         kernel_regularizer=self.regularizer,
         bias_regularizer=self.regularizer,
         activation=None)

    return logits
