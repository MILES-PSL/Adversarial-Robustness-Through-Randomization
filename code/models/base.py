
import re
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def _activation_summary(self, x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    if self.is_training and not tf.executing_eagerly():
      # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
      # session. This helps the clarity of presentation on tensorboard.
      tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
      # tf.summary.histogram('{}/activations'.format(tensor_name), x)
      # tf.summary.scalar('{}/sparsity'.format(tensor_name), tf.nn.zero_fraction(x))

  def create_model(self, model_input, n_classes, is_training, *args, **kwargs):
    raise NotImplementedError()
