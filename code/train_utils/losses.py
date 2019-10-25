"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf

from config import hparams as FLAGS

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, labels=None, logits=None, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      predictions = tf.nn.sigmoid(logits)
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class SoftmaxCrossEntropyWithLogits(BaseLoss):

  def calculate_loss(self, labels=None, logits=None, **unused_params):
    with tf.name_scope("loss"):
      if FLAGS.gradients["compute_hessian"] or FLAGS.fused_loss is False:
        if not FLAGS.one_hot_labels:
          raise ValueError("Labels needs to be one hot encoded.")
        # We are going to compute the hessian so we can't 
        # use the tf.losses.sparse_softmax_cross_entropy
        # because the ops are fused and the gradient of the 
        # cross entropy is blocked. Insted we compute it manually.
        # /!\ might not be efficient and not numerically stable
        epsilon = 10e-6
        predictions = tf.nn.softmax(logits)
        float_labels = tf.cast(labels, tf.float32)
        cross_entropy_loss = float_labels * tf.log(predictions + epsilon)
        cross_entropy_loss = tf.negative(cross_entropy_loss)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
      return tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

class SigmoidCrossEntropyWithLogits(BaseLoss):

  def calculate_loss(self, labels=None, logits=None, **unused_params):
    with tf.name_scope('loss'):
      return tf.losses.sigmoid_cross_entropy(
                  labels, logits, weights=1.0, label_smoothing=0)

class MeanSquareError(BaseLoss):

  def calculate_loss(self, labels=None, logits=None, **unused_params):
    with tf.name_scope('loss'):
      return tf.losses.mean_squared_error(
                  labels, logits, weights=1.0, label_smoothing=0)
