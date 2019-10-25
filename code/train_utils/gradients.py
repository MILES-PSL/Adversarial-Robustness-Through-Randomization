"""Contains a collection of util functions for training and evaluating.
"""
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import logging

from utils import summary_histogram
from config import hparams as FLAGS


def compute_hessian_and_summary(sess, summary_writer, global_step):
   """compute the full exact hessian from Hessian vector product"""

   vars_list = sess.run(tf.trainable_variables())
   total_params = 0
   var_shape_flatten = []
   var_shape = []
   for var in vars_list:
     shape = var.shape
     size = np.prod(shape)
     total_params += size
     var_shape.append(shape)
     var_shape_flatten.append(size)

   var_index = {}
   cumul = 0
   for i, size in enumerate(var_shape_flatten):
     for x in range(size):
       var_index[x+cumul] = (i, x)
     cumul += size

   hessian = np.zeros((total_params, total_params))
   vec_placeholder = tf.get_collection("vec_placeholder")
   Hv = tf.get_collection("Hv")[0]
   logging.info("computing hessian")
   for ix in range(total_params):

     # create a buch a zero vectors
     vec = [np.zeros((size, ), dtype=np.float32) for size in var_shape_flatten]
     i, x = var_index[ix]
     vec[i][x] = 1

     # add all zeros vec to data feed
     data_to_feed = {}
     for i in range(len(vec)):
       data_to_feed[vec_placeholder[i]] = np.reshape(vec[i], var_shape[i])
     Hv_ = sess.run(Hv, feed_dict=data_to_feed)
     Hv_ = np.concatenate([np.ravel(v) for v in Hv_])
     hessian[:, ix] = Hv_

   logging.info("computing eigenvalues of the hessian")
   hessian = np.nan_to_num(hessian)
   eig, _ = np.linalg.eig(hessian)
   eig = np.real(eig)
   neg_eig = (eig < 0).sum()
   pos_eig = (eig > 0).sum()
   zero_eig = (eig == 0).sum()
   make_summary("hessian/eigenvalues/neg", neg_eig, summary_writer, global_step)
   make_summary("hessian/eigenvalues/pos", pos_eig, summary_writer, global_step)
   make_summary("hessian/eigenvalues/zero", zero_eig, summary_writer, global_step)
   summary_histogram(summary_writer, "hessian/eigenvalues", eig, global_step)


def _hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.
  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.
  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.
  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.
  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.
  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.
  Raises:
    ValueError: `xs` and `v` have different length.
  """

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = tf.gradients(ys, xs)

  assert len(grads) == length
  elemwise_products = [
      tf.multiply(grad_elem, tf.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v)
      if grad_elem is not None
  ]

  # Second backprop
  return tf.gradients(elemwise_products, xs)


def combine_gradients(tower_grads):
  """Calculate the combined gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
     across all towers.
  """
  filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
  final_grads = []
  for i in range(len(filtered_grads[0])):
    grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
    grad = tf.stack([x[0] for x in grads], 0)
    grad = tf.reduce_sum(grad, 0)
    final_grads.append((grad, filtered_grads[0][i][1],))

  return final_grads


class ComputeAndProcessGradients:

  def __init__(self):
    self.config = FLAGS.gradients

  def _clip_gradient_norms(self, gradients_to_variables, max_norm):
    """Clips the gradients by the given value.

    Args:
      gradients_to_variables: A list of gradient to variable pairs (tuples).
      max_norm: the maximum norm value.

    Returns:
      A list of clipped gradient to variable pairs.
    """
    with tf.name_scope('clip_gradients'):
      clipped_grads_and_vars = []
      for grad, var in gradients_to_variables:
        if grad is not None:
          if isinstance(grad, tf.IndexedSlices):
            tmp = tf.clip_by_norm(grad.values, max_norm)
            grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
          else:
            grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))
      return clipped_grads_and_vars

  def _perturbed_gradients(self, gradients):
    with tf.name_scope('perturbed_gradients'):
      gradients_norm = tf.constant(0.)
      for grad, var in gradients:
        if grad is not None:
          norm = tf.norm(grad)
          gradients_norm = gradients_norm + norm
          tf.summary.scalar(var.op.name + '/gradients/norm', norm)
      gradients_norm = gradients_norm / len(gradients)
      tf.add_to_collection("gradients_norm", gradients_norm)

      norm_threshold = self.config['perturbed_threshold']
      activate_noise = tf.cond(tf.less(gradients_norm, norm_threshold),
                       true_fn=lambda: tf.constant(1.),
                       false_fn=lambda: tf.constant(0.))

      gradients_noise = []
      for grad, var in gradients:
        Y = tf.random_normal(shape=grad.shape)
        U = tf.random_uniform(shape=grad.shape, minval=0, maxval=1)
        noise = tf.sqrt(U) * (Y / tf.norm(Y)) * activate_noise
        tf.summary.histogram(var.op.name + '/gradients/noise', noise)
        grad = grad + noise
        gradients_noise.append((grad, var))
      return gradients_noise

  def _gradients_summary(self, gradients):
    for grad, var in gradients:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

  def _define_hessian_graph(self, loss):
    """Compute the hessian of the training variables"""
    with tf.name_scope("hessian"):
      var_list = tf.trainable_variables()
      vec_placeholder = []
      for var in var_list:
        v_placeholder = tf.placeholder(tf.float32, shape=var.get_shape())
        vec_placeholder.append(v_placeholder)
        tf.add_to_collection("vec_placeholder", v_placeholder)
      Hv = _hessian_vector_product(loss, var_list, vec_placeholder)
      tf.add_to_collection("Hv", Hv)

  def get_gradients(self, opt, loss, *args, **kwargs):

    gradients = opt.compute_gradients(loss, *args, **kwargs)
    if self.config['make_gradient_summary']:
      self._gradients_summary(gradients)

    # compute and record summary of hessians eigenvals
    if self.config['compute_hessian']:
      self._define_hessian_graph(loss)

    # to help convergence, inject noise in gradients
    if self.config['perturbed_gradients']:
      gradients = self._perturbed_gradients(gradients)
    else:
      tf.add_to_collection("gradients_norm", 0)

    # to regularize, clip the value of the gradients
    if self.config['clip_gradient_norm'] > 0:
      gradients = self._clip_gradient_norms(
        gradients, self.config['clip_gradient_norm'])
    return gradients

