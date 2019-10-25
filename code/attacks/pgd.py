"""
The ProjectedGradientDescent attack.
"""
import numpy as np
import tensorflow as tf

from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta, clip_by_value

from .fgm import FastGradientMethod


class ProjectedGradientDescent:
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param default_rand_init: whether to use random initialization by default
  :param kwargs: passed through to super constructor
  """

  FGM_CLASS = FastGradientMethod

  def __init__(self,
               batch_size=1,
               rand_minmax=0.3,
               eps=0.3,
               eps_iter=0.05,
               nb_iter=10,
               ord=np.inf,
               clip_min=None,
               clip_max=None,
               y_target=None,
               sample=1,
               sanity_checks=False):
    """
    Create a ProjectedGradientDescent instance.
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    Attack-specific parameters:
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.batch_size = batch_size
    self.eps = eps
    self.rand_init = rand_minmax > 0
    self.rand_minmax = rand_minmax
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y_target = y_target
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sample = sample
    self.sanity_checks = sanity_checks

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.ord == 'inf':
      self.ord = np.inf

    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

  def get_name(self):
    return 'ProjectedGradientDescent_{}_{}_{}_{}_{}_{}'.format(
      self.rand_minmax, self.eps, self.eps_iter, self.nb_iter, self.sample,
      self.ord)

  def generate(self, x, fn_logits, y=None):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    self.y = y
    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.sanity_checks and self.clip_min is not None:
      asserts.append(
        utils_tf.assert_greater_equal(x, tf.cast(self.clip_min, x.dtype)))

    if self.sanity_checks and self.clip_max is not None:
      asserts.append(
        utils_tf.assert_less_equal(x, tf.cast(self.clip_max, x.dtype)))

    # Initialize loop variables
    if self.rand_init:
      eta = tf.random_uniform(tf.shape(x),
                              tf.cast(-self.rand_minmax, x.dtype),
                              tf.cast(self.rand_minmax, x.dtype),
                              dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, self.ord, self.eps)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      adv_x = clip_by_value(adv_x, self.clip_min, self.clip_max)

    if self.y_target is not None:
      y = self.y_target
      targeted = True
    elif self.y is not None:
      y = self.y
      targeted = False
    else:
      model_preds = fn_logits(x)
      preds_max = tf.reduce_max(model_preds, 1, keepdims=True)
      y = tf.to_float(tf.equal(model_preds, preds_max))
      y = tf.stop_gradient(y)
      targeted = False
      del model_preds

    fgm_params = {
        'batch_size': self.batch_size,
        'eps': self.eps_iter,
        'ord': self.ord,
        'clip_min': self.clip_min,
        'clip_max': self.clip_max,
        'sample': self.sample,
        'sanity_checks': self.sanity_checks
    }
    if self.ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                " step for PGD when ord=1, because ord=1 FGM "
                                " changes only one pixel at a time. We need "
                                " to rigorously test a strong ord=1 PGD "
                                "before enabling this feature.")

    fgm_attack = self.FGM_CLASS(**fgm_params)

    # def cond(i, _):
    #   return tf.less(i, self.nb_iter)

    # def body(i, adv_x):
    #   adv_x = fgm_attack.generate(adv_x, fn_logits)

    #   # Clipping perturbation eta to self.ord norm ball
    #   eta = adv_x - x
    #   eta = clip_eta(eta, self.ord, self.eps)
    #   adv_x = x + eta

    #   # Redo the clipping.
    #   # FGM already did it, but subtracting and re-adding eta can add some
    #   # small numerical error.
    #   if self.clip_min is not None or self.clip_max is not None:
    #     adv_x = clip_by_value(adv_x, self.clip_min, self.clip_max)

    #   return i + 1, adv_x

    # _, adv_x = tf.while_loop(cond, body, (tf.zeros([]), adv_x), back_prop=True,
    #                          maximum_iterations=self.nb_iter)

    for i in range(self.nb_iter):
      adv_x = fgm_attack.generate(adv_x, fn_logits)

      # Clipping perturbation eta to self.ord norm ball
      eta = adv_x - x
      eta = clip_eta(eta, self.ord, self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
      if self.clip_min is not None or self.clip_max is not None:
        adv_x = clip_by_value(adv_x, self.clip_min, self.clip_max)

    # Asserts run only on CPU.
    # When multi-GPU eval code tries to force all PGD ops onto GPU, this
    # can cause an error.
    if self.sanity_checks:
      common_dtype = tf.float64
      asserts.append(utils_tf.assert_less_equal(
        tf.cast(self.eps_iter, dtype=common_dtype),
        tf.cast(self.eps, dtype=common_dtype)))
    if self.sanity_checks and self.ord == np.inf and self.clip_min is not None:
      # The 1e-6 is needed to compensate for numerical error.
      # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
      # clip_max=.7
      asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps, x.dtype),
                                                1e-6 + tf.cast(self.clip_max,
                                                               x.dtype)
                                                - tf.cast(self.clip_min,
                                                          x.dtype)))

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

