
import numpy as np
import tensorflow as tf

class FastGradientMethod:
  """
  This attack was originally implemented by Goodfellow et al. (2015) with the
  infinity norm (and is known as the "Fast Gradient Sign Method"). This
  implementation extends the attack to other norms, and is therefore called
  the Fast Gradient Method.
  Paper link: https://arxiv.org/abs/1412.6572
  """
  def __init__(self, batch_size=1, eps=0.3, ord=np.inf, clip_min=None,
               clip_max=None, targeted=False, sanity_checks=False, sample=1):
    """
    Create a FastGradientMethod instance.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :params sample: take the esperance of the logits to attack randomized
                         model.
    """
    self.batch_size = batch_size
    self.eps = eps
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.targeted = False
    self.sample = sample
    self.sanity_checks = sanity_checks

    if self.ord == 'inf':
      self.ord = np.inf

  def get_name(self):
    return 'FastGradientMethod_{}_{}_{}'.format(
      str(self.ord), self.eps, self.sample)

  def _optimize_linear(self, grad, eps, ord=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)
    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param ord: int specifying order of norm
    :returns:
      tf tensor containing optimal perturbation
    """

    red_ind = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
      # Take sign of gradient
      optimal_perturbation = tf.sign(grad)
      # The following line should not change the numerical results.
      # It applies only because `optimal_perturbation` is the output of
      # a `sign` op, which has zero derivative anyway.
      # It should not be applied for the other norms, where the
      # perturbation has a non-zero derivative.
      optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif ord == 1:
      abs_grad = tf.abs(grad)
      sign = tf.sign(grad)
      max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
      tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
      num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
      optimal_perturbation = sign * tied_for_max / num_ties
    elif ord == 2:
      square = tf.maximum(avoid_zero_div,
                          tf.reduce_sum(tf.square(grad),
                                     reduction_indices=red_ind,
                                     keepdims=True))
      optimal_perturbation = grad / tf.sqrt(square)
      optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    else:
      raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation

  def generate(self, x, fn_logits, y=None):
    """
    Returns the graph for Fast Gradient Method adversarial examples.
    :param x: The model's symbolic inputs.
    :param fn_logits: function to compute the logits
    :param y: (optional) A placeholder for the true labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :return: a tensor for the adversarial example
    """
    x_orig = x
    # compute logits
    if self.sample <= 1:
      logits = fn_logits(x)
      assert logits.op.type != 'Softmax'

      # Using model predictions as ground truth to avoid label leaking
      preds_max = tf.reduce_max(logits, 1, keepdims=True)
      y = tf.to_float(tf.equal(logits, preds_max))
      y = tf.stop_gradient(y)
      y = y / tf.reduce_sum(y, 1, keepdims=True)

      # Compute loss
      loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    else:
      tf.logging.info(
        "Monte Carlo (MC) on attacks, sample: {}".format(self.sample))
      eot_loss = []
      for i in range(self.sample):
        logits = fn_logits(x)
        if i == 0:
          assert logits.op.type != 'Softmax'

        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
        y = y / tf.reduce_sum(y, 1, keepdims=True)

        # Compute loss
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
        eot_loss.append(loss)
      loss = tf.reduce_mean(eot_loss, 0)


      # _, *shape = x.shape.as_list()
      # x = tf.layers.flatten(x)
      # _, dim = x.shape.as_list()
      # x = tf.tile(x, (1, self.sample))
      # x = tf.reshape(x, (-1, *shape))
      # logits = fn_logits(x)
      # assert logits.op.type != 'Softmax'

      # # Using model predictions as ground truth to avoid label leaking
      # preds_max = tf.reduce_max(logits, 1, keepdims=True)
      # y = tf.to_float(tf.equal(logits, preds_max))
      # y = tf.stop_gradient(y)
      # y = y / tf.reduce_sum(y, 1, keepdims=True)

      # # Compute loss
      # loss = tf.losses.softmax_cross_entropy_with_logits(labels=y, logits=logits)
      # loss = tf.reshape(loss, (-1, self.sample))
      # loss = tf.reduce_mean(loss, axis=1)

    if self.targeted:
      loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x_orig)

    optimal_perturbation = self._optimize_linear(grad, self.eps, self.ord)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x_orig + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if self.sanity_checks and \
       ( (self.clip_min is not None) or (self.clip_max is not None) ):
      # We don't currently support one-sided clipping
      assert self.clip_min is not None and self.clip_max is not None
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    return adv_x


