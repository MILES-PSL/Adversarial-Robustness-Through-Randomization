"""The CarliniWagnerL2 attack
"""
import numpy as np
import tensorflow as tf
from tensorflow import logging

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

class CarliniWagnerL2:
  """
  This attack was originally proposed by Carlini and Wagner. It is an
  iterative attack that finds adversarial examples on many defenses that
  are robust to other attacks.
  Paper link: https://arxiv.org/abs/1608.04644
  At a high level, this attack is an iterative attack using Adam and
  a specially-chosen loss function to find adversarial examples with
  lower distortion than other attacks. This comes at the cost of speed,
  as this attack is often much slower than others.
  """

  def __init__(self,
               y_target=None,
               batch_size=1,
               confidence=0,
               learning_rate=5e-3,
               binary_search_steps=9,
               max_iterations=1000,
               abort_early=True,
               initial_const=1e-2,
               clip_min=-1,
               clip_max=+1,
               sample=1):
    """
    :param y_target: (optional) A tensor with the target labels for a
              targeted attack.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param batch_size: Number of attacks to run simultaneously.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and confidence of the classification.
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param abort_early: If true, allows early aborts if gradient descent
                        is unable to make progress (i.e., gets stuck in
                        a local minimum).
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """
    self.y_target = y_target
    self.batch_size = batch_size
    self.confidence = confidence
    self.learning_rate = learning_rate
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.abort_early = abort_early
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sample = sample

    tf.logging.info("max_iterations: {}".format(max_iterations))
    tf.logging.info("batch_size: {}".format(batch_size))

  def get_name(self):
    return 'Carlini_{}_{}_{}'.format(
      self.max_iterations, self.binary_search_steps, self.sample)

  def generate(self, x, fn_logits, y=None):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.
    :param x: A tensor with the inputs.
    :param kwargs: See `parse_params`
    """
    preds = fn_logits(x)
    preds_max = tf.reduce_max(preds, 1, keepdims=True)
    original_predictions = tf.to_float(tf.equal(preds, preds_max))
    labels = tf.stop_gradient(original_predictions)
    nb_classes = labels.get_shape().as_list()[1]

    # the session is passed after 
    self.sess = None

    self.repeat = self.binary_search_steps >= 10

    shape = x.get_shape().as_list()[1:]
    self.shape = shape = tuple([self.batch_size] + list(shape))

    # the variable we're going to optimize over
    modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

    # these are variables to be more efficient in sending data to tf
    self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
    self.tlab = tf.Variable(
        np.zeros((self.batch_size, nb_classes)), dtype=tf_dtype, name='tlab')
    self.const = tf.Variable(
        np.zeros(self.batch_size), dtype=tf_dtype, name='const')

    # and here's what we use to assign them
    self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')
    self.assign_tlab = tf.placeholder(
        tf_dtype, (self.batch_size, nb_classes), name='assign_tlab')
    self.assign_const = tf.placeholder(
        tf_dtype, [self.batch_size], name='assign_const')

    # the resulting instance, tanh'd to keep bounded from clip_min
    # to clip_max
    self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
    self.newimg = self.newimg * (self.clip_max - self.clip_min) + self.clip_min

    # prediction BEFORE-SOFTMAX of the model
    def batch_prediction(img):
      shape_img = img.shape.as_list()[1:]
      img_sample = tf.layers.flatten(img)
      dim = img_sample.shape.as_list()[1:]
      img_sample = tf.tile(img_sample, (1, self.sample))
      img_sample = tf.reshape(img_sample, (self.batch_size*self.sample, *shape_img))
      logits = fn_logits(img_sample)
      assert logits.op.type != 'Softmax'
      _, dim = logits.shape.as_list()
      logits = tf.reshape(logits, (self.batch_size, self.sample, dim))
      output = tf.reduce_mean(logits, axis=1)
      return output

    if self.sample <= 1:
      self.output = fn_logits(self.newimg)
    else:
      tf.logging.info(
        "Monte Carlo (MC) on attacks, sample: {}".format(self.sample))
      tf.logging.info("batch_size: {}".format(self.batch_size))
      self.output = batch_prediction(self.newimg)
      # _, *shape_img = self.newimg.shape.as_list()
      # self.newimg_sample = tf.layers.flatten(self.newimg)
      # _, dim = self.newimg_sample.shape.as_list()
      # self.newimg_sample = tf.tile(
      #   self.newimg_sample, (1, self.sample))
      # self.newimg_sample = tf.reshape(
      #   self.newimg_sample, (self.batch_size*self.sample, *shape_img))
      # logits = fn_logits(self.newimg_sample)
      # assert logits.op.type != 'Softmax'
      # _, dim = logits.shape.as_list()
      # logits = tf.reshape(logits, (self.batch_size, self.sample, dim))
      # self.output = tf.reduce_mean(logits, axis=1)

    # distance to the input data
    self.other = (tf.tanh(self.timg) + 1) / \
        2 * (self.clip_max - self.clip_min) + self.clip_min
    self.l2dist = tf.reduce_sum(
        tf.square(self.newimg - self.other), list(range(1, len(shape))))

    # compute the probability of the label class versus the maximum other
    real = tf.reduce_sum((self.tlab) * self.output, 1)
    other = tf.reduce_max((1 - self.tlab) * self.output - self.tlab * 10000, 1)
    zero = np.asarray(0., dtype=np_dtype)
    if self.y_target:
      # if targeted, optimize for making the other class most likely
      loss1 = tf.maximum(zero, other - real + self.confidence)
    else:
      # if untargeted, optimize for making this class least likely.
      loss1 = tf.maximum(zero, real - other + self.confidence)

    # sum up the losses
    self.loss2 = tf.reduce_sum(self.l2dist)
    self.loss1 = tf.reduce_sum(self.const * loss1)
    self.loss = self.loss1 + self.loss2

    # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train = optimizer.minimize(self.loss, var_list=[modifier])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

    # these are the variables to initialize when we run
    self.setup = []
    self.setup.append(self.timg.assign(self.assign_timg))
    self.setup.append(self.tlab.assign(self.assign_tlab))
    self.setup.append(self.const.assign(self.assign_const))

    self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    # wrap attack function in py_func
    def cw_wrap(x_val, y_val):
      return self.attack(x_val, y_val)

    adv = tf.py_func(cw_wrap, [x, labels], tf_dtype)
    adv.set_shape(x.get_shape())
    return adv

  def attack(self, imgs, targets):
    """
    Perform the L_2 attack on the given instance for the given targets.
    If self.targeted is true, then the targets represents the target labels
    If self.targeted is false, then targets are the original class labels
    """

    r = []
    for i in range(0, len(imgs), self.batch_size):
      logging.debug(
          ("Running CWL2 attack on instance %s of %s", i, len(imgs)))
      adv = self.attack_batch(
        imgs[i:i + self.batch_size], targets[i:i + self.batch_size])
      r.extend(adv)
    return np.array(r)

  def attack_batch(self, imgs, labs):
    """
    Run the attack on a batch of instance and labels.
    """

    def compare(x, y):
      if not isinstance(x, (float, int, np.int64)):
        x = np.copy(x)
        if self.y_target:
          x[y] -= self.confidence
        else:
          x[y] += self.confidence
        x = np.argmax(x)
      if self.y_target:
        return x == y
      else:
        return x != y

    batch_size = self.batch_size

    oimgs = np.clip(imgs, self.clip_min, self.clip_max)

    # re-scale instances to be within range [0, 1]
    imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
    imgs = np.clip(imgs, 0, 1)
    # now convert to [-1, 1]
    imgs = (imgs * 2) - 1
    # convert to tanh-space
    imgs = np.arctanh(imgs * .999999)

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(batch_size)
    CONST = np.ones(batch_size) * self.initial_const
    upper_bound = np.ones(batch_size) * 1e10

    # placeholders for the best l2, score, and instance attack found so far
    o_bestl2 = [1e10] * batch_size
    o_bestscore = [-1] * batch_size
    o_bestattack = np.copy(oimgs)

    for outer_step in range(self.binary_search_steps):
      # completely reset adam's internal state.
      self.sess.run(self.init)
      batch = imgs[:batch_size]
      batchlab = labs[:batch_size]

      bestl2 = [1e10] * batch_size
      bestscore = [-1] * batch_size
      logging.debug("  Binary search step %s of %s",
                    outer_step, self.binary_search_steps)

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.binary_search_steps - 1:
        CONST = upper_bound

      # set the variables so that we don't have to send them over again
      self.sess.run(
          self.setup, {
              self.assign_timg: batch,
              self.assign_tlab: batchlab,
              self.assign_const: CONST
          })

      prev = 1e6
      for iteration in range(self.max_iterations):
        # perform the attack
        _, l, l2s, scores, nimg = self.sess.run([
            self.train, self.loss, self.l2dist, self.output,
            self.newimg
        ])

        if iteration % ((self.max_iterations // 10) or 1) == 0:
          logging.debug(("    Iteration {} of {}: loss={:.3g} " +
                         "l2={:.3g} f={:.3g}").format(
                             iteration, self.max_iterations, l,
                             np.mean(l2s), np.mean(scores)))

        # check if we should abort search if we're getting nowhere.
        if self.abort_early and \
           iteration % ((self.max_iterations // 10) or 1) == 0:
          if l > prev * .9999:
            msg = "    Failed to make progress; stop early"
            logging.debug(msg)
            break
          prev = l

        # adjust the best result found so far
        for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
          lab = np.argmax(batchlab[e])
          if l2 < bestl2[e] and compare(sc, lab):
            bestl2[e] = l2
            bestscore[e] = np.argmax(sc)
          if l2 < o_bestl2[e] and compare(sc, lab):
            o_bestl2[e] = l2
            o_bestscore[e] = np.argmax(sc)
            o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(batch_size):
        if compare(bestscore[e], np.argmax(batchlab[e])) and \
           bestscore[e] != -1:
          # success, divide const by two
          upper_bound[e] = min(upper_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e] = max(lower_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            CONST[e] *= 10
      logging.debug("  Successfully generated adversarial examples " +
                    "on {} of {} instances.".format(
                        sum(upper_bound < 1e9), batch_size))
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      logging.debug("   Mean successful distortion: {:.4g}".format(mean))

    # return the best solution found
    logging.info("  Successfully generated adversarial examples " +
                  "on {} of {} instances.".format(
                      sum(upper_bound < 1e9), batch_size))
    o_bestl2 = np.array(o_bestl2)
    mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
    logging.info("   Mean successful distortion: {:.4g}".format(mean))
    return o_bestattack



