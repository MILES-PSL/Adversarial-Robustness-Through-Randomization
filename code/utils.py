
import numpy as np
import tensorflow as tf


def make_summary(name, value, summary_writer, global_step_val):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  summary_writer.add_summary(summary, global_step_val)


def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary

def summary_histogram(writer, tag, values, step, bins=1000):

  # Create histogram using numpy
  counts, bin_edges = np.histogram(values, bins=bins)

  # Fill fields of histogram proto
  hist = tf.HistogramProto()
  hist.min = float(np.min(values))
  hist.max = float(np.max(values))
  hist.num = int(np.prod(values.shape))
  hist.sum = float(np.sum(values))
  hist.sum_squares = float(np.sum(values**2))

  # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
  # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
  # Thus, we drop the start of the first bin
  bin_edges = bin_edges[1:]

  # Add bin edges and counts
  for edge in bin_edges:
      hist.bucket_limit.append(edge)
  for c in counts:
      hist.bucket.append(c)

  # Create and write Summary
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
  writer.add_summary(summary, step)
  writer.flush()


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias

def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
  """Add the global_step summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    global_step_info_dict: a dictionary of the evaluation metrics calculated for
      a mini-batch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  this_hit_at_one = global_step_info_dict["hit_at_one"]
  this_perr = global_step_info_dict["perr"]
  this_loss = global_step_info_dict["loss"]
  examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

  if examples_per_second != -1:
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                    examples_per_second), global_step_val)

  summary_writer.flush()
  info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
          "| Examples_per_sec: {4:.3f}").format(
              global_step_val, this_hit_at_one, this_perr, this_loss,
              examples_per_second)
  return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
  """Add the epoch summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  epoch_id = epoch_info_dict["epoch_id"]
  avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
  avg_perr = epoch_info_dict["avg_perr"]
  avg_loss = epoch_info_dict["avg_loss"]
  aps = epoch_info_dict["aps"]
  gap = epoch_info_dict["gap"]
  mean_ap = np.mean(aps)

  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_MAP", mean_ap),
          global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
          global_step_val)
  summary_writer.flush()

  info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
          "| MAP: {3:.3f} | GAP: {4:.3f} | Avg_Loss: {5:3f}").format(
          epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss)
  return info


class MessageBuilder:

  def __init__(self):
    self.msg = []

  def add(self, name, values, align=">", width=0, format=None):
    if name:
      metric_str = "{}: ".format(name)
    else:
      metric_str = ""
    values_str = []
    if type(values) != list:
      values = [values]
    for value in values:
      if format:
        values_str.append("{value:{align}{width}{format}}".format(
          value=value, align=align, width=width, format=format))
      else:
        values_str.append("{value:{align}{width}}".format(
          value=value, align=align, width=width))
    metric_str += '/'.join(values_str)
    self.msg.append(metric_str)

  def get_message(self):
    return " | ".join(self.msg)


