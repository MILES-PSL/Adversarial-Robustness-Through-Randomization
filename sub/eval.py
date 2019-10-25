#!/usr/bin/env python3

import os
import json
import socket
import argparse
from os.path import join, exists

script = """
TRAIN_DIR="{path}/{folder}"
LOGS_DIR=$TRAIN_DIR"_logs"
CONFIG_FILE=$LOGS_DIR"/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu}';
python3 $PROJECTDIR/code/eval.py \\
  --config_file=$CONFIG_FILE \\
  --config_name=eval_test \\
  --train_dir=$TRAIN_DIR \\
  --data_dir=$DATADIR {params} \\
  &>> $LOGS_DIR"/log_eval_test.logs" &
"""

def main(args):
  global script

  train_dir = join(args['path'], args['folder'])
  assert exists(train_dir), "{} does not exist".format(train_dir)

  # if params is set, override the parameters in the config file
  if args['params']:
    try:
      _ = json.loads(args['params'])
      args['params'] = "--params '{}'".format(args['params'])
    except:
      raise ValueError("Could not parse override parameters")
  print(script.format(**args))


if __name__ == '__main__':

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  parser = argparse.ArgumentParser(
      description='Run attacks on trained models.')
  parser.add_argument("folder", type=str,
                        help="Folder of the trained models.")
  parser.add_argument("--path", type=str, default=path,
                        help="path of the trained folder.")
  parser.add_argument("--gpu", type=str, default="0,1,2,3",
                        help="Set CUDA_VISIBLE_DEVICES for eval.")

  # paramters for batch experiments
  parser.add_argument("--params", type=str, default='',
            help="Parameters to override in the config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if params is set.")
  args = vars(parser.parse_args())
  main(args)


