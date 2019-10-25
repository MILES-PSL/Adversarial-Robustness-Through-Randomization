#!/usr/bin/env python3

import os
import json
import argparse
import socket
from os.path import isdir, exists, join
from datetime import datetime

attacks = "fgm pgd carlini elasticnet"
date_format = "%Y-%m-%d_%H.%M.%S_%f"

script = """
CONFIG_PATH="$PROJECTDIR/{config_folder}/{config}.yaml"
TRAIN_DIR="{path}/{date}"
LOGS_DIR="{path}/{date}_logs"
mkdir $LOGS_DIR
cp $CONFIG_PATH $LOGS_DIR"/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu_train}';
python3 $PROJECTDIR/code/{train}.py \\
  --config_file=$CONFIG_PATH \\
  --config_name=train \\
  --train_dir=$TRAIN_DIR \\
  --data_dir=$DATADIR \\
  &>> $LOGS_DIR"/log_train.logs" &

export CUDA_VISIBLE_DEVICES='{gpu_eval}';
python3 $PROJECTDIR/code/eval.py \\
  --config_file=$CONFIG_PATH \\
  --config_name=eval_test \\
  --train_dir=$TRAIN_DIR \\
  --data_dir=$DATADIR \\
  &>> $LOGS_DIR"/log_eval_test.logs" &
"""

script_attacks = """
wait
export CUDA_VISIBLE_DEVICES='{gpu_attacks}';
for ATTACK in {attacks}
do
  python3 $PROJECTDIR/code/eval.py \\
    --config_file=$CONFIG_PATH \\
    --config_name=$ATTACK \\
    --train_dir=$TRAIN_DIR \\
    --data_dir=$DATADIR \\
    &>> "$LOGS_DIR/log_"$ATTACK".logs" &
  wait
done
"""


def get_name_id(outdir, name):
  id_ = 0
  while exists(join(
    outdir,  'config_{}_{}.yaml'.format(name, id_))):
    id_ += 1
  return 'config_{}_{}'.format(name, id_)

def make_config(args):
  if not args['name']:
    raise ValueError("Params is set. Name is are required")
  # load the template and populate the values
  projectdir = os.environ['PROJECTDIR']
  template = join(projectdir, 'config', '{}.yaml'.format(args['config']))
  with open(template) as f:
   template = f.read()
  config = template.format(**json.loads(args['params']))
  # save new config file in config_gen 
  outdir = join(projectdir, 'config_gen')
  # check if config_gen directory exists in PROJECTDIR
  # create the folder if it does not exist
  if not exists(outdir):
    os.mkdir(outdir)
  # save the config on disk 
  config_name = get_name_id(outdir,  args['name'])
  config_path = join(outdir, config_name)
  with open(config_path+'.yaml', "w") as f:
    f.write(config)
  return config_name

def main(args):
  global script

  # define folder name for training
  args['date'] = datetime.now().strftime(date_format)[:-2]
  args['train'] = "train"
  # check if config file exist
  projectdir = os.environ['PROJECTDIR']
  assert exists(
    join(projectdir, 'config', "{}.yaml".format(args['config']))), \
      "config file '{}' does not exist".format(args['config'])

  # check if path to model folder exists
  assert isdir(args['path']), \
      "path '{}' does not exist".format(args['path'])

  # if attacks is define, activate attacks after training
  if args['attacks']:
    if args['attacks'] == 'all':
      args['n_gpus_attacks'] = len(args['gpu_attacks'].split(','))
      args['attacks'] = attacks
    script += script_attacks
    args['attacks'] = list(dict.fromkeys(args['attacks'].split(' ')))
    assert set(args['attacks']).issubset(attacks.split(' ')), \
      "attacks not found"
    mapping = lambda x: "'attack_{}'".format(x)
    args['attacks'] = list(map(mapping, args['attacks']))
    args['attacks'] = ' '.join(args['attacks'])

  # if params is set, generate config file
  if args['params']:
    args['config_folder'] = 'config_gen'
    args['config'] = make_config(args)
  else:
    args['config_folder'] = 'config'

  print(script.format(**args))


if __name__ == '__main__':

  # default path 
  path = "{}/models".format(os.environ['WORKDIR'])

  parser = argparse.ArgumentParser(
      description='Run attacks on trained models.')
  parser.add_argument("config", type=str,
                        help="Config file to use for training.")
  parser.add_argument("--path", type=str, default=path,
                        help="Set path of trained folder.")
  parser.add_argument("--attacks", type=str, default='',
                        help="List of attacks to perform.")
  parser.add_argument("--train", type=str,
                        help="Set train scheme.")
  parser.add_argument("--gpu_train", type=str, default="0,1",
                        help="Set CUDA_VISIBLE_DEVICES for training")
  parser.add_argument("--gpu_eval", type=str, default="2,3",
                        help="Set CUDA_VISIBLE_DEVICES for eval.")
  parser.add_argument("--gpu_attacks", type=str, default="0",
                        help="Set CUDA_VISIBLE_DEVICES for attacks.")

  # paramters for batch experiments
  parser.add_argument("--params", type=str, default='',
            help="Parameters to override in the config file.")
  parser.add_argument("--name", type=str, default='',
            help="Name of the batch experiments. Required if params is set.")
  args = vars(parser.parse_args())
  main(args)

