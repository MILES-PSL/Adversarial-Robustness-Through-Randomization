#!/usr/bin/env python3

import os
import json
import socket
import argparse
from os.path import join, exists

attacks = "fgm pgd carlini elasticnet"

setup_ouessant = """
#BSUB -J {folder_id}_attack
#BSUB -gpu "num=4:mode=exclusive_process:mps=no:j_exclusive=yes"
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -W 1200
#BSUB -o {home}/%J.train.out
#BSUB -e {home}/%J.train.err

ml anaconda/py3 cudnn nccl
source activate tensorflow1.12-py3
export LD_LIBRARY_PATH={ld_library}
"""

script = """
TRAIN_DIR="{path}/{folder}"
LOGS_DIR=$TRAIN_DIR"_logs"
CONFIG_FILE="$LOGS_DIR/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu}';
for ATTACK in {attacks}
do
  python3 $PROJECTDIR/code/eval.py \\
    --config_file=$CONFIG_FILE \\
    --config_name=$ATTACK \\
    --train_dir=$TRAIN_DIR \\
    --data_dir=$DATADIR {params} \\
    &>> "$LOGS_DIR/log_$ATTACK.logs" &
  wait
done
"""

setup_fair = """#!/bin/bash
#SBATCH --job-name={folder_id}_attack
#SBATCH --output={home}/neuralnet/sample-%j.out
#SBATCH --error={home}/neuralnet/sample-%j.err
#SBATCH --time=4300
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:{n_gpus}
#SBATCH --get-user-env
#SBATCH --comment="NIPS deadline"

TRAIN_DIR="{path}/{folder}"
LOGS_DIR=$TRAIN_DIR"_logs"
CONFIG_FILE="$LOGS_DIR/model_flags.yaml"

export CUDA_VISIBLE_DEVICES='{gpu}';
for ATTACK in {attacks}
do
  srun -o "$LOGS_DIR/log_$ATTACK.logs" -u \\
    python3 $PROJECTDIR/code/eval.py \\
      --config_file=$CONFIG_FILE \\
      --config_name=$ATTACK \\
      --train_dir=$TRAIN_DIR \\
      --data_dir=$DATADIR {params}
done
"""



def main(args):
  global script

  train_dir = join(args['path'], args['folder'])
  assert exists(train_dir), "{} does not exist".format(train_dir)

  # check and processed attacks
  args['attacks'] = list(set(args['attacks'].split(' ')))
  assert set(args['attacks']).issubset(attacks.split(' ')), \
      "attacks not found"
  mapping = lambda x: "'attack_{}'".format(x)
  args['attacks'] = list(map(mapping, args['attacks']))
  args['attacks'] = ' '.join(args['attacks'])

  # if params is set, overide the parameters in the config file
  if args['params']:
    try:
      _ = json.loads(args['params'])
      args['params'] = "--params '{}'".format(args['params'])
    except:
      raise ValueError("Could not parse overide parameters")

  # setup ouessant job parameters 
  if "ouessant" in args['hostname']:
    script = setup_ouessant + script
    args['home'] = os.environ['home']
    args['ld_library'] = os.environ['LD_LIBRARY_PATH']
    args['folder_id'] = args['folder'][-4:]
  elif "fair" in args['hostname']:
    script = setup_fair
    args['home'] = os.environ['home']
    args['folder_id'] = args['folder'][-4:]
    args['n_gpus'] = len(args['gpu'].split(','))

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
  parser.add_argument("--attacks", type=str, default=attacks,
                        help="List of attacks to perform.")
  parser.add_argument("--gpu", type=str, default="0",
                        help="Set CUDA_VISIBLE_DEVICES.")
  parser.add_argument("--params", type=str, default='',
                        help="Parameters to override.")
  parser.add_argument("--partition", type=str, default="learnfair",
                       help="define the patition to use. FAIR only.")
  args = vars(parser.parse_args())

  # get hostname to setup job parameters
  args['hostname'] = socket.gethostname()
  main(args)
