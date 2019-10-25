# TensorFlow Research Library

## Setup library

```
export PROJECTDIR=`pwd`/neuralnet

# define the data and models folders
mkdir data models
export DATADIR=`pwd`/data
export WORKDIR=`pwd`

# download and processed datasets 
cd neuralnet
pip install -r requirements.txt
python3 code/dataset/generate_tfrecords.py --output_dir=$DATADIR
```


## Run training and eval

The config.yaml file should be in the config folder
```
./sub/train.py config | bash
```

To make an eval under attack after training
```
./sub/train.py config --attacks all | bash 
./sub/train.py config --attacks 'fgm pgd carlini' | bash
```

To config the GPUs
```
./sub/train.py config --gpu_train '0,1' --gpu_eval '2,3' --gpu_attacks 0 --attacks all | bash
```

Inside the PROJECTDIR the default folder is 'models' to change if
```
./sub/train.py config --path /new/path/for/models
```

## Train multiple models

To launch a series of experiment with different parameters, it possible to use the config file as a template and to populate it with values. 

Example of config file as template:
```
    ...
    distributions:            {distributions}
    scale_noise:              {scale_noise}
    ...
...
```

You can populate the values with the "params" parameter:
```
./sub/train.py config_template --params '{"distributions": "l1", "scale_noise": 0.01}' --name xp
./sub/train.py config_template --params '{"distributions": "l2", "scale_noise": 0.02}' --name xp
./sub/train.py config_template --params '{"distributions": "exp", "scale_noise": 0.03}' --name xp
```

New config file will be generated and saved in the 'config_gen' folder. Each config file will have a different id. 

## Run evaluation or attacks

Run an evaluation with the eval script followed by the name of the model folder the model folder should be in the WORKDIR/models otherwise you set the path with '--path'. The attacks script run all attacks by default. 
```
./sub/eval.py 2019-04-11_09.05.49_1541 | bash
./sub/attacks.py 2019-04-11_09.05.49_1541 | bash

./sub/eval.py 2019-04-11_09.05.49_1541 --path /new/path | bash
./sub/attacks.py 2019-04-11_09.05.49_1541 --path /new/path | bash
```

To config the GPUs
```
./sub/eval.py 2019-04-11_09.05.49_1541 --gpu '0,1' | bash
```

## Override parameters to run multiple evaluation

It is possible to run multiple evaluation with different parameters with the '--params' command. The '--params' takes a json string of the same structure as the config file. The parameters are overrided from the load model_flags.yaml inside the model folder.   

Example of config file:
```
...

attack_fgm:
  <<: *DEFAULT
  <<: *EVAL_TEST
  <<: *ATTACK
  attack_method:              FastGradientMethod
  FastGradientMethod:
    eps:                      0.3
    ord:                      inf
    clip_min:                 -1.0
    clip_max:                 +1.0

...

```

```
./sub/eval.py 2019-04-11_09.05.49_1541 --params '{"FastGradientMethod": {"eps": 0.1}}' 
./sub/eval.py 2019-04-11_09.05.49_1541 --params '{"FastGradientMethod": {"eps": 0.2}}' 
./sub/eval.py 2019-04-11_09.05.49_1541 --params '{"FastGradientMethod": {"eps": 0.2}}' 
```

The results will be inside the logs files and in the history file.



