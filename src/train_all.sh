#!/bin/bash
export STEPS=5000

export START=0
export END=9
NETWORKS=(deep frozen frozenv2 contrast frozen_contrast)
DBS=(cnn cnn_nopitts)

for i in $(seq $START $END); do
  export net_var=${NETWORKS[$(expr $i / 2)]}
  export db_var=${DBS[$(expr $i % 2)]}
  echo Training $net_var on $db_var
  export log_dir=sessions/$net_var/logs_$db_var
  pkill -9 tensorboard
  tensorboard --logdir=$log_dir 2> /dev/null &
  python src/train.py $i $STEPS
done

for i in $(seq 0 9); do
  python src/precisionrecall_rtabmap.py $i 0
  python src/precisionrecall_rtabmap.py $i 1
done

