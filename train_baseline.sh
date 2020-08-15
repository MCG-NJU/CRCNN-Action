#!/usr/bin/env sh


now=$(date +"%Y%m%d_%H:%M:%S")
now_short=$(date +"%H:%M:%S")

config=$1

work_dir=work_dirs/train-$now

mkdir -p $work_dir

python tools/train_net.py \
	--config_file $config \
	CHECKPOINT.DIR $work_dir \
        ${@:3} \
	2>&1|tee $work_dir/train.log &


