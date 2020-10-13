#!/usr/bin/env sh


now=$(date +"%Y%m%d_%H:%M:%S")
now_short=$(date +"%H:%M:%S")

config=$1

work_dir=work_dirs/train_lfb-$now

step1_model=$2

mkdir -p $work_dir

python tools/train_net.py \
	--config_file $config \
    LFB.MODEL_PARAMS_FILE $step1_model \
    LFB.WRITE_LFB True \
	CHECKPOINT.DIR $work_dir \
        ${@:3} \
	2>&1|tee $work_dir/train.log &


