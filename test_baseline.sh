#!/usr/bin/env sh

#mkdir -p test_log

now=$(date +"%Y%m%d_%H:%M:%S")
now_short=$(date +"%H:%M:%S")

config=$1
checkpoint=$2

work_dir=test_work_dirs/test-$now

mkdir -p $work_dir

python2 tools/test_net.py \
	--config_file $config \
    TEST.PARAMS_FILE $checkpoint \
    CHECKPOINT.DIR $work_dir \
	2>&1|tee $work_dir/test.log &


