#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

for trial in 1 2 3 4 5
do
	python train_ext.py \
	--dataset regdb  \
	--method adp \
	--augc 1 \
	--rande 0.5 \
	--alpha 1 \
	--square 1 \
	--gamma 1 \
	--trial $trial
done
echo 'Done!'