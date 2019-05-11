##!/usr/bin/bash

for step in 1 0.1; do
	for seed in 5 6; do
		echo ${seed} ${step}
	   	bash -c 'python3 run.py --horizon 5 --batch_size 1 --max_iters 2 --filename step'$step '--seed '$seed '--step_size '$step 
	done
done

