#!/usr/bin/env bash

 # job name
#SBATCH -J eqdisc
 # output file (%j expands to jobID)
#SBATCH -o out/%A.out
 # total nodes
#SBATCH -N 1
 # total cores
#SBATCH -n 1
#SBATCH --requeue
 # total limit (hh:mm:ss)
#SBATCH -t 24:00:00
#SBATCH --mem=10G

python -u linear_learner.py --slurm_id $SLURM_JOB_ID "$@"
