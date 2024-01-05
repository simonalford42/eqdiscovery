#!/usr/bin/env bash

sbatch -J acc1e-4 --partition ellis train.sh --lr 1e-4
sbatch -J acc1e-3 --partition ellis train.sh --lr 1e-3
sbatch -J acc1e-5 --partition ellis train.sh --lr 1e-5
