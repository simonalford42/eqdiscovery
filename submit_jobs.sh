#!/usr/bin/env bash

sbatch -J pos250 --partition ellis train.sh --lr 1e-4 --epochs 250
sbatch -J pos1000 --partition ellis train.sh --lr 1e-4 --epochs 1000

