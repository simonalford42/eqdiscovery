#!/usr/bin/env bash

# sbatch -J hp313 --partition ellis train.sh --lr 1e-3
# sbatch -J hp353 --partition ellis train.sh --lr 5e-3
# sbatch -J hp314 --partition gpu train.sh --lr 1e-4
# sbatch -J hp354 --partition gpu train.sh --lr 5e-4

sbatch -J lr5e-0 --partition gpu train.sh --lr 5e-0 --note lr5e-0
sbatch -J lr1e-0 --partition gpu train.sh --lr 1e-0 --note lr1e-0
sbatch -J lr5e-1 --partition gpu train.sh --lr 5e-1 --note lr5e-1
sbatch -J lr1e-1 --partition gpu train.sh --lr 1e-1 --note lr1e-1
sbatch -J lr5e-2 --partition gpu train.sh --lr 5e-2 --note lr5e-2
sbatch -J lr1e-2 --partition gpu train.sh --lr 1e-2 --note lr1e-2

