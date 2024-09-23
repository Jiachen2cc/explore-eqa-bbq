#!/bin/bash
#SBATCH --job-name=eval_object_retrieval
#SBATCH -o output/obj_retrieval_%j.out
#SBATCH -e output/obj_retrieval_%j.err
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="2080ti"

module load miniconda/22.11.1-1
module load gcc/13.2.0
# conda activate /work/pi_chuangg_umass_edu/yuncong/conda_envs/eqa-baseline
conda activate explore-eqa

python main.py -cf cfg/test_sr3d+.yaml