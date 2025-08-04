#!/bin/bash

# Parameters
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=1
#SBATCH --error=/bigwork/nhwpmoha/Mighty/Final_Evals/Mighty/hpo/SAC_SMAC/Walker2d-v4/seed_0/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --job-name=run_mighty
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/bigwork/nhwpmoha/Mighty/Final_Evals/Mighty/hpo/SAC_SMAC/Walker2d-v4/seed_0/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=ai,tnt
#SBATCH --signal=USR2@120
#SBATCH --time=2880
#SBATCH --wckey=submitit

# setup
source .venv/bin/activate
export JAX_PLATFORM_NAME=cpu

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /bigwork/nhwpmoha/Mighty/Final_Evals/Mighty/hpo/SAC_SMAC/Walker2d-v4/seed_0/.submitit/%A_%a/%A_%a_%t_log.out --error /bigwork/nhwpmoha/Mighty/Final_Evals/Mighty/hpo/SAC_SMAC/Walker2d-v4/seed_0/.submitit/%A_%a/%A_%a_%t_log.err /bigwork/nhwpmoha/Mighty/Final_Evals/Mighty/.venv/bin/python -u -m submitit.core._submit /bigwork/nhwpmoha/Mighty/Final_Evals/Mighty/hpo/SAC_SMAC/Walker2d-v4/seed_0/.submitit/%j
