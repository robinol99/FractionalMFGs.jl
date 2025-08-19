#!/bin/bash
#SBATCH --job-name=mfg_conv
#SBATCH --output=logs/mfg_conv_%j.out
#SBATCH --error=logs/mfg_conv_%j.err
#SBATCH --partition=seas_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00




# Move to project root
cd /n/netscratch/kozinsky_lab/Lab/uunneberg/repos/FractionalMFGs.jl

mkdir -p logs

# Use all allocated CPUs for Julia threads
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Optional: load Julia module if needed on your cluster
# module load julia/1.10

# Run the convergence rate script within this project's environment
srun julia --project=. -O3 --check-bounds=no notebooks/MFG_convergence_rate.jl


