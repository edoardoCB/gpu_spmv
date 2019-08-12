#!/bin/sh
#SBATCH --job-name=GPU_MDE_BS0032_KAY
#SBATCH --partition=GpuQ
#SBATCH --account=hpce3ic5
#SBATCH --nodes=1
#SBATCH --time=0:02:00
module load intel/2017u8
module load cuda/9.2
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=1
./gpuSpmv MDE.csr 0032
