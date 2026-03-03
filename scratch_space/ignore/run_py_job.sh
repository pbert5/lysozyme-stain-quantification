#!/bin/bash
#SBATCH --job-name=py_analysis        # a short name for your job
#SBATCH --output=logs/%x_%j.out       # %x→job-name, %j→job-id
#SBATCH --error=logs/%x_%j.err        # (you can combine with --output)
#SBATCH --nodes=1                     # how many nodes?
#SBATCH --ntasks=1                    # total MPI tasks (we’re single-process)
#SBATCH --cpus-per-task=4             # threads for Python (e.g. numba, joblib)
#SBATCH --mem=16                     # RAM per node (or per CPU in new Slurm)
#SBATCH --time=00:02:00               # hh:mm:ss walltime limit
#SBATCH --partition=12c             # or whatever partition you use



# 1) go to where you launched sbatch
cd $SLURM_SUBMIT_DIR # return to the directory where you ran sbatch

source /home/user/documents/PiereLab/lysozyme/.venv/bin/activate

# 3) (optional) verify it


                

python3 /home/user/documents/PiereLab/lysozyme/lysozyme-stain-quantification/src/run.py
