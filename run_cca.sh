#!/bin/bash
#
#SBATCH	       --requeue 
#SBATCH        --account=... 
#SBATCH        --partition=...
#SBATCH	       --cpus-per-task=16
#SBATCH	       --nodes=1 
#SBATCH        --time=48:00:00 
#SBATCH        --mem=20G 
#SBATCH        --job-name=cca
#SBATCH        -e ./job_err_cca/%j-job_err_cca.err 
#SBATCH        -o ./job_out_cca/%j-job_out_cca.out 
 
module purge
module load anaconda3
source activate cca

python launch_cca_abcd_rsfmri_cbcl.py 

