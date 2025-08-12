#!/bin/bash
#
##SBATCH        --begin=now+1hours
#SBATCH	       --requeue 
#SBATCH        --account=... 
#SBATCH        --partition=... 
#SBATCH	       --cpus-per-task=16
#SBATCH	       --nodes=1 
#SBATCH        --time=96:00:00 
#SBATCH        --mem=20G 
#SBATCH        --job-name=cca_perm
#SBATCH        -e ./job_err_cca/%j-job_err_cca.err 
#SBATCH        -o ./job_out_cca/%j-job_out_cca.out 
 
module purge
module load anaconda3
source activate cca

python gather_cca_abcd_rsfmri_cbcl.py --launch 24
python permutation_analysis_both.py --launch 24 --con both --type main --n_perm 500 --n_comp 2

