#!/bin/bash
#
#SBATCH	       --requeue 
#SBATCH        --account=minilab 
#SBATCH        --partition=minilab-cpu 
#SBATCH	       --cpus-per-task=16
#SBATCH	       --nodes=1 
#SBATCH        --time=48:00:00 
#SBATCH        --mem=20G 
#SBATCH        --mail-type=ALL 
#SBATCH        --mail-user=lem4012@med.cornell.edu 
#SBATCH        --job-name=cca
#SBATCH        -e ./job_err_cca/%j-job_err_cca.err 
#SBATCH        -o ./job_out_cca/%j-job_out_cca.out 
 
module purge
module load anaconda3
source activate cca

ulimit -n 25000

python launch_cca_abcd_rsfmri_cbcl.py 

