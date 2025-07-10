import os
import subprocess
import json
import numpy as np
from sklearn.model_selection import ParameterGrid
import time

# make launch folder
save_path = '/midtier/sablab/scratch/lem4012/save/cca_abcd_rsfmri_cbcl'
launch = 1 if len([f for f in os.listdir(save_path) if f.split('_')[0]=='launch'])==0 else np.max([int(f[-3:]) for f in os.listdir(save_path) if f.split('_')[0]=='launch'])+1
save_path = os.path.join(save_path, 'launch_{:0>3d}'.format(launch))
os.makedirs(save_path, exist_ok=True)

# parameters grid
parameters = {
    'cbcl_score':['r'],
    'rsfmr_file':['both'], #, 'cor_gp_gp'],
    'events':['0'], #, '1', '0 1'],
    'n_test':[10],
    'qc_data': [0],
    'n_site_per_test':[3],
    'use_site_lvl_cv':[0],
    'use_residuals_rsfmr':[1],
    'residuals_var_rsfmr':['all'], #['demo_site_id_l'], #['all'],
    'use_residuals_cbcl':[1],
    'use_std_scaler':[0],
    'n_cv_splits':[50],
    'num_p':[3],
    'latent_dim':[1], 
    'launch':[int(launch)],
    'seed':[0, 42, 99, 999],
}

# save params grid
with open(os.path.join(save_path, 'param_grid.txt'), 'a') as f:
    json.dump(parameters, f, indent=2)

base_command = 'python cca_abcd_rsfmri_cbcl_reg.py '

for i, params in enumerate(list(ParameterGrid(parameters))):
    print('Sending job params {}/{}'.format(i+1, len(list(ParameterGrid(parameters)))))
    params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
    command = base_command + ' '.join(params_list)
    print('     Command:', command)
    subprocess.Popen(command, shell=True).wait()
