import os
import numpy as np
import pandas as pd
import random
import json
from dictparse import DictionaryParser
import argparse

from sklearn.linear_model import LinearRegression

from cca_abcd_rsfmri_cbcl_utils import load_tab_data, load_rsfmr_data, load_demo_data, load_siblings_data, load_ace_data
from permutation_analysis_utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Permutation analysis')
parser.add_argument('--launch', type=int, default=1,
                    help='')
parser.add_argument('--con', type=str, choices=['gp_gp', 'gp_aseg'], default='gp_gp',
                    help='')
parser.add_argument('--type', type=str, choices=['main', 'sex', 'ace'], default='main',
                    help='')
parser.add_argument('--n_perm', type=int, default=10,
                    help='')
parser.add_argument('--n_comp', type=int, default=2,
                    help='')
perm_args = parser.parse_args()

save_path = '.../save/cca_abcd_rsfmri_cbcl'
save_path = os.path.join(save_path, 'launch_{:0>3d}'.format(perm_args.launch))
ext = ''

# Retrieve the launch folder and select the best run for a specific permutation analysis (main, sex, or ACE model) using the corresponding hyperparameters
results_summary = pd.read_csv(os.path.join(save_path, 'results.csv'), index_col=0)
results_summary = results_summary.sort_values(by='test_cor', ascending=False)

settings = results_summary.columns[results_summary.columns.str.startswith('param')].to_list()
settings.reverse()

best_runs = results_summary.loc[(results_summary['param14']=='rsfmr_file_cor_'+perm_args.con)]
best_runs_idx = best_runs.index
run = '-'.join(results_summary.loc[best_runs_idx[0], settings].to_list())
folder_path = os.path.join(save_path, run)

args_file = os.path.join(folder_path, 'args.txt')
with open(args_file) as f:
    args = json.load(f)

parser = DictionaryParser()
for arg in args.keys():
    parser.add_param(arg)
args = parser.parse_dict(args)

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

# Define data paths
data_path = '.../data/abcd-data-release-5.0/core'
rsfmri_path = os.path.join(data_path, 'imaging')
tabular_path = os.path.join(data_path, 'mental-health')
demo_path = os.path.join(data_path, 'abcd-general') 

id_col = 'src_subject_id'
event_col = 'eventname'
events = np.array(['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1', '4_year_follow_up_y_arm_1'])
events = events[args.events]
variables_of_interest = ["cbcl_scr_syn_anxdep_","cbcl_scr_syn_withdep_","cbcl_scr_syn_somatic_", "cbcl_scr_syn_social_",
                         "cbcl_scr_syn_thought_","cbcl_scr_syn_attention_", "cbcl_scr_syn_rulebreak_","cbcl_scr_syn_aggressive_",
                        ]
variables_of_interest = [var + args.cbcl_score for var in variables_of_interest]

# Load tab (cbcl) data
tab_data = load_tab_data(args, tabular_path, id_col, event_col, events, variables_of_interest)
print('Number of subjects w/ cbcl data: ', len(tab_data))

# Load rsfmr data
rsfmri_data = load_rsfmr_data(args, rsfmri_path, id_col, event_col, events)
print('Number of subjects w/ rsfmri data (quality): ', len(rsfmri_data))

# Load demographic data
demo_data = load_demo_data(args, demo_path, id_col, event_col, events)
print('Number of subjects w/ demo data: ', len(demo_data))

# Load siblings data
siblings_data = load_siblings_data(args, demo_path, id_col)
print('Number of subjects w/o siblings duplicates: ', len(siblings_data))

# Merge data
data = siblings_data[[id_col]].merge(rsfmri_data, how='left', on=[id_col]) 
data = data.merge(tab_data, how='left', on=[id_col, event_col])
data = data.merge(demo_data, how='left', on=[id_col, event_col])
print('Number of subjects w/ merged data: ', len(data))
data.dropna(inplace=True)
print('Number of subjects w/ merged data (drop NA): ', len(data))

# Filter dataset to include only subjects that passed motion QC and merge with full data
if args.qc_data:
    qc_subjects = pd.read_csv(os.path.join(data_path, 'subjects_motion_QC.txt'), header=None, names=['src_subject_id'])
    qc_subjects['src_subject_id'] = qc_subjects['src_subject_id'].map(lambda x : x[:4]+'_'+x[4:])
    data = qc_subjects[[id_col]].merge(data, how='left', on=[id_col])
    data.dropna(inplace=True)
    print('Number of subjects w/ merged data (drop QC control): ', len(data))

rsfmri_cols = data.columns[data.columns.str.startswith('rsfmri_')].to_list()
cbcl_cols = data.columns[data.columns.str.startswith('cbcl_')].to_list()
cbcl_cols = ["cbcl_scr_syn_anxdep_",
             "cbcl_scr_syn_thought_",
             "cbcl_scr_syn_attention_", 
             "cbcl_scr_syn_social_",
             "cbcl_scr_syn_rulebreak_",
             "cbcl_scr_syn_aggressive_",
             "cbcl_scr_syn_withdep_",
             "cbcl_scr_syn_somatic_",
             ]
cbcl_cols = [var + args.cbcl_score for var in cbcl_cols]
demo_cols = data.columns[data.columns.str.startswith('demo_')].to_list()

results = pd.read_csv(os.path.join(folder_path, 'results_train_test_launch_{:0>3d}.csv'.format(args.launch)), index_col=0)
str_sites = [results.iloc[t]['sites'] for t in range(args.n_test)]
site_lists = [[int(site) for site in str_site.split('_')] for str_site in str_sites] 

# Define test site splits for cross-validation based on the number of test sets specified
if args.n_test > 0:
    training_data = [data.loc[~data['demo_site_id_l'].isin(sites)] for sites in site_lists]
    testing_data = [data.loc[data['demo_site_id_l'].isin(sites)] for sites in site_lists]
elif args.n_test == 0:
    training_data = [data]
    testing_data = [data]

# Run permutation analysis based on the selected type: main, sex-specific, or ACE-specific models, with appropriate data and function calls
if perm_args.type == 'main':
    ext = '_' + perm_args.type
    cors, loadings_cbcl, loadings_con = perm_components(perm_args.n_perm, perm_args.n_comp, str_sites, folder_path, training_data, testing_data, cbcl_cols, rsfmri_cols, demo_cols)
elif perm_args.type == 'sex':
    ext = '_' + perm_args.type
    cors, loadings_cbcl, loadings_con = perm_components_sex(perm_args.n_perm, perm_args.n_comp, str_sites, folder_path, training_data, testing_data, cbcl_cols, rsfmri_cols, demo_cols)
elif perm_args.type == 'ace'  or perm_args.type == 's_ace':
    ext = '_' + perm_args.type

    df_ace = load_ace_data(arg, data_path)
    ace_train_data, ace_test_data = [], []
    for i_s in range(args.n_test):
        ace_train_data.append(training_data[i_s].merge(df_ace[['ace', 'src_subject_id']], how='left', on='src_subject_id'))
        ace_test_data.append(testing_data[i_s].merge(df_ace[['ace', 'src_subject_id']], how='left', on='src_subject_id')) 

    cors, loadings_cbcl, loadings_con = perm_components_ace(perm_args.n_perm, perm_args.n_comp, str_sites, folder_path, ace_train_data, ace_test_data, cbcl_cols, rsfmri_cols, demo_cols) 

# Save permutation results to numpy file
np.savez(os.path.join(folder_path, 'permutation_analysis'+ext+'_n_perm_{}_n_comp_{}.npz'.format(perm_args.n_perm, perm_args.n_comp)),
         cors=cors,
         loadings_cbcl=loadings_cbcl,
         loadings_con=loadings_con)


