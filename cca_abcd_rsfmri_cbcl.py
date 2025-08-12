import os
import numpy as np
import pandas as pd
import random
import json
import argparse
import sys

from cca_zoo.linear import rCCA
from cca_zoo.model_selection import GridSearchCV

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from cca_abcd_rsfmri_cbcl_utils import load_tab_data, load_rsfmr_data, load_demo_data, load_siblings_data, get_balanced_site_splits

import warnings
warnings.filterwarnings("ignore")

# Define command-line arguments to configure CCA analysis and preprocessing options
parser = argparse.ArgumentParser(description='CCA Analysis rs-fMRI connectivity features vs CBCL scores')
parser.add_argument('--cbcl_score', choices=['r', 't'], type=str, default='r',
                    help='use raw or t scores')
parser.add_argument('--rsfmr_file', type=str, default='cor_gp_gp',
                    help='connectivity features')
parser.add_argument('--events', type=int, nargs="+",
                    help='follow-up exams to consider')
parser.add_argument('--qc_data', type=int, default=0,
                    help='use qc data')
parser.add_argument('--n_test', type=int, default=1,
                    help='nb of train/test splits')
parser.add_argument('--n_site_per_test', type=int, default=3,
                    help='nb of sites in the train set')
parser.add_argument('--use_site_lvl_cv', type=int, default=0,
                    help='perform site level CV')
parser.add_argument('--use_residuals_rsfmr', type=int, default=0,
                    help='residualize connectivity features')
parser.add_argument('--residuals_var_rsfmr', type=str, default='all',
                    help='residualize with all demographic variables or site only')
parser.add_argument('--use_residuals_cbcl', type=int, default=0,
                    help='residualize cbcl scores with all demographic variables (but site)')
parser.add_argument('--use_std_scaler', type=int, default=0,
                    help='standard scaling of the data')
parser.add_argument('--n_cv_splits', type=int, default=10,
                    help='nb of CV folds')
parser.add_argument('--num_p', type=int, default=1,
                    help='nb points btw each regularization parameter on the log scale')
parser.add_argument('--latent_dim', type=int, default=1,
                    help='nb of latent dimensions of CCA (metric=cumulative corr)')
parser.add_argument('--launch', type=int, default=1,
                    help='launch id')
parser.add_argument('--seed', type=int, default=42,
                    help='select random seed')
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

# Create a unique directory for the current launch and save all arguments and output logs
save_path = '.../save/cca_abcd_rsfmri_cbcl'
save_path = os.path.join(save_path, 'launch_{:0>3d}'.format(args.launch))
save_folder = '-'.join(['{}_{}'.format(arg, getattr(args, arg)) for arg in vars(args)])
save_path = os.path.join(save_path, save_folder) 
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, 'args.txt'), 'a') as f:
    json.dump(args.__dict__, f, indent=2)
orig_stdout = sys.stdout
f = open(os.path.join(save_path,'out.txt'), 'a')
sys.stdout = f

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
                         "cbcl_scr_syn_thought_","cbcl_scr_syn_attention_", "cbcl_scr_syn_rulebreak_","cbcl_scr_syn_aggressive_",]
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
demo_cols = data.columns[data.columns.str.startswith('demo_')].to_list()

# Custom scoring function
def scorer(estimator, X):
    dim_corrs = estimator.score(X)
    return dim_corrs.mean()

# Define grid of potential regularization parameters
num_per_int = args.num_p
e1, e2 = -3, 1
c1 = np.logspace(e1, e2, num=(e2-e1)*num_per_int+1)
e1, e2 = -3, 1
c2 = np.logspace(e1, e2, num=(e2-e1)*num_per_int+1)
param_grid = {'c': [c1, c2],
            'pca': [True],
            }
print(param_grid)

# Define test site splits for cross-validation based on the number of test sets specified
if args.n_test > 0:
    # Usage:
    sites = sorted(data['demo_site_id_l'].unique().tolist())
    test_sites = get_balanced_site_splits(sites, n_splits=10, sites_per_split=3, seed=args.seed)
elif args.n_test == 0:
    test_sites = [-1]

# Cross-validation loop
train_scores, test_scores, best_cv_scores, str_sites_list = [], [], [], []
for i_s, sites in enumerate(test_sites):
    print('Train/test split {}/{}'.format(i_s+1, len(test_sites)))
    str_sites = '_'.join(['{:0>2d}'.format(int(site)) for site in sites])
    str_sites_list.append(str_sites)

    # Split train/test according to site
    if args.n_test > 0:
        training_data = data.loc[~data['demo_site_id_l'].isin(sites)]
        testing_data = data.loc[data['demo_site_id_l'].isin(sites)]
    elif args.n_test == 0:
        training_data = data
        testing_data = data
        
    # Scale data
    with_std = True if args.use_std_scaler else False
    train_rsfmri_sc = StandardScaler(with_std=with_std)
    train_cbcl_sc = StandardScaler(with_std=with_std)

    training_data[rsfmri_cols] =  train_rsfmri_sc.fit_transform(training_data[rsfmri_cols])
    training_data[cbcl_cols] =  train_cbcl_sc.fit_transform(training_data[cbcl_cols])

    test_rsfmri_sc = StandardScaler(with_std=with_std)
    test_cbcl_sc = StandardScaler(with_std=with_std)

    testing_data[rsfmri_cols] =  test_rsfmri_sc.fit_transform(testing_data[rsfmri_cols])
    testing_data[cbcl_cols] =  test_cbcl_sc.fit_transform(testing_data[cbcl_cols])

    # Transforn rsfmr data to residuals with confounders (demo vars)
    if args.use_residuals_rsfmr:
        if args.residuals_var_rsfmr == 'all':
            conf_cols = demo_cols
        else:
            conf_cols = [args.residuals_var_rsfmr]

        for var in rsfmri_cols:
            model = LinearRegression().fit(training_data[conf_cols], training_data[[var]])
            training_data['r_' + var] = training_data[[var]] - model.predict(training_data[conf_cols])
            testing_data['r_' + var] = testing_data[[var]] - model.predict(testing_data[conf_cols])
        rsfmri_cols = ['r_' + var for var in rsfmri_cols]

    # Transforn cbcl data to residuals with confounders (demo vars excluding site) 
    if args.use_residuals_cbcl:
        conf_cols_cbcl = demo_cols.copy()
        conf_cols_cbcl.remove('demo_site_id_l')
        for var in cbcl_cols:
            model = LinearRegression().fit(training_data[conf_cols_cbcl], training_data[[var]])
            training_data['r_' + var] = training_data[[var]] - model.predict(training_data[conf_cols_cbcl])
            testing_data['r_' + var] = testing_data[[var]] - model.predict(testing_data[conf_cols_cbcl])
        cbcl_cols = ['r_' + var for var in cbcl_cols]

    # CV splits
    gss = GroupShuffleSplit(n_splits=args.n_cv_splits, test_size=0.2)
    groups = training_data[['demo_site_id_l']] if args.use_site_lvl_cv else training_data[['src_subject_id']] 
    cv = []
    for i_cv, (train_idx, valid_idx) in enumerate(gss.split(training_data, y=None, groups=groups)):
        cv.append((train_idx, valid_idx))

    # Conduct grid search
    grid_cv = GridSearchCV(rCCA(latent_dimensions=args.latent_dim), param_grid=param_grid,
                        cv=cv, verbose=True, 
                        scoring=scorer, 
                        #return_train_score=True
                        ).fit((training_data[rsfmri_cols], training_data[cbcl_cols]))

    to_drop_cols = ['split{}_test_score'.format(i) for i in range(gss.get_n_splits())]  
    cv_results = pd.DataFrame(grid_cv.cv_results_).drop(columns=to_drop_cols).sort_values(by='mean_test_score', ascending=False)
    cv_results.to_csv(os.path.join(save_path, 'cv_results_test_sites_{}.csv'.format(str_sites)))
    best_cv_scores.append(cv_results.loc[cv_results['rank_test_score']==1]['mean_test_score'].to_numpy()[0])

    # Save best parameters
    best_params = grid_cv.best_params_
    with open(os.path.join(save_path, 'best_params_test_sites_{}.txt'.format(str_sites)), 'a') as f:
        json.dump(best_params, f, indent=2)

    # Retrain model on whole train set and evaluate on test set
    best_cca = grid_cv.best_estimator_
    best_cca.fit((training_data[rsfmri_cols], training_data[cbcl_cols]))
    train_cor = best_cca.score((training_data[rsfmri_cols], training_data[cbcl_cols]))
    train_scores.append(train_cor)
    print(' Train correlation :', train_cor)
    test_cor = best_cca.score((testing_data[rsfmri_cols], testing_data[cbcl_cols]))
    test_scores.append(test_cor)
    print(' Test correlation :', test_cor)

# Save results
results = pd.DataFrame(data={'sites': str_sites_list, 
                             'train_cor': train_scores, 
                             'test_cor': test_scores,
                             'best_mean_cv_cor': best_cv_scores})
results = pd.concat([results, results.describe()])
results.to_csv(os.path.join(save_path, 'results_train_test_launch_{:0>3d}.csv'.format(args.launch)))

sys.stdout = orig_stdout
f.close()
