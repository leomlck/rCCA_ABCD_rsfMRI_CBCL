import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Gather scores')
parser.add_argument('--launch', type=int, default=1,
                    help='')
args = parser.parse_args()

save_path = '.../save/cca_abcd_rsfmri_cbcl/launch_{:0>3d}'.format(args.launch)

# Collect performance metrics from result files across hyperparameter folders, and compile them into a sorted CSV summary
folders, train_scores, test_scores = [], [], []
data = []
for f in os.listdir(save_path):
    file = os.path.join(save_path, f, 'results_train_test_launch_{:0>3d}.csv'.format(args.launch))
    if os.path.exists(file):
        line = f.split('-')
        n_params = len(line)
        folders.append(f)
        df = pd.read_csv(file, index_col=0)
        line.append(df.loc['mean','best_mean_cv_cor'])
        line.append(df.loc['mean','train_cor'])
        line.append(df.loc['mean','test_cor'])
        line.reverse()
        data.append(line)

'''results = pd.DataFrame({'folder':folders,
                        'train_cor':train_scores,
                        'test_cor': test_scores}).sort_values(by='test_cor', ascending=False)'''

cols = ['test_cor', 'train_cor', 'best_mean_cv_cor'] + ['param{:0>2d}'.format(i) for i in range(n_params)] 
results = pd.DataFrame(data, columns=cols).sort_values(by='test_cor', ascending=False)
results.to_csv(os.path.join(save_path, 'results.csv'))
