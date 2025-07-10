import os
import numpy as np
import pandas as pd
import random
import json

from cca_zoo.linear import rCCA, CCA
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Perform permutation testing of CCA model across multiple sites and permutations, storing correlation and loading distributions for CBCL and connectivity features
def perm_components(n_permutations, n_components, str_sites, folder_path, training_data, testing_data, cbcl_cols, rsfmri_cols):
    n_cbcl = len(cbcl_cols)
    n_con = len(rsfmri_cols)
    cors = np.zeros((2, len(str_sites), n_components, n_permutations))
    loadings_cbcl = np.zeros((2, len(str_sites), n_components, n_cbcl, n_permutations))
    loadings_con = np.zeros((2, len(str_sites), n_components, n_con, n_permutations))

    for i_s, str_site in enumerate(str_sites):
        print('Test sites :', str_site)
        best_params_file = os.path.join(folder_path, 'best_params_test_sites_{}.txt'.format(str_site))
        with open(best_params_file) as f:
            best_params = json.load(f)

        best_cca = rCCA(latent_dimensions=n_components, **best_params)
        for i_p in range(n_permutations):
            train_cbcl_p = training_data[i_s][cbcl_cols].sample(frac=1)
            test_cbcl_p = testing_data[i_s][cbcl_cols].sample(frac=1)
            X1_c_train, X2_c_train = best_cca.fit_transform((training_data[i_s][rsfmri_cols], train_cbcl_p))
            X1_c_test, X2_c_test = best_cca.transform((testing_data[i_s][rsfmri_cols], test_cbcl_p))

            for i in range(n_components):
                train_cor = pearsonr(np.squeeze(X1_c_train[:,i]), np.squeeze(X2_c_train[:,i]))
                test_cor = pearsonr(np.squeeze(X1_c_test[:,i]), np.squeeze(X2_c_test[:,i]))
                cors[0,i_s,i,i_p] = train_cor[0]
                cors[1,i_s,i,i_p] = test_cor[0]

                for i_cbcl in range(n_cbcl):
                    loadings_cbcl[0,i_s,i,i_cbcl,i_p] = pearsonr(train_cbcl_p.to_numpy()[:,i_cbcl], X2_c_train[:,i])[0]
                    loadings_cbcl[1,i_s,i,i_cbcl,i_p] = pearsonr(test_cbcl_p.to_numpy()[:,i_cbcl], X2_c_test[:,i])[0]  

                for i_con in range(n_con):                  
                    loadings_con[0,i_s,i,i_con,i_p] = pearsonr(training_data[i_s][rsfmri_cols].to_numpy()[:,i_con], X1_c_train[:,i])[0]
                    loadings_con[1,i_s,i,i_con,i_p] = pearsonr(testing_data[i_s][rsfmri_cols].to_numpy()[:,i_con], X1_c_test[:,i])[0]

    return cors, loadings_cbcl, loadings_con

# Perform permutation testing of CCA separately for male and female subgroups to evaluate sex-specific brain-behavior associations
def perm_components_sex(n_permutations, n_components, str_sites, folder_path, training_data, testing_data, cbcl_cols, rsfmri_cols):
    sexes = {'male':1, 'female':2}
    n_cbcl = len(cbcl_cols)
    n_con = len(rsfmri_cols)
    cors = np.zeros((2, len(str_sites), n_components, n_permutations))
    loadings_cbcl = np.zeros((2, len(str_sites), n_components, n_cbcl, n_permutations))
    loadings_con = np.zeros((2, len(str_sites), n_components, n_con, n_permutations))

    for i_s, str_site in enumerate(str_sites):
        print('Test sites :', str_site)
        best_params_file = os.path.join(folder_path, 'best_params_test_sites_{}.txt'.format(str_site))
        with open(best_params_file) as f:
            best_params = json.load(f)

        best_cca = rCCA(latent_dimensions=n_components, **best_params)
        for i_p in range(n_permutations):
            male_data_train, female_data_train = train_test_split(training_data[i_s], test_size=0.5)
            male_data_test, female_data_test = train_test_split(testing_data[i_s], test_size=0.5)
            best_cca.fit((male_data_train[rsfmri_cols], male_data_train[cbcl_cols]))
            X1_c_male, X2_c_male = best_cca.transform((male_data_test[rsfmri_cols], male_data_test[cbcl_cols]))
            best_cca.fit((female_data_train[rsfmri_cols], female_data_train[cbcl_cols]))
            X1_c_female, X2_c_female = best_cca.transform((female_data_test[rsfmri_cols], female_data_test[cbcl_cols]))

            for i in range(n_components):
                male_cor = pearsonr(np.squeeze(X1_c_male[:,i]), np.squeeze(X2_c_male[:,i]))
                female_cor = pearsonr(np.squeeze(X1_c_female[:,i]), np.squeeze(X2_c_female[:,i]))
                cors[0,i_s,i,i_p] = male_cor[0]
                cors[1,i_s,i,i_p] = female_cor[0]

                for i_cbcl in range(n_cbcl):
                    loadings_cbcl[0,i_s,i,i_cbcl,i_p] = pearsonr(male_data_test[cbcl_cols].to_numpy()[:,i_cbcl], X2_c_male[:,i])[0]
                    loadings_cbcl[1,i_s,i,i_cbcl,i_p] = pearsonr(female_data_test[cbcl_cols].to_numpy()[:,i_cbcl], X2_c_female[:,i])[0]

                for i_con in range(n_con):
                    loadings_con[0,i_s,i,i_con,i_p] = pearsonr(male_data_test[rsfmri_cols].to_numpy()[:,i_con], X1_c_male[:,i])[0]
                    loadings_con[1,i_s,i,i_con,i_p] = pearsonr(female_data_test[rsfmri_cols].to_numpy()[:,i_con], X1_c_female[:,i])[0]

    return cors, loadings_cbcl, loadings_con

# Perform permutation testing of CCA separately for low and high ACE subgroups to compare brain-behavior mappings across adverse childhood experience levels
def perm_components_ace(n_permutations, n_components, str_sites, folder_path, training_data, testing_data, cbcl_cols, rsfmri_cols):
    n_cbcl = len(cbcl_cols)
    n_con = len(rsfmri_cols)
    cors = np.zeros((2, len(str_sites), n_components, n_permutations))
    loadings_cbcl = np.zeros((2, len(str_sites), n_components, n_cbcl, n_permutations))
    loadings_con = np.zeros((2, len(str_sites), n_components, n_con, n_permutations))

    for i_s, str_site in enumerate(str_sites):
        print('Test sites :', str_site)
        best_params_file = os.path.join(folder_path, 'best_params_test_sites_{}.txt'.format(str_site))
        with open(best_params_file) as f:
            best_params = json.load(f)

        best_cca = rCCA(latent_dimensions=n_components, **best_params)
        for i_p in range(n_permutations):
            l_ace_data_train, h_ace_data_train = train_test_split(training_data[i_s], test_size=0.5)
            l_ace_data_test, h_ace_data_test = train_test_split(testing_data[i_s], test_size=0.5)
            best_cca.fit((l_ace_data_train[rsfmri_cols], l_ace_data_train[cbcl_cols]))
            X1_c_l_ace, X2_c_l_ace = best_cca.transform((l_ace_data_test[rsfmri_cols], l_ace_data_test[cbcl_cols]))
            best_cca.fit((h_ace_data_train[rsfmri_cols], h_ace_data_train[cbcl_cols]))
            X1_c_h_ace, X2_c_h_ace = best_cca.transform((h_ace_data_test[rsfmri_cols], h_ace_data_test[cbcl_cols]))

            for i in range(n_components):
                l_ace_cor = pearsonr(np.squeeze(X1_c_l_ace[:,i]), np.squeeze(X2_c_l_ace[:,i]))
                h_ace_cor = pearsonr(np.squeeze(X1_c_h_ace[:,i]), np.squeeze(X2_c_h_ace[:,i]))
                cors[0,i_s,i,i_p] = l_ace_cor[0]
                cors[1,i_s,i,i_p] = h_ace_cor[0]

                for i_cbcl in range(n_cbcl):
                    loadings_cbcl[0,i_s,i,i_cbcl,i_p] = pearsonr(l_ace_data_test[cbcl_cols].to_numpy()[:,i_cbcl], X2_c_l_ace[:,i])[0]
                    loadings_cbcl[1,i_s,i,i_cbcl,i_p] = pearsonr(h_ace_data_test[cbcl_cols].to_numpy()[:,i_cbcl], X2_c_h_ace[:,i])[0]  

                for i_con in range(n_con):                  
                    loadings_con[0,i_s,i,i_con,i_p] = pearsonr(l_ace_data_test[rsfmri_cols].to_numpy()[:,i_con], X1_c_l_ace[:,i])[0]
                    loadings_con[1,i_s,i,i_con,i_p] = pearsonr(h_ace_data_test[rsfmri_cols].to_numpy()[:,i_con], X1_c_h_ace[:,i])[0]

    return cors, loadings_cbcl, loadings_con



