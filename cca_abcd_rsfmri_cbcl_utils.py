import os
import numpy as np
import pandas as pd
import random

# Load and filter CBCL tabular data for selected events and variables of interest
def load_tab_data(args, tabular_path, id_col, event_col, events, variables_of_interest):
    tab_data = pd.read_csv(os.path.join(tabular_path, 'mh_p_cbcl.csv'))
    tab_data = tab_data[[id_col, event_col] + variables_of_interest]
    tab_data.dropna(subset=variables_of_interest, how='all', inplace=True)
    tab_data = tab_data.loc[tab_data['eventname'].isin(events)]
    return tab_data

# Load and filter rs-fMRI connectivity data for selected events, applying quality control inclusion criteria
def load_rsfmr_data(args, rsfmri_path, id_col, event_col, events):
    rsfmri_file = 'mri_y_rsfmr_' + args.rsfmr_file + '.csv'
    rsfmri_data = pd.read_csv(os.path.join(rsfmri_path, rsfmri_file))
    rsfmri_data = rsfmri_data.loc[rsfmri_data['eventname'].isin(events)]

    rsfmri_incl = pd.read_csv(os.path.join(rsfmri_path, 'mri_y_qc_incl.csv'))
    rsfmri_incl = rsfmri_incl.loc[rsfmri_incl['eventname'].isin(events)]
    rsfmri_data = rsfmri_data.merge(rsfmri_incl[[id_col, event_col, 'imgincl_rsfmri_include']], how='left', on=[id_col, event_col])
    rsfmri_data = rsfmri_data.loc[(rsfmri_data['imgincl_rsfmri_include']==1) | (rsfmri_data[event_col]=='4_year_follow_up_y_arm_1')].drop(columns='imgincl_rsfmri_include')
    return rsfmri_data

# Load and merge demographic and site data, filtering by event and formatting site IDs
def load_demo_data(args, demo_path, id_col, event_col, events):
    demo_data = pd.read_csv(os.path.join(demo_path, 'abcd_p_demo.csv'))
    demo_data = demo_data.loc[demo_data['eventname'].isin(['baseline_year_1_arm_1'])]
    demo_data.drop(columns='eventname', inplace=True)
    demo_vars_of_interest = ['demo_brthdat_v2', 'demo_sex_v2', 'race_ethnicity', 'demo_prnt_ed_v2']
    demo_data = demo_data[[id_col] + demo_vars_of_interest]
    demo_data.rename(columns={'race_ethnicity': 'demo_race_ethnicity'}, inplace=True)
    site_data = pd.read_csv(os.path.join(demo_path, 'abcd_y_lt.csv'))
    site_data = site_data.loc[site_data['eventname'].isin(events)]
    demo_data = site_data[[id_col, event_col, 'site_id_l']].merge(demo_data, how='left', on=[id_col])
    demo_data['demo_site_id_l'] = demo_data['site_id_l'].apply(lambda x: int(x[4:]))
    return demo_data

# Load sibling information and keep one unique subject per family to avoid relatedness bias
def load_siblings_data(args, demo_path, id_col):
    siblings_data = pd.read_csv(os.path.join(demo_path, 'abcd_y_lt.csv'), usecols=[id_col, 'rel_family_id'])
    siblings_data.dropna(inplace=True) 
    siblings_data.drop_duplicates(subset='rel_family_id', inplace=True)
    return siblings_data

# Load and compute Adverse Childhood Experience (ACE) scores by merging multiple ABCD subscales related to trauma, family environment, parental monitoring, and family history
def load_ace_data(args, data_path):
    ksads = pd.read_csv(os.path.join(data_path, 'mental-health/mh_p_ksads_ptsd.csv'))

    ksads = ksads[['src_subject_id','eventname','ksads_ptsd_raw_760_p','ksads_ptsd_raw_761_p',
                'ksads_ptsd_raw_766_p','ksads_ptsd_raw_767_p','ksads_ptsd_raw_768_p']]
    df_ace = ksads
    ce = pd.read_csv(os.path.join(data_path, 'culture-environment/ce_p_fes.csv'))
    ce = ce[['src_subject_id','eventname','fam_enviro6_p','fam_enviro3_p']]

    df_ace = pd.merge(df_ace,ce,on=['src_subject_id','eventname'])
    ce = pd.read_csv(os.path.join(data_path, 'culture-environment/ce_y_crpbi.csv'))
    ce = ce[['src_subject_id','eventname','crpbi_caregiver15_y']]

    df_ace = pd.merge(df_ace,ce,on=['src_subject_id','eventname'])
    ce = pd.read_csv(os.path.join(data_path, 'culture-environment/ce_y_pm.csv'))
    ce = ce[['src_subject_id','eventname','parent_monitor_q1_y','parent_monitor_q2_y','parent_monitor_q3_y']]

    df_ace = pd.merge(df_ace,ce,on=['src_subject_id','eventname'])
    su = pd.read_csv(os.path.join(data_path, 'mental-health/mh_p_fhx.csv'))

    su = su[['src_subject_id','eventname','famhx_4_p','fam_history_5_yes_no','fam_history_6_yes_no','fam_history_9_yes_no','fam_history_13_yes_no']]
    df_ace = pd.merge(df_ace,su,on=['src_subject_id','eventname'])
    demo = pd.read_csv(os.path.join(data_path, 'abcd-general/abcd_p_demo.csv'))

    demo = demo[['src_subject_id','eventname','demo_prnt_marital_v2']]
    df_ace = pd.merge(df_ace,demo,on=['src_subject_id','eventname'])
    ace = (((df_ace['ksads_ptsd_raw_760_p'] + df_ace['ksads_ptsd_raw_761_p'])>0).astype(int) +
        ((df_ace['ksads_ptsd_raw_767_p'] + df_ace['ksads_ptsd_raw_768_p'])>0).astype(int) +
        ((df_ace['ksads_ptsd_raw_766_p'] + df_ace['fam_enviro6_p'] + df_ace['fam_enviro3_p']) > 0 ).astype(int) +
        ((df_ace['famhx_4_p'] + df_ace['fam_history_5_yes_no'])>0).astype(int) +
        ((df_ace['fam_history_6_yes_no'] + df_ace['fam_history_13_yes_no'])>0).astype(int)+
        ((df_ace['demo_prnt_marital_v2']) == 3 | (df_ace['demo_prnt_marital_v2'] == 4)).astype(int)+
        (df_ace['fam_history_9_yes_no']>0).astype(int)+
        (df_ace['crpbi_caregiver15_y']<3).astype(int)+
        ((df_ace['parent_monitor_q1_y'] <3) | (df_ace['parent_monitor_q2_y'] <3) | (df_ace['parent_monitor_q3_y'] <3)).astype(int))
    sum(ace>0)

    df_ace['ace'] = ace
    df_ace['ace'][df_ace['ace'] > 9] = np.nan
    df_ace.dropna(subset=['ace'])

    return df_ace

# Generate balanced site-based cross-validation splits ensuring all sites are included across splits
def get_balanced_site_splits(sites, n_splits=10, sites_per_split=3, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_sites = set(sites)
    test_sites = []
    seen_sites = set()

    attempt = 0
    max_attempts = 10000  # prevent infinite loop

    while attempt < max_attempts:
        test_sites = []
        seen_sites = set()

        for _ in range(n_splits):
            split = tuple(sorted(random.sample(sites, sites_per_split)))
            test_sites.append(split)
            seen_sites.update(split)

        if seen_sites == all_sites:
            return test_sites

        attempt += 1

    raise RuntimeError("Failed to generate valid site splits after many attempts.")


