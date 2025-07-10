# Regularized CCA Identifies Sex-Specific Brain-Behavior Associations in Adolescent Psychopathology

This project investigates sex-specific associations between resting-state functional connectivity (rs-fMRI) and psychiatric symptoms in early adolescence using the ABCD dataset. Using Canonical Correlation Analysis (CCA) and rigorous cross-validation, we identify multivariate brain-behavior mappings across eight Child Behavior Checklist (CBCL) syndrome scales. In contrast to previous studies, we explicitly model sex differences in these brain-behavior associations, revealing distinct neural correlates for attention, thought, and internalizing problems in males and females. Our findings highlight the importance of considering sex-specific mechanisms in adolescent psychopathology and support the development of sex-informed clinical strategies.

---

## Data

Data used in the preparation of this article were obtained from the Adolescent Brain Cognitive Development℠ (ABCD) Study ([https://abcdstudy.org](https://abcdstudy.org)), publicly released in the NIMH Data Archive (NDA). This is a multisite, longitudinal study designed to recruit more than 10,000 children age 9–10 and follow them over 10 years into early adulthood.

The ABCD Study® is supported by the National Institutes of Health and additional federal partners under award numbers U01DA041048, U01DA050989, U01DA051016, U01DA041022, U01DA051018, U01DA051037, U01DA050987, U01DA041174, U01DA041106, U01DA041117, U01DA041028, U01DA041134, U01DA050988, U01DA051039, U01DA041156, U01DA041025, U01DA041120, U01DA051038, U01DA041148, U01DA041093, U01DA041089, U24DA041123, U24DA041147.  
A full list of supporters is available at [https://abcdstudy.org/federal-partners.html](https://abcdstudy.org/federal-partners.html).  
A listing of participating sites and a complete listing of the study investigators can be found at [https://abcdstudy.org/consortium_members/](https://abcdstudy.org/consortium_members/).

ABCD consortium investigators designed and implemented the study and/or provided data but did not necessarily participate in the analysis or writing of this report. This manuscript reflects the views of the authors and may not reflect the opinions or views of the NIH or ABCD consortium investigators.

---

## Code Overview

This repository contains all scripts used to perform the regularized CCA analysis, cross-validation, and permutation testing.

- `cca_abcd_rsfmri_cbcl.py` — Main script for running CCA with cross-validation.
- `cca_abcd_rsfmri_cbcl_utils.py` — Utility functions for data loading and preprocessing.
- `permutation_analysis.py` — Script to perform permutation testing for full cohort and sex-stratified models.
- `permutation_analysis_utils.py` — Supporting functions for permutation training and correlation analysis.
- `gather_cca_abcd_rsfmri_cbcl.py` — Script to gather and summarize cross-validation results.
- `launch_cca_abcd_rsfmri_cbcl.py` — Helper script to launch CCA runs across parameter configurations.
- `run_cca.sh` — Slurm batch script to launch CCA jobs via `launch_cca_abcd_rsfmri_cbcl.py`.
- `run_perm.sh` — Slurm batch script to launch permutation analysis via `permutation_analysis.py`.

---

## Citation

If you use this codebase or reference this work in your research, please cite our corresponding paper:

**[TO BE COMPLETED]**
