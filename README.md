# Regularized CCA Identifies Sex-Specific Brain-Behavior Associations in Adolescent Psychopathology

This project investigates sex-specific associations between resting-state functional connectivity (rs-fMRI) and psychiatric symptoms in early adolescence using the ABCD dataset. Using Canonical Correlation Analysis (CCA) and rigorous cross-validation, we identify multivariate brain-behavior mappings across eight Child Behavior Checklist (CBCL) syndrome scales. In contrast to previous studies, we explicitly model sex differences in these brain-behavior associations, revealing distinct neural correlates for attention, thought, and internalizing problems in males and females. Our findings highlight the importance of considering sex-specific mechanisms in adolescent psychopathology and support the development of sex-informed clinical strategies.

📄 Read the paper here: [DOI](https://doi.org/10.1038/s41398-025-03678-9)

---

## Data

Data used in the preparation of this article were obtained from the Adolescent Brain Cognitive Development℠ (ABCD) Study ([https://abcdstudy.org](https://abcdstudy.org)), publicly released in the NIMH Data Archive (NDA). This is a multisite, longitudinal study designed to recruit more than 10,000 children age 9–10 and follow them over 10 years into early adulthood.

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

L. Milecki, C. Gonzalez, E. Adeli, K. B. Nooner, M. R. Sabuncu, A. Kuceyeski, and Q. Zhao. ''Regularized CCA identifies sex-specific brain-behavior associations in adolescent psychopathology.'' Transl Psychiatry 15, 405 (2025). https://doi.org/10.1038/s41398-025-03678-9

Bibtex: **[TO BE COMPLETED]**
