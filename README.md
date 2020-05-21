# Code for deep Bayesian dose-response modeling

This is the implementation of the paper:

```
Dose-Response Modeling in High-Throughput Cancer Drug Screenings: An end-to-end approach
W. Tansey, K. Li, H. Zhang, S. W. Linderman, D. M. Blei, R. Rabadan, and C. H. Wiggins
Preprint, December 2018. https://arxiv.org/abs/1812.05691
```

## Running the code

The pipeline is meant to be run in steps. First is the GDSC-specific preprocessing code to handle the contamination and negative/positive control modeling:

1) `python python/step1_remove_contamination.py data/raw.csv`

2) `python python/step2_fit_negative_control_priors.py data/raw_step1.csv`

3) `python python/step3_fit_positive_control_priors.py data/raw_step2.csv`

After you have those, you can fit the deep learning model, fit the curve posteriors, and run the biomarker tests:

4) Two options:
    i) `python python/step4_fit_prior_missing.py --name model_name` -- this runs a slower version that handles missing features via an embedding approach. The results here are used to generate the evaluations in Section 3 of the paper.

    ii) `python python/step4_fit_prior_fast.py model_name`  -- this runs a fast version of the predictive model. The tradeoff is a slightly worse performing model, but still much better than the standard approaches. This is used for the conditional randomization testing procedure in Section 4.

    If you care more about prior predictions, go with step (i). If you care about fitting posterior dose-response curves to actual data, use step (ii). In general, the prior will be fairly weak relative to the data likelihood, so you're better off going with a fast model for posterior fits.

    All the code in the subsequent pipeline assumes you are running using the fast model.

5) `python python/step5_fit_posteriors.py model_name --drug drugnum` where `drugnum` is the index of the drug in the step4 model. For GDSC, this should be a number between 0 and 264. You can run all the drugs independently in parallel.

6) `python python/step6_factorize_features.py data/gdsc_all_features.csv` -- creates the biomarker features and runs a binary matrix factorization algorithm (courtesy of Jackson Loper) on them to get k=50 latent factors.

7) `python python/step7_test_biomarkers.py model_name --drug drugnum` -- runs the amortized conditional randomization test for the specified drug. All drugs can again be run independently in parallel.

A fully-trained version of the fast model (approximately 50 GB) is available via Dropbox here: https://www.dropbox.com/sh/e8pspfdoy9vhzr6/AADLv-inQVa-9xd2ieaUijI3a?dl=0

All data is from the [Genomics of Drug Sensitivity in Cancer](https://www.cancerrxgene.org/). This pipeline is specifically based on version 1 of the dataset.

