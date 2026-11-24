# AssurPrime — Agricultural Insurance Claims Modeling

**Data Challenge project**  
**Authors:** Théa El-Rostom · Agathe Guerrini-Touchais · Lucas Cabrera  
**Report date:** 2 November 2026

## Project overview

This project was developed for the AssurPrime data challenge, based on an agricultural multi-risk insurance use case. The objective is to predict the annual claim charge (`CHARGE`) for insured contracts and compare alternative machine-learning approaches with the insurer's Generalized Linear Model (GLM) benchmark.

The target is decomposed as:

$$
\text{CHARGE} = \text{FREQ} \times \text{CM} \times \text{ANNEE ASSURANCE}
$$


where `FREQ` is claim frequency, `CM` is average claim cost, and `ANNEE ASSURANCE` represents the contract's exposure duration during the financial year.

The evaluation metric is **Root Mean Squared Error (RMSE)**. The dataset described in the report contains 383,610 training contracts and approximately 370 explanatory variables.

## Repository structure

```text
Data_Challenge_Assurprime/
├── notebooks/
│   └── notebook_fusionne_corrige_avec_stacking.ipynb
├── reports/
│   └── DataChallenge_Cabrera_Guerrini_El Rostom.pdf
├── .gitignore
└── README.md
```

## Methodology

### 1. Data preparation and feature processing

- Handle missing numerical values by replacing them with `0` and missing categorical values with `-999`.
- Exclude identifier (`ID`) and exposure (`ANNEE ASSURANCE`) fields from the explanatory variables; exposure is used when reconstructing the final claim charge.
- Clip negative average-claim-cost targets to zero for compatibility with the Tweedie model.
- Use one-hot encoding for GLMs, count encoding for tree-based models, and native categorical handling for LightGBM.
- Standardize numerical features for the GLM workflow.

### 2. Models evaluated

- **Benchmark GLMs:** Poisson regression with a log link for claim frequency and Tweedie regression with a log link for average claim cost.
- **Tree-based models:** Random Forest and scikit-learn Gradient Boosting, evaluated through different frequency/cost model pairings.
- **LightGBM:** Poisson and Tweedie models, with native categorical-variable handling and hyperparameter optimization.
- **Lasso:** Direct regularized regression on `CHARGE`, using an L1 penalty.
- **Stacking:** Out-of-fold predictions from the GLM benchmark, the selected course-model combination, and LightGBM are combined by a linear meta-model. Two approaches are explored: non-negative least squares and Ridge regression with its regularization parameter selected using exact Leave-One-Out calculations.

### 3. Validation and optimization

The work compares an initial 80/20 train-validation split with 3-fold cross-validation for base-model selection and hyperparameter search. The stacking meta-model is also evaluated using cross-validation; the Ridge regularization parameter is tuned using the exact Leave-One-Out formula for a linear model with three inputs.

## Key findings

The report's comparison table gives the following RMSE values:

| Model | RMSE |
|---|---:|
| Poisson GLM + Tweedie GLM (benchmark) | 6,802.77 |
| Random Forest frequency + Gradient Boosting cost (80/20 split) | 6,665.19 |
| Gradient Boosting frequency + Gradient Boosting cost (80/20 split) | 6,662.10 |
| Random Forest frequency + Random Forest cost (K-Fold) | 7,002.68 |
| Bayesian-optimized LightGBM (Poisson + Tweedie) | 6,794.38 |
| Lasso | **6,795.84** |

Main observations reported by the team:

- The Gradient Boosting pairings performed better than the benchmark on the initial 80/20 split, while the Random Forest–Random Forest pairing generalized poorly under K-Fold validation.
- LightGBM substantially reduced execution time in the reported setup (approximately 25 minutes to 4 minutes), but did not materially outperform the other approaches on RMSE.
- Bayesian optimization slightly improved the LightGBM result compared with the non-Bayesian LightGBM result.
- Lasso was retained as the preferred direct-charge approach in the report, with an RMSE of 6,795.84, and was valued for its regularization and interpretability.

**Metric context:** the report distinguishes scores obtained on a single 80/20 split, cross-validation scores, and training-set reference scores. These values should not be treated as directly interchangeable. The report does not provide a final public leaderboard score; none is inferred here.

## Limitations and further work

The report highlights several limitations:

- RMSE is highly sensitive to a small number of very large claims, while most contracts have a zero charge.
- Replacing missing values with fixed values does not distinguish missing observations from genuine zeros.
- No additional engineered features were created.
- Hyperparameters for frequency and average cost were optimized separately, although the final metric is computed on their product.
- The average-cost model was trained on all contracts, including contracts with no claim.
- More work is needed to ensure non-negative and well-calibrated cost predictions and to assess generalization on unseen data.

Potential next steps include richer missing-value indicators and feature engineering, direct optimization against the final charge metric, and more robust validation focused on the heavy-tailed claim distribution.

## Files

- [`notebooks/notebook_fusionne_corrige_avec_stacking.ipynb`](notebooks/notebook_fusionne_corrige_avec_stacking.ipynb): consolidated notebook covering preprocessing, benchmark GLMs, tree-based models, LightGBM, stacking, Bayesian optimization, and Lasso.
- [`reports/DataChallenge_Cabrera_Guerrini_El%20Rostom.pdf`](reports/DataChallenge_Cabrera_Guerrini_El%20Rostom.pdf): final project report (French).

## Reproducibility

The notebook expects the challenge input files and supplementary resources described in the report. Those source datasets are not included in this repository. Place the required files in the paths expected by the notebook before running it, and execute the notebook from top to bottom so that model objects and out-of-fold predictions are available to the stacking section.

The notebook uses Python data-science and machine-learning libraries, including pandas, NumPy, scikit-learn, and LightGBM. Exact package versions and the original challenge datasets were not specified as repository artifacts.
