# AssurPrime Data Challenge — Fire Pure-Premium Modelling

Data challenge on multi-risk agricultural insurance (Pacifica, Crédit Agricole Assurances), carried out at École Centrale de Nantes by **Théa El Rostom, Agathe Guerrini-Touchais and Lucas Cabrera**.

📄 Full write-up (French): [`reports/DataChallenge_Cabrera_Guerrini_El_Rostom.pdf`](reports/DataChallenge_Cabrera_Guerrini_El_Rostom.pdf)
📓 Code: [`notebooks/notebook_fusionne_corrige_avec_stacking.ipynb`](notebooks/notebook_fusionne_corrige_avec_stacking.ipynb)

## Objective

Predict the annual claim cost (`CHARGE`) of agricultural fire-insurance contracts and beat the insurer's GLM benchmark. The target decomposes as:

```
CHARGE = FREQ × CM × ANNEE_ASSURANCE
```

(claim frequency × average claim cost × exposure over the policy year). Models are ranked on a single metric, **RMSE** on `CHARGE`, and also judged on interpretability and business constraints (GDPR, no black-box models).

**Benchmark:** Poisson GLM (frequency, log link) + Tweedie GLM (average cost, log link).

## Data

- 383,610 training contracts, ~370 explanatory variables: contract data (activity, guarantees, insured capital, surfaces, prevention measures), geography, CORINE land-use, demographics, building geometry, and weather (temperature, wind, rainfall).
- Highly skewed target: ~90% of contracts have zero claims, a few very large claims dominate the error. Many columns are sparsely filled.
- **The data is not included in this repository** (see [Reproducing](#reproducing)).

## Methodology

1. **Preprocessing** — missing values set to `0` (numeric) / `-999` (categorical); `ID` and `ANNEE_ASSURANCE` removed from features; negative `CM` clipped to 0 (required by Tweedie). One-hot encoding + standardisation for GLMs, `CountEncoder` for scikit-learn trees, native categorical handling for LightGBM.
2. **Models tested**
   - GLM reference (Poisson / Tweedie), penalty tuned by `GridSearchCV`
   - Random Forest / Gradient Boosting pairs (RF-GB, RF-RF, GB-GB, GB-RF) compared by 80/20 split and K-Fold CV
   - LightGBM (Poisson / Tweedie), ~25 → 4 min runtime versus scikit-learn boosting; Bayesian hyperparameter search with Optuna
   - Lasso fitted directly on `CHARGE`
   - Discarded: KNN (memory), linear regression on `FREQ` (R² ≈ 0.003, negative predictions)
3. **Stacking** — out-of-fold predictions of three base approaches (GLM, course models, LightGBM) combined by a linear meta-model, compared two ways: non-negative least squares (K-Fold) and Ridge with λ selected by **exact closed-form Leave-One-Out** error, `(ŷᵢ − yᵢ) / (1 − hᵢᵢ)`, verified against brute-force LOO on a subsample.

## Key findings

| Model | RMSE |
|---|---|
| GLM Poisson / Tweedie (reference) | 6802.77 |
| GB-GB (80/20 split) | 6662.10 |
| GB-GB (K-Fold) | 6802.29 |
| RF-RF (K-Fold) | 7002.68 |
| LightGBM Poisson / Tweedie | 6802.64 |
| LightGBM + Bayesian optimisation | 6794.38 |
| Stacking, non-negative LS (5-fold, notebook) | 6798.50 |
| Stacking, Ridge + exact LOO (notebook) | 6798.54 |
| **Lasso on `CHARGE` (α = 10)** | **6795.84** |

- Gaps between well-tuned models are under ~10 RMSE points: the error is dominated by a handful of very large claims.
- The 80/20 split flattered tree models; with K-Fold CV they fall back to the benchmark level. Unconstrained RF-RF overfits extreme claims and is clearly worst.
- Further hyperparameter tuning slightly *degraded* public leaderboard scores (overfitting to train); the Lasso gave the real improvement.
- **Retained approach: Lasso** — competitive accuracy while remaining interpretable, which matters for a regulated insurer.
- Limitations: missing values imputed as 0 with no feature engineering, FREQ and CM tuned separately though evaluated through their product, CM model trained on all contracts including non-claims.

## Repository structure

```
Data_Challenge_Assurprime/
├── notebooks/   # merged notebook: GLM, RF/GB, LightGBM, stacking, Optuna, Lasso
├── reports/     # final report (PDF, in French)
├── data/        # place challenge CSVs here (git-ignored)
├── README.md
└── .gitignore
```

## Reproducing

1. Obtain the challenge files (`train_input.csv`, `train_output.csv`, `test_input.csv`) and place them in the notebook's working directory (or `data/`, adjusting paths).
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn lightgbm optuna category_encoders jupyter
   ```
3. Run `notebooks/notebook_fusionne_corrige_avec_stacking.ipynb` top to bottom. The RF/GB and stacking sections are the slowest.

## Authors

Théa El Rostom · Agathe Guerrini-Touchais · Lucas Cabrera — École Centrale de Nantes, 2026.

*Generative AI (Gemini, Claude) was used for literature search, debugging and rewording; see the report's appendix.*
