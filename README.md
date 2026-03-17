# Before the Bubble Burst: Predicting Housing Prices in Ames, IA

**Author:** Deval Mehta &nbsp;|&nbsp; **Context:** General Assembly Data Science Bootcamp — Project 2 &nbsp;|&nbsp; **Stack:** Python, pandas, scikit-learn, seaborn, NumPy

---

## What It Does

This project builds and iteratively improves a set of linear regression models to predict residential sale prices in Ames, Iowa using the Ames Housing Dataset — an 80-feature dataset compiled from tax appraisals circa 2008. Starting from a single-feature baseline, I work through native numeric features, ordinal encoding of quality scales, engineered interaction terms, and feature scaling, documenting each modeling decision and its measured impact on predictive accuracy. The final model achieves an RMSE of approximately $32,000 against a $20,000 target — a meaningful result that surfaces exactly where more sophisticated methods are needed next.

**Example output:** True vs. predicted sale price scatter plots with an equality line, generated at each modeling stage to visually diagnose residual patterns.

---

## Why I Built This

**Context:** This was Project 2 of the General Assembly Data Science Bootcamp, completed as part of an internal Kaggle competition with my cohort. The project was assigned partway through the bootcamp, at the point where we had covered linear regression, basic feature engineering, and exploratory data analysis — but not yet regularization, cross-validation at scale, or tree-based models. The goal was to show how far careful feature selection and incremental preprocessing could take a linear model.

**Problem:** The Ames Housing Dataset is a deliberately messy, real-world dataset — 79 features, abundant missing values, a mix of numeric, ordinal, and nominal categorical variables, and non-trivial multicollinearity. A naive approach of just throwing numeric features at a LinearRegression model produces something, but the interesting question is *how much* systematic feature engineering can move the needle before you need a fundamentally different algorithm.

**Solution approach:** I treated the project as a controlled experiment rather than a one-shot model build. Each modeling stage adds exactly one new preprocessing decision — first native numerics, then ordinal encoding, then interaction terms, then scaling — so the marginal impact of each choice is measurable. I used RMSE as the primary evaluation metric (it's in the same units as sale price, making results interpretable) alongside R² to track overfitting.

**Results:** The best model achieved an expected Kaggle RMSE of approximately $32,000 — better than the ~$35,000 baseline but short of the $20,000 target. The persistent ~80% R² across all model iterations was the key insight: the ceiling wasn't model complexity, it was feature representation. One-hot encoding of neighborhood and building type, logarithmic transformation of sale price, and regularization (Lasso/Ridge) were the logical next steps — all of which I documented as explicit next steps.

---

## What I Learned

**Technical skills**
- Ordinal encoding of quality-scale categorical features using custom mapping dictionaries — a cleaner alternative to one-hot encoding for ordered categorical data
- Adapting the elbow method (from PCA) to correlation coefficient plots for feature selection — a practical heuristic when there's no obvious threshold
- Multicollinearity detection combining heatmaps with pairwise scatter plots, and making principled decisions about which of a collinear pair to retain
- Manually engineering interaction terms by multiplying feature pairs (e.g., `Fireplace Qu * Fireplaces`) to capture joint effects that linear models can't represent individually
- StandardScaler workflow: fit on training data only, then transform both train and test — a discipline I got wrong on my first pass (see Limitations)

**Data science insights**
- The ~80% R² ceiling across all modeling stages was the project's most useful finding. Incremental improvements shrank RMSE by roughly $500–$1,500 per stage, but the persistent tail of high-value underestimates pointed to a structural issue: high-end homes have characteristics that linear combinations of these features can't capture without transforming the target or introducing nonlinearity
- Scaling had less impact than I expected. Standard scaling doesn't change the predictions of OLS linear regression with fixed features — it only affects the coefficient magnitudes. The visual improvement I anticipated was more about reducing feature scale disparities than fundamentally improving the model
- The one-hot encoding attempt failed due to a column alignment mismatch between the training and validation sets after `pd.get_dummies()`. This is a classic real-world gotcha: `get_dummies()` applied separately to two DataFrames with different category distributions produces different columns. The fix (using `sklearn`'s `OneHotEncoder` within a pipeline) was the right approach but ran out of time

**Software engineering practices**
- During standardization (post-bootcamp), I replaced all `inplace=True` calls with explicit assignment — a subtler point than it looks, since `inplace` doesn't guarantee efficiency and actively obscures what the operation returns
- Reproducibility discipline: most of my `train_test_split` calls lacked a `random_state`, making results non-reproducible run to run. I've since added this consistently in subsequent projects

**Unexpected learnings**
- The StandardScaler data leakage bug I found post-bootcamp: I applied `fit_transform()` to both the training and test sets instead of `fit_transform()` on train and `transform()` on test only. The test set effectively used its own mean and standard deviation rather than the training set's, which slightly inflated the apparent test performance. The effect was small given the similar distributions, but it's the kind of mistake that matters at scale or with smaller samples
- Removing rows with null values in the training set while imputing zeros in the validation set is an asymmetric and somewhat arbitrary treatment. In production, the right answer is a fitted imputer applied consistently to both splits — which the code imports but doesn't actually use for this purpose

---

## Quick Start

This project was built before I adopted virtual environment conventions, so there's no `requirements.txt`. The dependencies are standard data science libraries available via pip or conda.

```bash
# Clone the repository
git clone https://github.com/dmehta94/project-2-ames-housing-prediction.git
cd project-2-ames-housing-prediction

# Install dependencies (in a virtual environment of your choice)
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch the notebook
jupyter notebook code/Project-2-Deval.ipynb
```

**Dataset:** The Ames Housing Dataset is required but not included in this repository due to its original Kaggle competition context. Download the `train.csv` and `test.csv` files from the [Ames Housing Dataset on Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) and place them in the `datasets/` directory.

**Directory structure:**
```
project-2-ames-housing-prediction/
├── code/
│   └── Project-2-Deval.ipynb    # Main analysis notebook
├── datasets/                     # train.csv and test.csv (not included)
├── images/                       # Generated plots (saved by notebook)
├── outputs/                      # Kaggle submission CSVs
└── README.md
```

---

## Sample Output

The notebook generates a series of true-vs-predicted scatter plots at each modeling stage, overlaid with the line of equality. Early models show a pronounced curved tail at the high end — high-value homes are systematically underestimated — that persists through scaling, confirming that a log-transform of the target would be the right next step.

Kaggle public leaderboard RMSE by submission:

| Model | Kaggle RMSE |
|---|---|
| Baseline (Overall Qual only) | $81,146 |
| Native numeric features (14) | $45,278 |
| + Ordinal quality features | $33,851 |
| + Interaction terms | $32,484 |
| + Scaling | $32,484 |

The largest single improvement came from moving beyond the baseline to a 14-feature numeric model — a $36,000 drop in RMSE. Ordinal encoding and interaction terms each closed the gap further, though with diminishing returns. Scaling had no effect whatsoever, which is exactly what the theory predicts: OLS linear regression is mathematically scale-invariant with a fixed feature set, so standard scaling cannot change predictions.

---

## Technical Details

**Stack:** Python 3, pandas, NumPy, scikit-learn, Matplotlib, Seaborn

**Modeling approach:** Ordinary least squares linear regression (`sklearn.linear_model.LinearRegression`) with train-test split evaluation. No cross-validation (time constraint noted in the notebook).

**Key preprocessing decisions:**

- *Missing value imputation:* Categorical NaNs replaced with `"NV"` (no value) string before ordinal encoding. Numeric NaN rows dropped from training data after confirming which rows contain nulls; validation set NaNs replaced with 0 (expedient, not ideal — see Limitations)
- *Ordinal encoding:* 13 quality/condition features mapped to integer scales (0–5 for most; custom scales for `Bsmt Exposure` and `BsmtFin Type 1/2`)
- *Feature selection:* Elbow method applied to a sorted correlation plot against `SalePrice`, yielding a 14-feature numeric candidate set before ordinal features were added
- *Multicollinearity:* Collinear pairs identified via heatmap (|r| > 0.7) and confirmed visually; one feature dropped per pair (e.g., `Total Bsmt SF` dropped in favor of `1st Flr SF`)
- *Interaction terms:* Three engineered features — `Kitchen Exter Qual`, `Fireplace Index`, `Garage Space` — capturing joint effects of quality × presence or area × capacity

**Key functions and structure:** The notebook is procedural rather than modular — appropriate for a bootcamp analytical notebook. Each modeling stage is self-contained: define features, split, fit, score, visualize, submit.

---

## Limitations

- **The $20,000 RMSE target was not achieved.** The gap between ~$32,000 and $20,000 is real and points to what's missing: one-hot encoding of neighborhood and building type, a logarithmic transformation of `SalePrice` to handle the right skew, and regularization (LassoCV) to handle the increased feature dimensionality that would follow
- **One-hot encoding failed.** The `pd.get_dummies()` approach produced column misalignment between the training and validation sets, since different category values appeared in each. The correct solution is `sklearn`'s `OneHotEncoder` in a pipeline — imported but not implemented due to time
- **Data leakage in the scaling section.** I applied `fit_transform()` to both training and test sets instead of fitting only on training. The effect on results was small given similar distributions, but this is corrected in the standardized version of the notebook
- **No cross-validation.** Train-test split with a single random seed leaves results somewhat sensitive to the split. KFold CV (5 folds) would produce more stable RMSE estimates
- **Asymmetric null treatment.** Dropping null rows from training data while zero-imputing the validation set creates an inconsistency. A fitted `SimpleImputer` applied to both would be the correct approach
- **Logarithmic regression section is incomplete.** The final notebook section ("Taming the Tail") was started but not finished due to competition time constraints. The empty code cell is preserved as-is for historical accuracy

---

## Credits

This project was completed independently as part of the General Assembly Data Science Bootcamp (Project 2). The Ames Housing Dataset was created by Dean De Cock and is available via Kaggle. Post-bootcamp documentation, code standardization, and this README were produced in collaboration with Claude AI (Anthropic, 2025).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

**Contact:** [GitHub @dmehta94](https://github.com/dmehta94) | [LinkedIn](https://www.linkedin.com/in/devalmehta94)