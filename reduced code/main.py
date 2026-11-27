"""
main.py
-------
Intrusion Detection System on CSE-CIC-IDS2018 (Machine Learning project, Fall 2026)

HOW TO RUN
  1. Put the 10 CSV files in a folder called "data" next to this file.
  2. Install the libraries (one time):
       pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy
  3. Run:
       python main.py
     Set QUICK_MODE = True below for a small, fast test run first.

Results are saved in:
  results/tables/     (CSV tables)
  results/figures/    (the confusion matrix chart of each model)
While it runs, it prints a short message each time a step is finished.

This file runs the experiment from top to bottom, one step at a time:
  1  Load and clean the data            6  Final test results
  3  Train/test split + feature filter  9  Error analysis
  4  Hyperparameter tuning              11 Feature importance
  5  K-fold cross-validation

Every model is trained on PCA data: log -> scaler -> PCA (keeps 95% of the
information) -> model. See build_pipeline() in models.py.

(This is the reduced version. These steps were removed: 2 Explore the data
(charts), 7 Ablation study, 8 Leave-one-attack-out, 10 Fairness audit,
12 Human review and 13 Anomaly detection. The other steps keep their
numbers, so they still match the names of the result files.)

The project has 4 files:
  main.py                -> this file: runs the steps and works out the results
  data_prep.py           -> loading and cleaning the data
  models.py              -> building, training and testing the models
  figures_and_tables.py  -> every table (.csv) and chart (.png) that is saved
"""

import os

# Use 1 thread for the math libraries (like OMP_NUM_THREADS in Lab 3).
# On this Mac, two of them can get stuck waiting on each other and the run
# freezes in Step 4. This must come BEFORE numpy is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import data_prep                      # our file data_prep.py
import models                         # our file models.py
import figures_and_tables as ft       # our file figures_and_tables.py ("ft" is a short name, like "pd" for pandas)

# =============================================================================
# SETTINGS (change the numbers here)
# =============================================================================
QUICK_MODE = False           # True = small fast test run (about 5-10 minutes)

# The data is not copied into this folder (it is 6.4 GB). Instead this points to
# the data folder of ids_project_simple. ".." means "the folder above this one".
DATA_FOLDER = os.path.join("..", "ids_project_simple", "data")
SEED = 42                    # same seed = same results every run

BENIGN_FRACTION = 0.02       # keep 2% of normal traffic
ATTACK_FRACTION = 0.20       # keep 20% of attack traffic (and all Web Attacks)
MAX_BENIGN = 150_000         # at most this many normal rows
MAX_PER_ATTACK = 30_000      # at most this many rows of each attack family

TEST_SIZE = 0.20             # 80% train, 20% test
FOLDS = 5                    # K in K-fold cross-validation
CV_ROWS = 50_000             # rows used for cross-validation (for speed)
TUNING_ROWS = 30_000         # rows used for tuning (for speed)

if QUICK_MODE:               # much smaller numbers for a fast test run
    MAX_BENIGN = 6_000
    MAX_PER_ATTACK = 1_500
    CV_ROWS = 4_000
    TUNING_ROWS = 3_000


def subset(X, y, n):
    """Take n rows with the same % of each family (stratified), for speed."""
    if len(X) <= n:
        return X, y
    X_small, X_rest, y_small, y_rest = train_test_split(X, y, train_size=n, stratify=y, random_state=SEED)
    return X_small, y_small


# =============================================================================
# The experiment
# =============================================================================
start = time.time()                  # the time now (used at the end to show how long the run took)
print(f"Starting the experiment (QUICK_MODE = {QUICK_MODE})")


# -----------------------------------------------------------------------------
# STEP 1: Load and clean the data
# -----------------------------------------------------------------------------
df = data_prep.load_data(DATA_FOLDER, BENIGN_FRACTION, ATTACK_FRACTION, MAX_BENIGN, MAX_PER_ATTACK, SEED)
df, cleaning_counts = data_prep.clean_data(df)
ft.table_01_cleaning_report(cleaning_counts)

# The features are every column except these 3
features = []
for col in df.columns:
    if col not in ["Label", "Family", "Day"]:
        features.append(col)

print("Finished Step 1: loading and cleaning the data")


# -----------------------------------------------------------------------------
# STEP 3: Train/test split and feature selection
# -----------------------------------------------------------------------------
# Turn the family names into numbers 0..6 (like LabelEncoder in Lab 2)
encoder = LabelEncoder()
y = encoder.fit_transform(df["Family"])
families = list(encoder.classes_)            # the family names, in the same order as the numbers
benign = families.index("Benign")            # the number that means "Benign"

# Split into training and test data (like in Lab 2).
# stratify=y keeps the same % of each family in both parts.
X_all = df[features]
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

# Remove features that are near-copies of another one (found on the training data only)
dropped = data_prep.find_correlated(X_train_all)
kept = []
for f in features:
    if f not in dropped:
        kept.append(f)
X_train = X_train_all[kept]
X_test = X_test_all[kept]

print("Finished Step 3: train/test split and feature selection")


# -----------------------------------------------------------------------------
# STEP 4: Hyperparameter tuning (GridSearchCV)
# -----------------------------------------------------------------------------
X_tune, y_tune = subset(X_train, y_train, TUNING_ROWS)
all_models = models.make_models()
tuned = {}               # model name -> tuned pipeline
tuning_scores = {}       # model name -> best macro F1 found while tuning
tuning_settings = {}     # model name -> the settings that gave it

for name in all_models:
    pipeline = models.build_pipeline(all_models[name])
    best_pipeline, best_settings, best_score = models.tune(name, pipeline, X_tune, y_tune)
    tuned[name] = best_pipeline
    tuning_settings[name] = best_settings
    tuning_scores[name] = best_score

tuned["Stacking"] = models.make_stacking(tuned)     # built from the tuned Random Forest and XGBoost
ft.table_04_tuning(tuning_scores, tuning_settings)
print("Finished Step 4: hyperparameter tuning")


# -----------------------------------------------------------------------------
# STEP 5: K-fold cross-validation (K = FOLDS)
# -----------------------------------------------------------------------------
X_cv, y_cv = subset(X_train, y_train, CV_ROWS)
X_cv = X_cv.reset_index(drop=True)       # number the rows 0, 1, 2, ... again
cv = {}                                  # model name -> list of fold scores

for name in tuned:
    cv[name] = models.cross_validate(name, tuned[name], X_cv, y_cv, FOLDS)

# This table is sorted best model first, and compares every model with the
# simple baseline (Logistic Regression)
cv_table = ft.table_05_cross_validation(cv)
ft.table_05_cv_fold_scores(cv)

# The best model is chosen by cross-validation, never by the test set
ranking = list(cv_table["model"])        # the model names, best first
best_name = ranking[0]
print(f"Finished Step 5: cross-validation (best model: {best_name})")


# -----------------------------------------------------------------------------
# STEP 6: Train on all training data, then test
# -----------------------------------------------------------------------------
predictions = {}           # model name -> predicted family of every test flow
test_scores = {}           # model name -> dictionary of scores (from models.scores)
train_seconds = {}         # model name -> training time

for name in tuned:
    pipeline = tuned[name]
    t0 = time.time()
    models.train(name, pipeline, X_train, y_train)
    train_seconds[name] = time.time() - t0

    probabilities = pipeline.predict_proba(X_test)            # one column per family
    predictions[name] = probabilities.argmax(axis=1)          # the family with the highest probability

    test_scores[name] = models.scores(y_test, predictions[name], benign)
    ft.figure_06_confusion(y_test, predictions[name], families, name)

ft.table_06_test_results(test_scores, train_seconds)

# Results per family for the best model
best_pred = predictions[best_name]
report = classification_report(y_test, best_pred, target_names=families, output_dict=True)
# output_dict=True gives a dictionary, e.g. report["DoS"]["recall"]
ft.table_06_per_family_best_model(report, families)

# How many PCA components were kept (every pipeline learns the same PCA from X_train)
pca = tuned[best_name].named_steps["pca"]
print(f"Finished Step 6: training and testing (PCA turned the {len(kept)} features into {pca.n_components_} components)")


# -----------------------------------------------------------------------------
# STEP 9: Error analysis (best model)
# -----------------------------------------------------------------------------
matrix = confusion_matrix(y_test, best_pred)      # counts, not %
ft.table_09_top_mistakes(matrix, families)
print("Finished Step 9: error analysis")


# -----------------------------------------------------------------------------
# STEP 11: Feature importance (Random Forest, PCA components)
# -----------------------------------------------------------------------------
# With PCA, the models see PCA components (PC1, PC2, ...) instead of the
# original features, so the importances belong to the components
rf_pipeline = tuned["Random Forest"]
forest = rf_pipeline.named_steps["model"]
X_test_pca = models.pca_components(rf_pipeline, X_test)     # the test flows as PC1, PC2, ...
component_names = list(X_test_pca.columns)
importance = ft.table_11_feature_importance(component_names, forest.feature_importances_)   # top 15, most important first

# Why are attacks missed? Find the family with the lowest recall (the hardest one)
hardest = None
lowest_recall = 2                                 # higher than any recall, so the first family replaces it
for family in families:
    if family != "Benign":
        if report[family]["recall"] < lowest_recall:
            lowest_recall = report[family]["recall"]
            hardest = family

top = list(importance["component"].head(8))       # the 8 most important components
ft.table_11_missed_vs_caught(X_test_pca, y_test, best_pred, families, benign, hardest, top)
print("Finished Step 11: feature importance")


minutes = (time.time() - start) / 60
print(f"Done in {round(minutes, 1)} minutes. The results are in results/tables and results/figures")
