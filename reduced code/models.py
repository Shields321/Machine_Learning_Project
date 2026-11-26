"""
models.py
---------
Everything to do with the MODELS. main.py uses these functions:

  Building and training the models
    build_pipeline()         -> log -> scaler -> PCA -> model, in one object
    pca_components()         -> turn flows into their PCA components (used in Step 11)
    make_models()            -> the 4 models (Logistic Regression is the simple baseline)
    train()                  -> fit a model (XGBoost gets its class weights here)
    tune()                   -> hyperparameter tuning with GridSearchCV
    make_stacking()          -> Random Forest + XGBoost combined (stacking, like Lab 2)

  Evaluation
    scores()                 -> accuracy, macro F1, detection rate, false alarm rate
    cross_validate()         -> K-fold cross-validation

(The tables and charts of the results are in figures_and_tables.py.)
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42      # same seed = same results every run


# =============================================================================
# Building and training the models
# =============================================================================
def signed_log(X):
    """Most flows are tiny and a few are huge. log(1 + x) squashes the huge
    values so Logistic Regression and PCA work better. np.sign keeps the
    minus sign, because a few columns can be -1."""
    return np.sign(X) * np.log1p(np.abs(X))


def build_pipeline(model):
    """A Pipeline puts the preparation steps and the model into ONE object,
    like make_pipeline(StandardScaler(), clf) in Lab 1.

    Why: when the pipeline is trained, the scaler and PCA only learn from
    the training data, so nothing leaks from the test data.

    Every step has a name. The model step is called "model", which is why the
    tuning settings below start with "model__" (two underscores)."""
    steps = []
    steps.append(("log", FunctionTransformer(signed_log)))     # 1. squash the huge numbers
    steps.append(("scaler", StandardScaler()))                 # 2. put every feature on the same scale

    # 3. PCA, written out like in Lab 3. It keeps 95% of the information.
    #    The pipeline calls pca.fit_transform() on the training data for us
    #    (like x_pca = pca.fit_transform(X) in Lab 3), and pca.transform()
    #    on the test data, so the test data never changes the components.
    pca = PCA(
        # Dimensionality
        n_components=0.95,          # keep 95% of the information (explained variance)

        # Centering & scaling
        whiten=False,

        # SVD solver
        svd_solver='auto',

        # Randomness & reproducibility
        random_state=SEED,

        # Other
        tol=0.0,
        iterated_power='auto'
    )
    steps.append(("pca", pca))

    steps.append(("model", model))                             # last step: the model itself
    pipeline = Pipeline(steps)
    return pipeline


def pca_components(pipeline, X):
    """Run the flows X through the log, scaler and PCA steps of a TRAINED
    pipeline, and return their PCA components as a table with the columns
    PC1, PC2, PC3, ... (used in Step 11, because with PCA the models see
    components, not the original features)."""
    X_done = pipeline.named_steps["log"].transform(X)
    X_done = pipeline.named_steps["scaler"].transform(X_done)
    X_done = pipeline.named_steps["pca"].transform(X_done)

    names = []                                  # PC1, PC2, PC3, ...
    for i in range(X_done.shape[1]):            # shape[1] = number of components
        names.append("PC" + str(i + 1))
    return pd.DataFrame(X_done, columns=names)


def make_models(balanced=True):
    """The 4 models, in a dictionary: name -> model.
    balanced=True makes rare attacks count more (class weights).
    XGBoost has no class_weight setting, so it gets its weights in train() instead."""
    if balanced:
        weight = "balanced"      # rare families count more
    else:
        weight = None            # every row counts the same

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=weight),
        "Decision Tree": DecisionTreeClassifier(class_weight=weight, random_state=SEED),
        "Random Forest": RandomForestClassifier(class_weight=weight, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED),
    }
    return models


# The settings that tuning can choose from (like param_grid in Lab 2).
# "model__C" means: the setting C of the step called "model" in the pipeline.
# GridSearchCV tries EVERY combination, so the lists are kept short (for run time).
SETTINGS_TO_TRY = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"model__max_depth": [5, 10, 20, None],
                      "model__min_samples_leaf": [1, 5, 10]},
    "Random Forest": {"model__n_estimators": [100, 200],
                      "model__max_depth": [10, 20, None],
                      "model__min_samples_leaf": [1, 4]},
    "XGBoost": {"model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.1, 0.2]},
}


def train(name, pipeline, X, y, balanced=True):
    """Train (fit) a pipeline. XGBoost gets sample weights so rare attacks count more."""
    if name == "XGBoost" and balanced:
        # Give every row a weight: rows of rare families get a big weight
        weights = compute_sample_weight("balanced", y)
        # model__sample_weight = give these weights to the step called "model"
        pipeline.fit(X, y, model__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    return pipeline


def tune(name, pipeline, X, y):
    """Hyperparameter tuning with GridSearchCV (like Lab 2): tries every
    combination of settings in SETTINGS_TO_TRY with 3-fold cross-validation
    and keeps the best one. It scores with macro F1 (the F1 score averaged
    over all 7 families) so rare attacks count as much as common ones."""
    grid = GridSearchCV(pipeline, SETTINGS_TO_TRY[name], scoring="f1_macro",
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                        n_jobs=-1, refit=False)
    if name == "XGBoost":
        grid.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))   # same class weights as in train()
    else:
        grid.fit(X, y)
    # use the best settings (** unpacks the dictionary, e.g. set_params(model__C=10))
    pipeline.set_params(**grid.best_params_)
    return pipeline, grid.best_params_, grid.best_score_


def make_stacking(tuned):
    """Stacking, like in Lab 2: the tuned Random Forest and XGBoost are the
    base models, and a Logistic Regression (the meta model) learns how much
    to trust each one."""
    rf_model = tuned["Random Forest"].named_steps["model"]     # take the tuned model out of its pipeline
    xgb_model = tuned["XGBoost"].named_steps["model"]

    base_models = [("rf", rf_model), ("xgb", xgb_model)]
    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced")

    stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3)
    return build_pipeline(stack)


# =============================================================================
# Evaluation
# =============================================================================
def scores(y_true, y_pred, benign):
    """The main scores, in a dictionary.
    detection rate   = share of attacks flagged as any attack (caught)
    false alarm rate = share of normal flows wrongly flagged as an attack"""
    attack = y_true != benign        # True where the flow really is an attack
    flagged = y_pred != benign       # True where the model said "attack" (any family)

    result = {}
    result["accuracy"] = accuracy_score(y_true, y_pred)
    result["macro F1"] = f1_score(y_true, y_pred, average="macro")
    result["detection rate"] = flagged[attack].mean()          # of the real attacks, how many were flagged
    result["false alarm rate"] = flagged[~attack].mean()       # of the normal flows, how many were flagged
    return result


def cross_validate(name, pipeline, X, y, folds):
    """Stratified K-fold cross-validation (like cross_val_score in Lab 1,
    but written out, because XGBoost needs its class weights from train()).
    Train on K-1 parts, test on the last part, and rotate until every part
    has been the test part once. Returns one macro F1 score per fold."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_scores = []

    # splitter.split gives the row numbers of the training part and the test part, one fold at a time
    for train_rows, test_rows in splitter.split(X, y):
        X_fold_train = X.iloc[train_rows]
        y_fold_train = y[train_rows]
        X_fold_test = X.iloc[test_rows]
        y_fold_test = y[test_rows]

        fresh = clone(pipeline)          # clone = a fresh, untrained copy with the same settings
        model = train(name, fresh, X_fold_train, y_fold_train)
        prediction = model.predict(X_fold_test)
        fold_scores.append(f1_score(y_fold_test, prediction, average="macro"))

    return fold_scores
