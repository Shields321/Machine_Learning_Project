"""
models.py
---------
Everything to do with the MODELS:
  1. make_models()      -> the 4 models (Logistic Regression is the simple baseline)
  2. make_pipeline()    -> log transform -> scaler -> (optional PCA) -> model
  3. tune()             -> hyperparameter tuning with RandomizedSearchCV
  4. make_stacking()    -> Random Forest + XGBoost combined (stacking)
  5. train()            -> fit a model (handles XGBoost's class weights)
  6. Evaluation tools   -> scores, K-fold cross-validation, statistical tests
"""

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42


# ---------------------------------------------------------------------------
# Building the models
# ---------------------------------------------------------------------------
def signed_log(X):
    """Most flows are tiny and a few are huge. log(1 + x) squashes the huge
    values so Logistic Regression and PCA work better (the sign is kept
    because a few columns can be -1)."""
    return np.sign(X) * np.log1p(np.abs(X))


def make_pipeline(model, use_pca=False):
    """Pipeline = the preparation steps + the model, in one object.
    The scaler (and PCA) only learn from the training data inside each fold,
    so nothing leaks from the test data."""
    steps = [("log", FunctionTransformer(signed_log)), ("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=SEED)))   # keep 95% of the information
    steps.append(("model", model))
    return Pipeline(steps)


def make_models(balanced=True):
    """The 4 models. balanced=True makes rare attacks count more (class weights).
    XGBoost has no class_weight setting, so it gets weights in train() instead."""
    weight = "balanced" if balanced else None
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=weight),
        "Decision Tree": DecisionTreeClassifier(class_weight=weight, random_state=SEED),
        "Random Forest": RandomForestClassifier(class_weight=weight, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED),
    }


# Settings that tuning can choose from ("model__" = the "model" step of the pipeline)
SETTINGS_TO_TRY = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"model__max_depth": [5, 10, 20, None],
                      "model__min_samples_leaf": [1, 2, 5, 10]},
    "Random Forest": {"model__n_estimators": [100, 200, 300],
                      "model__max_depth": [10, 20, None],
                      "model__min_samples_leaf": [1, 2, 4]},
    "XGBoost": {"model__n_estimators": [200, 300, 500],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1, 0.2]},
}


def train(name, pipeline, X, y, balanced=True):
    """Fit a pipeline. XGBoost gets sample weights so rare attacks count more."""
    if name == "XGBoost" and balanced:
        pipeline.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipeline.fit(X, y)
    return pipeline


def tune(name, pipeline, X, y, tries):
    """Hyperparameter tuning: RandomizedSearchCV tries `tries` random settings
    with 3-fold cross-validation and keeps the best one.
    It scores with macro F1 (not accuracy) so the rare attacks count as much
    as the common ones."""
    search = RandomizedSearchCV(pipeline, SETTINGS_TO_TRY[name], n_iter=tries,
                                scoring="f1_macro", n_jobs=-1, random_state=SEED, refit=False,
                                cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
    if name == "XGBoost":
        search.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        search.fit(X, y)
    pipeline.set_params(**search.best_params_)            # use the best settings
    return pipeline, search.best_params_, search.best_score_


def make_stacking(tuned):
    """Stacking: the tuned Random Forest and XGBoost make predictions, and a
    Logistic Regression (the meta model) learns how much to trust each one."""
    stack = StackingClassifier(
        estimators=[("rf", tuned["Random Forest"].named_steps["model"]),
                    ("xgb", tuned["XGBoost"].named_steps["model"])],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3)
    return make_pipeline(stack)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def scores(y_true, y_pred, benign):
    """The main scores.
    detection rate   = % of attacks flagged as any attack (caught)
    false alarm rate = % of normal flows wrongly flagged as an attack"""
    attack = y_true != benign
    flagged = y_pred != benign
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro F1": f1_score(y_true, y_pred, average="macro"),
            "detection rate": flagged[attack].mean(),
            "false alarm rate": flagged[~attack].mean()}


def cross_validate(name, pipeline, X, y, folds, repeats):
    """Repeated stratified K-fold cross-validation.
    Train on K-1 parts, test on the last part, rotate, then repeat everything
    with a new shuffle. Returns one macro F1 score per fold."""
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    fold_scores = []
    for train_rows, test_rows in splitter.split(X, y):
        model = train(name, clone(pipeline), X.iloc[train_rows], y[train_rows])
        prediction = model.predict(X.iloc[test_rows])
        fold_scores.append(f1_score(y[test_rows], prediction, average="macro"))
    return fold_scores


def corrected_t_test(scores_a, scores_b, folds):
    """Is model A really better than model B, or just lucky?
    A normal t-test is too optimistic for K-fold scores (the folds share
    training data), so we use the Nadeau & Bengio (2003) correction.
    Returns (mean difference, p-value)."""
    diff = np.array(scores_a) - np.array(scores_b)
    k = len(diff)
    test_share = 1 / folds                                  # e.g. 1/5 of the data is the test fold
    variance = (1 / k + test_share / (1 - test_share)) * diff.var(ddof=1)
    if variance == 0:
        return diff.mean(), 1.0
    t = diff.mean() / np.sqrt(variance)
    return diff.mean(), 2 * stats.t.sf(abs(t), df=k - 1)


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test on the test set: looks only at flows where exactly one
    of the two models is right. Returns (A right & B wrong, A wrong & B right, p-value)."""
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, stats.chi2.sf(chi2, df=1)


def bootstrap_ci(y_true, y_pred, rounds=1000):
    """95% confidence interval for macro F1: resample the test set many times
    and keep the middle 95% of the scores."""
    rng = np.random.default_rng(SEED)
    results = []
    for _ in range(rounds):
        rows = rng.integers(0, len(y_true), len(y_true))
        results.append(f1_score(y_true[rows], y_pred[rows], average="macro"))
    return np.percentile(results, 2.5), np.percentile(results, 97.5)


def service(port):
    """Turn a destination port number into the service it belongs to
    (used in the fairness audit)."""
    names = {21: "FTP (21)", 22: "SSH (22)", 53: "DNS (53)", 80: "HTTP (80)",
             8080: "HTTP (8080)", 443: "HTTPS (443)", 3389: "Remote Desktop (3389)",
             139: "File sharing (139/445)", 445: "File sharing (139/445)"}
    port = int(port)
    if port in names:
        return names[port]
    return "Other well-known (<1024)" if port < 1024 else "High ports (1024+)"
