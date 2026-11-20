"""
main.py
-------
Intrusion Detection System on CSE-CIC-IDS2018 (Machine Learning project, Fall 2026)

HOW TO RUN
  1. Put the 10 Kaggle CSV files in a folder called "data" next to this file.
  2. Install the libraries (one time):
       pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy
  3. Run:
       python main.py
     Set QUICK_MODE = True below for a small, fast test run first.

Results are saved in:
  results/tables/   (CSV tables)
  results/figures/  (PNG charts)
  results/run_log.txt (everything that was printed)

STEPS
  1  Load and clean the data            7  Ablation study
  2  Explore the data (charts)          8  Leave-one-attack-out (new attacks)
  3  Train/test split + feature filter  9  Error analysis
  4  Hyperparameter tuning              10 Fairness audit
  5  K-fold cross-validation + tests    11 Feature importance
  6  Final test results                 12 Human review of unsure predictions
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_prep import clean_data, explore, find_correlated, load_data
from models import (bootstrap_ci, corrected_t_test, cross_validate, make_models,
                    make_pipeline, make_stacking, mcnemar_test, scores, service,
                    train, tune)

# =============================================================================
# SETTINGS (change the numbers here)
# =============================================================================
QUICK_MODE = False           # True = small fast test run (about 5-10 minutes)

DATA_FOLDER = "data"
SEED = 42                    # same seed = same results every run

BENIGN_FRACTION = 0.02       # keep 2% of normal traffic
ATTACK_FRACTION = 0.20       # keep 20% of attack traffic (and all Web Attacks)
MAX_BENIGN = 150_000         # at most this many normal rows
MAX_PER_ATTACK = 30_000      # at most this many rows of each attack family

TEST_SIZE = 0.20             # 80% train, 20% test
FOLDS = 5                    # K in K-fold cross-validation
REPEATS = 2                  # repeat K-fold twice -> 10 scores per model
CV_ROWS = 50_000             # rows used for cross-validation (for speed)
TUNING_ROWS = 30_000         # rows used for tuning (for speed)
TUNING_TRIES = 10            # settings tried per model when tuning
REVIEW_THRESHOLD = 0.60      # below this confidence, a human analyst checks the flow

if QUICK_MODE:
    MAX_BENIGN, MAX_PER_ATTACK = 6_000, 1_500
    CV_ROWS, TUNING_ROWS, TUNING_TRIES, REPEATS = 4_000, 3_000, 3, 1

# Output folders
TABLES = os.path.join("results", "tables")
FIGURES = os.path.join("results", "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
LOG_FILE = os.path.join("results", "run_log.txt")


# =============================================================================
# Small helpers
# =============================================================================
def log(text=""):
    """Print a message and also save it in results/run_log.txt."""
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {text}" if text else ""
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_table(table, name):
    """Save a table as CSV and print it."""
    table.to_csv(os.path.join(TABLES, name + ".csv"), index=False)
    log("\n" + table.round(4).to_string(index=False) + "\n")


def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, name + ".png"), dpi=150)
    plt.close()


def subset(X, y, n):
    """Take n rows with the same % of each family (stratified)."""
    if len(X) <= n:
        return X, y
    X_small, _, y_small, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=SEED)
    return X_small, y_small


# =============================================================================
# The experiment
# =============================================================================
def main():
    open(LOG_FILE, "w").close()                     # start a fresh log
    start = time.time()
    log(f"Intrusion detection experiment   QUICK_MODE={QUICK_MODE}   SEED={SEED}")

    # -------------------------------------------------------------------------
    log("\n===== STEP 1: Load and clean the data =====")
    # -------------------------------------------------------------------------
    df = load_data(DATA_FOLDER, BENIGN_FRACTION, ATTACK_FRACTION, MAX_BENIGN, MAX_PER_ATTACK, SEED)
    df, cleaning_report = clean_data(df)
    save_table(cleaning_report, "01_cleaning_report")
    features = [c for c in df.columns if c not in ("Label", "Family", "Day")]

    # -------------------------------------------------------------------------
    log("===== STEP 2: Explore the data (charts in results/figures) =====")
    # -------------------------------------------------------------------------
    n_pca = explore(df, features, FIGURES, SEED)
    log(f"Flows per family:\n{df['Family'].value_counts().to_string()}")
    log(f"PCA: {n_pca} of {len(features)} components keep 95% of the information")

    # -------------------------------------------------------------------------
    log("\n===== STEP 3: Train/test split and feature selection =====")
    # -------------------------------------------------------------------------
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Family"])          # family names -> numbers 0..6
    families = list(encoder.classes_)
    benign = families.index("Benign")

    X_train_all, X_test_all, y_train, y_test, port_train, port_test = train_test_split(
        df[features], y, df["Dst Port"].to_numpy(),
        test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # Remove features that are near-copies of another (training data only)
    dropped = find_correlated(X_train_all)
    kept = [f for f in features if f not in dropped]
    X_train, X_test = X_train_all[kept], X_test_all[kept]
    log(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")
    log(f"Correlation filter removed {len(dropped)} features, {len(kept)} left")
    log(f"Removed: {dropped}")

    # -------------------------------------------------------------------------
    log("\n===== STEP 4: Hyperparameter tuning (RandomizedSearchCV) =====")
    # -------------------------------------------------------------------------
    X_tune, y_tune = subset(X_train, y_train, TUNING_ROWS)
    tuned = {}
    rows = []
    for name, model in make_models().items():
        log(f"Tuning {name} ...")
        tuned[name], best, score = tune(name, make_pipeline(model), X_tune, y_tune, TUNING_TRIES)
        rows.append({"model": name, "best macro F1 in tuning": score,
                     "best settings": str({k.replace("model__", ""): v for k, v in best.items()})})
    tuned["Stacking"] = make_stacking(tuned)
    save_table(pd.DataFrame(rows), "04_tuning")

    # -------------------------------------------------------------------------
    log(f"===== STEP 5: {FOLDS}-fold cross-validation x {REPEATS} repeats =====")
    # -------------------------------------------------------------------------
    X_cv, y_cv = subset(X_train, y_train, CV_ROWS)
    X_cv = X_cv.reset_index(drop=True)
    cv = {}
    for name, pipeline in tuned.items():
        log(f"Cross-validating {name} ...")
        cv[name] = cross_validate(name, pipeline, X_cv, y_cv, FOLDS, REPEATS)

    cv_table = pd.DataFrame({"model": list(cv),
                             "macro F1 mean": [np.mean(s) for s in cv.values()],
                             "macro F1 std": [np.std(s) for s in cv.values()]})
    cv_table = cv_table.sort_values("macro F1 mean", ascending=False)
    save_table(cv_table, "05_cross_validation")
    pd.DataFrame(cv).to_csv(os.path.join(TABLES, "05_cv_fold_scores.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.boxplot([cv[m] for m in cv_table["model"]])
    plt.xticks(range(1, len(cv) + 1), cv_table["model"])
    plt.ylabel("Macro F1 per fold")
    plt.title("K-fold cross-validation scores")
    save_figure("05_cross_validation")

    # The best model is chosen by cross-validation, never by the test set
    ranking = list(cv_table["model"])
    best_name = ranking[0]
    log(f"Best model by cross-validation: {best_name}")

    # Statistical test: every model vs. the simple baseline, and best vs. second best
    pairs = [(m, "Logistic Regression") for m in ranking if m != "Logistic Regression"]
    pairs.append((ranking[0], ranking[1]))
    rows = []
    for a, b in pairs:
        diff, p = corrected_t_test(cv[a], cv[b], FOLDS)
        rows.append({"model A": a, "model B": b, "F1 difference (A - B)": diff,
                     "p-value": p, "significant (p < 0.05)": p < 0.05})
    log("Corrected paired t-test on the fold scores:")
    save_table(pd.DataFrame(rows), "05_t_tests")

    # -------------------------------------------------------------------------
    log("===== STEP 6: Train on all training data, then test =====")
    # -------------------------------------------------------------------------
    predictions, confidence, rows = {}, {}, []
    for name, pipeline in tuned.items():
        log(f"Training {name} ...")
        t0 = time.time()
        train(name, pipeline, X_train, y_train)
        seconds = time.time() - t0
        probabilities = pipeline.predict_proba(X_test)
        predictions[name] = probabilities.argmax(axis=1)
        confidence[name] = probabilities.max(axis=1)

        low, high = bootstrap_ci(y_test, predictions[name])
        rows.append({"model": name, **scores(y_test, predictions[name], benign),
                     "macro F1 95% CI": f"{low:.4f} - {high:.4f}", "train time (s)": seconds})

        # Confusion matrix (% of each true family)
        matrix = confusion_matrix(y_test, predictions[name], normalize="true") * 100
        plt.figure(figsize=(9, 7))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=families, yticklabels=families)
        plt.xlabel("Predicted family")
        plt.ylabel("True family")
        plt.title(f"{name} (% of each true family)")
        save_figure(f"06_confusion_{name.replace(' ', '_')}")

    save_table(pd.DataFrame(rows).sort_values("macro F1", ascending=False), "06_test_results")

    best_pred = predictions[best_name]
    report = pd.DataFrame(classification_report(y_test, best_pred, target_names=families,
                                                output_dict=True)).T.loc[families]
    log(f"Results per family for {best_name}:")
    save_table(report.reset_index(names="family"), "06_per_family_best_model")

    # McNemar's test: best model vs. each other model on the test set
    rows = []
    for other in tuned:
        if other != best_name:
            b, c, p = mcnemar_test(y_test, best_pred, predictions[other])
            rows.append({"model A": best_name, "model B": other, "A right, B wrong": b,
                         "A wrong, B right": c, "p-value": p, "significant (p < 0.05)": p < 0.05})
    log("McNemar's test on the test set:")
    save_table(pd.DataFrame(rows), "06_mcnemar")

    # -------------------------------------------------------------------------
    # STEP 7: Ablation study - remove or add one part at a time.
    # Uses the best of Random Forest / XGBoost (Stacking is too slow to retrain many times).
    # -------------------------------------------------------------------------
    single = [m for m in ranking if m in ("Random Forest", "XGBoost")][0]
    log(f"===== STEP 7: Ablation study ({single}) =====")
    base_model = tuned[single].named_steps["model"]

    def variant(description, use_pca=False, balanced=True, columns=kept):
        model = clone(base_model)
        if single == "Random Forest" and not balanced:
            model.set_params(class_weight=None)
        pipeline = train(single, make_pipeline(model, use_pca), X_train_all[columns], y_train, balanced)
        result = scores(y_test, pipeline.predict(X_test_all[columns]), benign)
        log(f"  {description}: macro F1 = {result['macro F1']:.4f}")
        return {"variant": description, **result}

    rows = [variant("Full design"),
            variant("+ PCA", use_pca=True),
            variant("- class weights", balanced=False),
            variant("- correlation filter (all features)", columns=features),
            variant("- Dst Port feature", columns=[f for f in kept if f != "Dst Port"])]
    ablation = pd.DataFrame(rows)
    ablation["change in macro F1"] = ablation["macro F1"] - ablation["macro F1"].iloc[0]
    save_table(ablation, "07_ablation")

    # -------------------------------------------------------------------------
    # STEP 8: Leave-one-attack-out - can the model catch an attack it never saw?
    # Binary task (attack vs normal). Train WITHOUT one family, then test on it.
    # -------------------------------------------------------------------------
    log(f"===== STEP 8: Leave-one-attack-out ({single}) =====")
    rows = []
    for i, family in enumerate(families):
        if i == benign:
            continue
        log(f"  Training without {family} ...")
        model = clone(base_model)
        if single == "XGBoost":
            model.set_params(objective="binary:logistic")       # 2 answers now: attack or normal
        not_this = y_train != i
        pipeline = train(single, make_pipeline(model), X_train[not_this],
                         (y_train[not_this] != benign).astype(int))
        this_family = y_test == i
        rows.append({
            "attack family": family,
            "detected when SEEN in training (%)": 100 * (best_pred[this_family] != benign).mean(),
            "detected when NEVER SEEN (%)": 100 * pipeline.predict(X_test[this_family]).mean(),
            "false alarms on normal traffic (%)": 100 * pipeline.predict(X_test[y_test == benign]).mean()})
    unseen = pd.DataFrame(rows)
    save_table(unseen, "08_leave_one_attack_out")

    unseen.plot(x="attack family", y=["detected when SEEN in training (%)",
                                      "detected when NEVER SEEN (%)"], kind="bar", figsize=(10, 5))
    plt.ylabel("Attacks detected (%)")
    plt.title("Seen vs. never-seen attacks")
    plt.xticks(rotation=20)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    save_figure("08_leave_one_attack_out")

    # -------------------------------------------------------------------------
    log("===== STEP 9: Error analysis (best model) =====")
    # -------------------------------------------------------------------------
    matrix = confusion_matrix(y_test, best_pred)
    rows = []
    for i in range(len(families)):
        for j in range(len(families)):
            if i != j and matrix[i, j] > 0:
                rows.append({"true family": families[i], "predicted as": families[j],
                             "count": matrix[i, j], "% of true family": 100 * matrix[i, j] / matrix[i].sum()})
    log("Most common mistakes:")
    save_table(pd.DataFrame(rows).sort_values("count", ascending=False).head(10), "09_top_mistakes")

    # -------------------------------------------------------------------------
    log("===== STEP 10: Fairness audit (best model) =====")
    # -------------------------------------------------------------------------
    # a) Is every attack family detected equally well?
    rows = []
    for i, family in enumerate(families):
        if i != benign:
            this_family = y_test == i
            rows.append({"attack family": family, "test flows": int(this_family.sum()),
                         "detected as an attack (%)": 100 * (best_pred[this_family] != benign).mean()})
    by_family = pd.DataFrame(rows)
    save_table(by_family, "10_fairness_attack_families")

    # b) Does normal traffic of some services get more false alarms?
    normal = y_test == benign
    audit = pd.DataFrame({"service": [service(p) for p in port_test[normal]],
                          "false alarm": best_pred[normal] != benign})
    by_service = (audit.groupby("service")["false alarm"]
                  .agg(["count", "mean"]).reset_index()
                  .rename(columns={"count": "normal test flows", "mean": "false alarm rate (%)"}))
    by_service["false alarm rate (%)"] *= 100
    save_table(by_service, "10_fairness_services")
    big = by_service[by_service["normal test flows"] >= 30]
    log(f"Detection gap between attack families: "
        f"{by_family['detected as an attack (%)'].max() - by_family['detected as an attack (%)'].min():.1f} points")
    log(f"False alarm gap between services (30+ flows): "
        f"{big['false alarm rate (%)'].max() - big['false alarm rate (%)'].min():.1f} points")

    by_service.plot(x="service", y="false alarm rate (%)", kind="barh", legend=False, figsize=(9, 5))
    plt.xlabel("False alarm rate on normal traffic (%)")
    plt.title(f"{best_name}: false alarms per service")
    save_figure("10_fairness_services")

    # -------------------------------------------------------------------------
    log("===== STEP 11: Feature importance (Random Forest) =====")
    # -------------------------------------------------------------------------
    forest = tuned["Random Forest"].named_steps["model"]
    importance = (pd.DataFrame({"feature": kept, "importance": forest.feature_importances_})
                  .sort_values("importance", ascending=False).head(15))
    save_table(importance, "11_feature_importance")
    importance.iloc[::-1].plot(x="feature", y="importance", kind="barh", legend=False, figsize=(9, 6))
    plt.title("Top 15 features (Random Forest)")
    save_figure("11_feature_importance")

    # Why are attacks missed? Compare missed vs. caught flows of the hardest family
    recall = report.drop("Benign")["recall"]
    hardest = recall.idxmin()
    h = families.index(hardest)
    top = list(importance["feature"].head(8))
    compare = pd.DataFrame({
        "feature": top,
        f"{hardest} caught (mean)": X_test[(y_test == h) & (best_pred == h)][top].mean().values,
        f"{hardest} missed (mean)": X_test[(y_test == h) & (best_pred != h)][top].mean().values,
        "normal traffic (mean)": X_test[y_test == benign][top].mean().values})
    log(f"Hardest family: {hardest} (recall {recall.min():.3f})")
    save_table(compare, "11_missed_vs_caught")

    # -------------------------------------------------------------------------
    log(f"===== STEP 12: Human review (confidence below {REVIEW_THRESHOLD}) =====")
    # Ethics mitigation: unsure predictions go to a security analyst
    # -------------------------------------------------------------------------
    unsure = confidence[best_name] < REVIEW_THRESHOLD
    correct = best_pred == y_test
    review = pd.DataFrame({
        "group": ["Confident (automatic decision)", "Unsure (sent to analyst)"],
        "% of flows": [100 * (~unsure).mean(), 100 * unsure.mean()],
        "accuracy (%)": [100 * correct[~unsure].mean() if (~unsure).any() else np.nan,
                         100 * correct[unsure].mean() if unsure.any() else np.nan]})
    save_table(review, "12_human_review")

    log(f"DONE in {(time.time() - start) / 60:.1f} minutes. Best model: {best_name}")
    log("Tables: results/tables   Figures: results/figures")


if __name__ == "__main__":
    main()
