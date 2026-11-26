"""
figures_and_tables.py
---------------------
All the FIGURES (.png charts) and TABLES (.csv files) of the project.

main.py trains and tests the models and works out the results. It then
calls the functions in this file to turn those results into tables and charts.
Each function is named after the file it makes, for example:
  table_06_test_results()   makes  results/tables/06_test_results.csv
  figure_06_confusion()     makes  results/figures/06_confusion_<model name>.png

The only charts are the confusion matrices (Step 6). Everything else is a table.

EVERY FILE THAT IS MADE (in the order main.py makes them)

  results/run_log.txt                            everything that was printed during the run

  STEP 1 - Load and clean the data
    tables/01_cleaning_report.csv                how many rows / columns each cleaning step removed

  STEP 4 - Hyperparameter tuning
    tables/04_tuning.csv                         best settings and best tuning score of each model

  STEP 5 - K-fold cross-validation
    tables/05_cross_validation.csv               mean and spread of the fold scores of each model (best first),
                                                 and the difference vs. the Logistic Regression baseline
    tables/05_cv_fold_scores.csv                 every single fold score

  STEP 6 - Final test results
    figures/06_confusion_<model name>.png        confusion matrix of each model (5 charts)
    tables/06_test_results.csv                   accuracy, macro F1, detection rate, false alarms, train time
    tables/06_per_family_best_model.csv          precision, recall and F1 per family for the best model

  STEP 9 - Error analysis
    tables/09_top_mistakes.csv                   the 10 most common mistakes of the best model

  STEP 11 - Feature importance (the models use PCA, so these are PCA components)
    tables/11_feature_importance.csv             the 15 most important PCA components (Random Forest)
    tables/11_missed_vs_caught.csv               hardest family: missed vs. caught flows on the top 8 components

  (Steps 2, 7, 8, 10, 12 and 13 were removed in this reduced version, so they make no files.)
"""

import os
import time

import matplotlib
matplotlib.use("Agg")      # save charts to files instead of showing them on screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Where everything is saved (the folders are created if they do not exist yet)
TABLES = os.path.join("results", "tables")
FIGURES = os.path.join("results", "figures")
LOG_FILE = os.path.join("results", "run_log.txt")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


# =============================================================================
# Helpers used by every function below
# =============================================================================
def start_log():
    """Empty the log file at the start of a run."""
    open(LOG_FILE, "w").close()


def log(text=""):
    """Print a message and also save it in results/run_log.txt."""
    if text == "":
        line = ""
    else:
        stamp = time.strftime("%H:%M:%S")      # the time now, e.g. 14:05:32
        line = "[" + stamp + "] " + text
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:   # "a" = add to the end of the file
        f.write(line + "\n")


def save_table(table, name):
    """Save a table in results/tables as a CSV file, and print it."""
    table.to_csv(os.path.join(TABLES, name + ".csv"), index=False)
    log("\n" + table.round(4).to_string(index=False) + "\n")


def save_figure(name):
    """Save the current chart in results/figures as a PNG file."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, name + ".png"), dpi=150)
    plt.close()


# =============================================================================
# STEP 1 - Load and clean the data
# =============================================================================
def table_01_cleaning_report(counts):
    """tables/01_cleaning_report.csv
    How many rows each cleaning step removed (for the Data Statement in the report).
    counts is the dictionary from data_prep.clean_data(): step name -> count."""
    steps = []
    numbers = []
    for step in counts:
        steps.append(step)
        numbers.append(counts[step])
    table = pd.DataFrame({"Step": steps, "Count": numbers})
    save_table(table, "01_cleaning_report")


# =============================================================================
# STEP 4 - Hyperparameter tuning
# =============================================================================
def table_04_tuning(best_scores, best_settings):
    """tables/04_tuning.csv
    For each model: the best macro F1 found while tuning, and the settings
    that gave it. best_scores and best_settings are dictionaries: model name -> value."""
    rows = []
    for name in best_scores:
        # Remove "model__" from the setting names so the table is easier to read
        short_settings = {}
        for setting in best_settings[name]:
            short_name = setting.replace("model__", "")
            short_settings[short_name] = best_settings[name][setting]

        rows.append({"model": name,
                     "best macro F1 in tuning": best_scores[name],
                     "best settings": str(short_settings)})
    save_table(pd.DataFrame(rows), "04_tuning")


# =============================================================================
# STEP 5 - K-fold cross-validation
# =============================================================================
def table_05_cross_validation(cv):
    """tables/05_cross_validation.csv
    For each model: the mean and the spread (std) of its fold scores, best
    model first, and how much better or worse it is than the simple baseline
    (Logistic Regression). cv is a dictionary: model name -> list of fold scores.
    Returns the table, because main.py uses its order to rank the models."""
    names = []
    means = []
    stds = []
    for name in cv:
        names.append(name)
        means.append(np.mean(cv[name]))
        stds.append(np.std(cv[name]))
    table = pd.DataFrame({"model": names, "macro F1 mean": means, "macro F1 std": stds})
    table = table.sort_values("macro F1 mean", ascending=False)

    # Compare every model with the simple baseline (Logistic Regression)
    baseline = np.mean(cv["Logistic Regression"])
    table["difference vs Logistic Regression"] = table["macro F1 mean"] - baseline
    log("Cross-validation results compared with the baseline:")
    save_table(table, "05_cross_validation")
    return table


def table_05_cv_fold_scores(cv):
    """tables/05_cv_fold_scores.csv
    Every single fold score: one column per model, one row per fold."""
    table = pd.DataFrame(cv)
    table.to_csv(os.path.join(TABLES, "05_cv_fold_scores.csv"), index=False)


# =============================================================================
# STEP 6 - Final test results
# =============================================================================
def figure_06_confusion(y_test, prediction, families, name):
    """figures/06_confusion_<model name>.png   (one chart per model)
    Confusion matrix in %: row = true family, column = predicted family.
    Each row adds up to 100%. The diagonal = correct, the rest = mistakes."""
    matrix = confusion_matrix(y_test, prediction, normalize="true") * 100
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=families, yticklabels=families)
    plt.xlabel("Predicted family")
    plt.ylabel("True family")
    plt.title(f"{name} (% of each true family)")
    save_figure("06_confusion_" + name.replace(" ", "_"))


def table_06_test_results(test_scores, train_seconds):
    """tables/06_test_results.csv
    The main results table: one row per model, best macro F1 first.
    test_scores:   model name -> dictionary from models.scores()
    train_seconds: model name -> training time"""
    rows = []
    for name in test_scores:
        result = test_scores[name]
        rows.append({"model": name,
                     "accuracy": result["accuracy"],
                     "macro F1": result["macro F1"],
                     "detection rate": result["detection rate"],
                     "false alarm rate": result["false alarm rate"],
                     "train time (s)": train_seconds[name]})
    table = pd.DataFrame(rows)
    table = table.sort_values("macro F1", ascending=False)
    save_table(table, "06_test_results")


def table_06_per_family_best_model(report, families, best_name):
    """tables/06_per_family_best_model.csv
    Precision, recall and F1 for each family, for the best model.
    report is the dictionary from classification_report(..., output_dict=True)."""
    rows = []
    for family in families:
        rows.append({"family": family,
                     "precision": report[family]["precision"],
                     "recall": report[family]["recall"],
                     "f1-score": report[family]["f1-score"],
                     "support": report[family]["support"]})     # support = number of test flows
    log(f"Results per family for {best_name}:")
    save_table(pd.DataFrame(rows), "06_per_family_best_model")


# =============================================================================
# STEP 9 - Error analysis
# =============================================================================
def table_09_top_mistakes(matrix, families):
    """tables/09_top_mistakes.csv
    The 10 most common mistakes of the best model, e.g. "Infiltration
    predicted as Benign". matrix is the confusion matrix (counts, not %)."""
    rows = []
    for i in range(len(families)):                    # i = true family
        for j in range(len(families)):                # j = predicted family
            if i != j and matrix[i, j] > 0:           # a mistake that happened at least once
                rows.append({"true family": families[i],
                             "predicted as": families[j],
                             "count": matrix[i, j],
                             "% of true family": 100 * matrix[i, j] / matrix[i].sum()})
    table = pd.DataFrame(rows)
    table = table.sort_values("count", ascending=False)
    log("Most common mistakes:")
    save_table(table.head(10), "09_top_mistakes")


# =============================================================================
# STEP 11 - Feature importance
# =============================================================================
def table_11_feature_importance(components, importances):
    """tables/11_feature_importance.csv
    The 15 most important PCA components (PC1, PC2, ...) of the Random
    Forest, most important first. (The models are trained on PCA data, so
    they see components, not the original features.)
    Returns the table, because main.py takes the top 8 components from it."""
    table = pd.DataFrame({"component": components, "importance": importances})
    table = table.sort_values("importance", ascending=False)
    table = table.head(15)                            # the top 15
    save_table(table, "11_feature_importance")
    return table


def table_11_missed_vs_caught(X_test, y_test, best_pred, families, benign, hardest, top):
    """tables/11_missed_vs_caught.csv
    Why are attacks missed? For the hardest family, the average value of the
    top PCA components for: flows that were caught, flows that were missed,
    and normal traffic. If the missed flows look like normal traffic, that
    explains the misses. X_test here is the test flows as PCA components."""
    h = families.index(hardest)                       # the number of the hardest family
    caught = (y_test == h) & (best_pred == h)
    missed = (y_test == h) & (best_pred != h)
    normal = y_test == benign
    table = pd.DataFrame({
        "component": top,
        hardest + " caught (mean)": X_test[caught][top].mean().values,
        hardest + " missed (mean)": X_test[missed][top].mean().values,
        "normal traffic (mean)": X_test[normal][top].mean().values})
    save_table(table, "11_missed_vs_caught")
