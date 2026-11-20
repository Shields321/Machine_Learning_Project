"""
data_prep.py
------------
Everything to do with the DATA:
  1. load_data()   -> read the 10 CSV files and keep a random sample
  2. clean_data()  -> fix numbers, remove infinity/missing/duplicates
  3. explore()     -> save charts that describe the data (for the report)
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")                 # save charts to files instead of showing them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Columns that identify machines (not behaviour). If the model sees the
# attacker's IP address it can "cheat", so these are removed.
ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]


def label_to_family(label):
    """Group the 15 labels in the dataset into 7 families.
    Example: 'DoS attacks-Hulk' and 'DoS attacks-Slowloris' both become 'DoS'."""
    label = str(label).lower()
    if label == "label":                 # a header row repeated inside a file
        return None
    if "benign" in label:
        return "Benign"
    if "ddos" in label:                  # check DDoS before DoS
        return "DDoS"
    if "dos" in label:
        return "DoS"
    if "web" in label or "xss" in label or "sql" in label:
        return "Web Attack"
    if "ftp" in label or "ssh" in label:
        return "Brute Force"
    if "infil" in label:                 # the dataset spells it "Infilteration"
        return "Infiltration"
    if "bot" in label:
        return "Bot"
    return None


def load_data(folder, benign_fraction, attack_fraction, max_benign, max_per_attack, seed):
    """Read every CSV file in the folder and combine them into one table.

    The files are too big to open at once, so each file is read 500,000 rows
    at a time and only a random sample of the rows is kept:
      - benign_fraction of the normal traffic (e.g. 2%)
      - attack_fraction of the attack traffic (e.g. 20%)
      - every Web Attack row (there are only a few hundred)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in the '{folder}' folder")

    rng = np.random.default_rng(seed)
    kept = []

    for path in files:
        print(f"  Reading {os.path.basename(path)} ...")
        for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()                 # remove spaces in names
            chunk = chunk.drop(columns=[c for c in ID_COLUMNS if c in chunk.columns])
            chunk["Family"] = chunk["Label"].map(label_to_family)
            chunk = chunk[chunk["Family"].notna()]                    # drop header rows

            # Random sample: give each row a random number between 0 and 1
            r = rng.random(len(chunk))
            benign = (chunk["Family"] == "Benign").to_numpy()
            web = (chunk["Family"] == "Web Attack").to_numpy()
            keep = (benign & (r < benign_fraction)) | (~benign & (r < attack_fraction)) | web

            sample = chunk[keep].copy()
            sample["Day"] = os.path.basename(path).split("_")[0]      # e.g. 'Wednesday-14-02-2018'
            kept.append(sample)

    df = pd.concat(kept, ignore_index=True)

    # Cap each family so one family cannot take over the data
    capped = []
    for family, group in df.groupby("Family"):
        limit = max_benign if family == "Benign" else max_per_attack
        capped.append(group.sample(n=min(limit, len(group)), random_state=seed))
    return pd.concat(capped, ignore_index=True)


def clean_data(df):
    """Clean the data. Returns the clean table and a small report that says
    how many rows each step removed (for the Data Statement in the report)."""
    features = [c for c in df.columns if c not in ("Label", "Family", "Day")]
    report = [("Rows after sampling", len(df))]

    # 1. Make every feature a number (text that is not a number becomes NaN)
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # 2. Infinity appears when a flow lasts 0 seconds (divide by zero).
    #    Turn it into NaN, then remove every row with a missing value.
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=features)
    report.append(("Removed: missing or infinite values", before - len(df)))

    # 3. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=features + ["Family"])
    report.append(("Removed: duplicate rows", before - len(df)))

    # 4. Remove columns that have the same value in every row
    constant = [c for c in features if df[c].nunique() <= 1]
    df = df.drop(columns=constant)
    report.append(("Removed: constant columns", len(constant)))

    report.append(("Rows after cleaning", len(df)))
    report.append(("Features after cleaning", len(features) - len(constant)))
    return df.reset_index(drop=True), pd.DataFrame(report, columns=["Step", "Count"])


def find_correlated(X_train, threshold=0.95):
    """Find features that are almost a copy of another one (correlation above
    the threshold). Only the training data is used, so the test set stays unseen."""
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))   # each pair once
    return [col for col in upper.columns if (upper[col] > threshold).any()]


def explore(df, features, folder, seed):
    """Save 4 charts that describe the data. Returns how many PCA components
    are needed to keep 95% of the information."""

    # 1. How many flows per family (shows the class imbalance)
    df["Family"].value_counts().sort_values().plot(kind="barh", logx=True, color="steelblue")
    plt.xlabel("Number of flows (log scale)")
    plt.title("Flows per family")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "eda_class_counts.png"), dpi=150)
    plt.close()

    # 2. Which attacks happen on which day
    per_day = pd.crosstab(df["Day"], df["Family"])
    dates = pd.to_datetime(per_day.index.str[-10:], format="%d-%m-%Y")   # sort days by date
    per_day = per_day.iloc[np.argsort(dates)]
    per_day.to_csv(os.path.join(folder, "eda_attacks_per_day.csv"))
    plt.figure(figsize=(10, 5))
    sns.heatmap(per_day, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Flows per day and family (in our sample)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "eda_attacks_per_day.png"), dpi=150)
    plt.close()

    # Scale the features for PCA (log first, because the values are very skewed)
    X = df[features].to_numpy()
    X = StandardScaler().fit_transform(np.sign(X) * np.log1p(np.abs(X)))

    # 3. PCA: how many components keep 95% of the information
    pca = PCA(random_state=seed).fit(X)
    total = np.cumsum(pca.explained_variance_ratio_)
    n_needed = int(np.argmax(total >= 0.95) + 1)
    plt.plot(range(1, len(total) + 1), total, "o-", markersize=3)
    plt.axhline(0.95, color="red", linestyle="--")
    plt.axvline(n_needed, color="red", linestyle="--")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Information kept (explained variance)")
    plt.title(f"PCA: {n_needed} of {len(features)} components keep 95%")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "eda_pca_variance.png"), dpi=150)
    plt.close()

    # 4. PCA picture: the data squashed to 2 dimensions (5,000 random flows)
    rows = np.random.default_rng(seed).choice(len(df), size=min(5000, len(df)), replace=False)
    points = PCA(n_components=2, random_state=seed).fit_transform(X[rows])
    families = df["Family"].to_numpy()[rows]
    plt.figure(figsize=(8, 6))
    for family in sorted(set(families)):
        plt.scatter(points[families == family, 0], points[families == family, 1],
                    s=5, alpha=0.6, label=family)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Flows on the first 2 PCA components")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "eda_pca_2d.png"), dpi=150)
    plt.close()

    return n_needed
