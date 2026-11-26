"""
data_prep.py
------------
Everything to do with the DATA. main.py uses these functions:

  load_data()        -> read the 10 CSV files and keep a random sample of the rows
  clean_data()       -> fix numbers, remove infinity / missing / duplicate rows
  find_correlated()  -> find features that are almost a copy of another feature

Small helper used by load_data():
  label_to_family()  -> turn the 15 labels in the dataset into 7 families

(The table of how many rows were removed is made in figures_and_tables.py.)
"""

import glob        # finds all the CSV files in a folder
import os          # builds file paths and file names

import numpy as np
import pandas as pd


# Columns that identify machines (not behaviour). If the model sees the
# attacker's IP address it can "cheat", so these columns are removed.
ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]


def label_to_family(label):
    """Group the 15 labels in the dataset into 7 families.
    Example: 'DoS attacks-Hulk' and 'DoS attacks-Slowloris' both become 'DoS'."""
    label = str(label).lower()           # lower case, so 'DoS' and 'dos' are the same

    if label == "label":                 # a header row repeated inside a file -> not a real row
        return None
    if "benign" in label:
        return "Benign"
    if "ddos" in label:                  # check DDoS before DoS ('ddos' also contains 'dos')
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
    return None                          # unknown label


def load_data(folder, benign_fraction, attack_fraction, max_benign, max_per_attack, seed):
    """Read every CSV file in the folder and combine them into one table.

    The files are too big to open at once, so each file is read 500,000 rows
    at a time and only a random sample of the rows is kept:
      - benign_fraction of the normal traffic (e.g. 2%)
      - attack_fraction of the attack traffic (e.g. 20%)
      - every Web Attack row (there are only a few hundred)
    """
    # Find every CSV file in the folder (sorted, so the order is always the same)
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError("No CSV files found in the '" + folder + "' folder")

    rng = np.random.default_rng(seed)      # random number generator (same seed = same sample)
    kept = []                              # the sampled rows from every file go in this list

    for path in files:
        file_name = os.path.basename(path)             # e.g. '02-14-2018.csv'
        print("  Reading", file_name, "...")

        # The day is the file name without '.csv' (and without anything after a '_')
        day = os.path.splitext(file_name)[0]           # '02-14-2018.csv' -> '02-14-2018'
        day = day.split("_")[0]                        # 'Wednesday-14-02-2018_TrafficForML' -> 'Wednesday-14-02-2018'

        # Read the file 500,000 rows at a time (each piece is called a "chunk")
        for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()      # remove spaces around the column names

            # Remove the ID columns that are in this file
            to_remove = []
            for col in ID_COLUMNS:
                if col in chunk.columns:
                    to_remove.append(col)
            chunk = chunk.drop(columns=to_remove)

            # Add the family of every row. Header rows repeated inside the file get None.
            chunk["Family"] = chunk["Label"].apply(label_to_family)
            chunk = chunk[chunk["Family"].notna()]         # keep only the rows that have a family

            # Random sample: give every row a random number between 0 and 1
            r = rng.random(len(chunk))
            benign = (chunk["Family"] == "Benign").to_numpy()      # True for normal rows
            web = (chunk["Family"] == "Web Attack").to_numpy()     # True for Web Attack rows

            keep_benign = benign & (r < benign_fraction)    # e.g. 2% of the normal rows
            keep_attack = ~benign & (r < attack_fraction)   # e.g. 20% of the attack rows (~ means "not")
            keep = keep_benign | keep_attack | web          # | means "or": keep the row if any is True

            sample = chunk[keep].copy()
            sample["Day"] = day
            kept.append(sample)

    df = pd.concat(kept, ignore_index=True)    # put all the samples into one table

    # Cap each family, so one big family cannot take over the data
    capped = []
    for family in sorted(df["Family"].unique()):
        group = df[df["Family"] == family]                 # all the rows of this family
        if family == "Benign":
            limit = max_benign
        else:
            limit = max_per_attack
        n = min(limit, len(group))                         # a small family keeps all its rows
        capped.append(group.sample(n=n, random_state=seed))
    return pd.concat(capped, ignore_index=True)


def clean_data(df):
    """Clean the data. Returns the clean table and a dictionary that says
    how many rows each step removed (figures_and_tables.py turns it into a table)."""
    # The features are every column except these 3
    features = []
    for col in df.columns:
        if col not in ["Label", "Family", "Day"]:
            features.append(col)

    counts = {}                                    # step name -> count
    counts["Rows after sampling"] = len(df)

    # 1. Make every feature a number. Text that is not a number becomes NaN (missing).
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype("float32")        # float32 uses half the memory of float64

    # 2. Infinity appears when a flow lasts 0 seconds (divide by zero).
    #    Turn infinity into NaN, then remove every row with a missing value.
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=features)
    counts["Removed: missing or infinite values"] = before - len(df)

    # 3. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=features + ["Family"])
    counts["Removed: duplicate rows"] = before - len(df)

    # 4. Remove columns that have the same value in every row (they tell the model nothing)
    constant = []
    for col in features:
        if df[col].nunique() <= 1:
            constant.append(col)
    df = df.drop(columns=constant)
    counts["Removed: constant columns"] = len(constant)

    counts["Rows after cleaning"] = len(df)
    counts["Features after cleaning"] = len(features) - len(constant)

    df = df.reset_index(drop=True)          # number the rows 0, 1, 2, ... again
    return df, counts


def find_correlated(X_train, threshold=0.95):
    """Find features that are almost a copy of an earlier feature
    (correlation above the threshold). Only the training data is used,
    so the test set stays unseen. Returns the names of the features to drop."""
    corr = X_train.corr().abs()          # correlation of every pair (0 = unrelated, 1 = copies)
    values = corr.to_numpy()             # the same table as a numpy array
    columns = list(corr.columns)

    dropped = []
    for j in range(len(columns)):        # for every feature ...
        for i in range(j):               # ... compare it with every feature that comes BEFORE it
            if values[i, j] > threshold:
                dropped.append(columns[j])   # it is a near-copy of an earlier feature: drop it
                break                        # no need to check more pairs for this feature
    return dropped
