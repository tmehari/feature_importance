import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

LVH_FILE_BIG = "./data/Feature-Importance-Feature-Table-1.xlsx"
LVH_FILE_SMALL = "./data/Feature-Importance-Feature-Table-2.xlsx"
LBBB_FILE = "./data/Feature-Importance-Feature-Tables-LBBB.xlsx"
RBBB_FILE = "./data/Feature-Importance-Feature-Tables-RBBB.xlsx"
AVBLOCK_FILE = "./data/Feature-Importance-Feature-Tables-AVBlock.xlsx"
MULTI_CLASS_TABLE_FILE = "./data/Feature-Importance-Feature-Table-CinC.xlsx"
FILE_MAPPING = dict(
    zip(["avblock", "lbbb", "rbbb"], [AVBLOCK_FILE, LBBB_FILE, RBBB_FILE])
)
PTB_INFO_FILE = "./data/ptbxl_database.csv"


def replace_nans(df):
    cols = list(df.columns)
    cols.remove("label")
    # replace NaNs with median of respective column
    for col in cols:
        df[col].fillna((df[col].median()), inplace=True)
    return df


def clean_df(df):
    if "Label" in df.columns:
        df["label"] = df["Label"]
        df = df.drop(["Label"], axis=1)
    df = replace_nans(df)
    return df


def get_stratified_multiclass_table():
    df = pd.read_excel(MULTI_CLASS_TABLE_FILE)
    stratify(df)
    df = clean_df(df)
    return df


def get_split(df, patho):
    train_set = df[df["strat_fold"] < 10]
    test_set = df[df["strat_fold"] == 10]

    drop_columns = ["strat_fold", "label", "ecg_id"]
    X_train = train_set.drop(drop_columns, axis=1)
    features = X_train.columns
    X_train = X_train.values.astype("float")
    X_test = test_set.drop(drop_columns, axis=1).values.astype("float")

    y_train = np.where(train_set["label"].str.lower() == patho.lower(), 1, 0)
    y_test = np.where(test_set["label"].str.lower() == patho.lower(), 1, 0)

    train_indices = np.random.choice(len(X_train), size=int(len(X_train)), replace=None)
    test_indices = np.random.choice(len(X_test), size=int(len(X_test)), replace=None)

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    noncategorical_feature_ids = [np.where(features==feat)[0][0] for feat in features if 'Morph' not in feat]
    # Scale data
    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    X_train[:, noncategorical_feature_ids] = scaler.fit_transform(X_train[:, noncategorical_feature_ids])
    X_test[:, noncategorical_feature_ids] = scaler.transform(X_test[:, noncategorical_feature_ids])

    return X_train, X_test, y_train, y_test, features


def stratify(df):
    def partition(indices, n):
        random.Random(42).shuffle(indices)
        return [indices[i::n] for i in range(n)]

    indices = np.arange(len(df))
    strat_fold_indices = partition(indices, 10)
    strat_folds = np.zeros(len(indices), dtype=int)
    for i, fold in enumerate(strat_fold_indices):
        strat_folds[fold] = i + 1
    df["strat_fold"] = strat_folds


def load_data(patho, reduced):
    tag = "binary_"
    if patho.lower() == "lvh":
        if reduced:
            df = pd.read_excel(LVH_FILE_SMALL)
            tag = "reduced_binary_"
        else:
            df = pd.read_excel(LVH_FILE_BIG)
    elif patho.lower() in ["avblock", "lbbb", "rbbb"]:
        df = pd.read_excel(FILE_MAPPING[patho.lower()])
    else:
        raise ("unknown pathology")

    df = add_strat_folds_from_ptb_info(df)
    return df, tag


def add_strat_folds_from_ptb_info(df):
    ptb_info = pd.read_csv(PTB_INFO_FILE)
    df = clean_df(df)
    df_ecg_ids = set(df.ecg_id)
    # pdb.set_trace()
    intersection = ptb_info[ptb_info["ecg_id"].apply(lambda x: x in df_ecg_ids)]
    assert list(intersection.ecg_id) == list(df.ecg_id)
    df["strat_fold"] = intersection["strat_fold"].values
    return df
