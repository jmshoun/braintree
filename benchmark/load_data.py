import json

import pandas as pd
import numpy as np

import braintree


def abalone_transform(df):
    df["gender"] = [1 if gender == "M" else 0 for gender in df["gender"]]
    return df


def sgemm_transform(df):
    df["log_run"] = np.mean(np.log(df.iloc[:, 14:].values), axis=1)
    del df["Run1 (ms)"]
    del df["Run2 (ms)"]
    del df["Run3 (ms)"]
    del df["Run4 (ms)"]
    return df


def facebook_transform(df):
    df["num_new_comments"] = np.log(df["num_new_comments"] + 1)
    return df


def blog_feedback_transform(df):
    df["num_comments_next_24h"] = np.log(df["num_comments_next_24h"] + 1)
    return df


TRANSFORMS = {"abalone": abalone_transform,
              "sgemm": sgemm_transform,
              "facebook": facebook_transform,
              "blog-feedback": blog_feedback_transform}


def load_data(name):
    with open(f"benchmark/data/{name}-meta.json") as meta_file:
        metadata = json.load(meta_file)
    try:
        data = load_csv(f"benchmark/data/{name}.csv", metadata)
        return data.shuffle().split(0.7)
    except FileNotFoundError:
        train = load_csv(f"benchmark/data/{name}-train.csv", metadata)
        test = load_csv(f"benchmark/data/{name}-test.csv", metadata)
        return train, test


def load_csv(filename, meta):
    if meta["name"] in TRANSFORMS:
        pandas_header = 0 if meta["header"] else None
        df = pd.read_csv(filename, delimiter=meta.get("delimiter", ","), header=pandas_header)
        if "features" in meta:
            df.columns = meta["features"]
        df = TRANSFORMS[meta["name"]](df)
        return braintree.BrainTreeData.from_data_frame(df, meta["responses"])
    else:
        return braintree.BrainTreeData.from_csv(filename, meta["responses"], meta.get("features"),
                                                meta["header"], meta.get("delimiter", ","))
