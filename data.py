from datetime import datetime, date
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def calculate_duration(lastVisit):
    if lastVisit:
        lastVisit = datetime.strptime(lastVisit, "%m/%d/%Y %H:%M").date()
        today = date.today()
        return (today - lastVisit).days
    else:
        return 0


def preprocess_training_data(raw_data):
    df = raw_data.copy()

    # fill missing values
    df[["gender", "house_type"]] = df[["gender", "house_type"]].fillna(value="missing")
    df["income"] = df["income"].fillna((df["income"].mean()))
    df["var1"] = df["var1"].fillna((df["var1"].mean()))
    df["age"] = df["age"].fillna((df["age"].mean()))

    # log transformation
    df["income"] = np.log(df["income"])

    # prepare target variable
    df.loc[df.product02 == "Nee", "product02"] = 0
    df.loc[df.product02 == "Ja", "product02"] = 1
    df["product02"] = df.product02.astype(int)

    df["duration"] = df["lastVisit"].apply(calculate_duration)
    df = df.drop(["subscriber", "lastVisit"], axis=1)

    # one-hot encoding the categorical features
    df_X = df.loc[:, df.columns != "product02"]
    df_X_onehot = pd.get_dummies(df_X, dummy_na=True)
    df_onehot = pd.concat([df_X_onehot, df[["product02"]]], axis=1)

    strat = df["product02"]
    df_train, df_test = train_test_split(
        df_onehot, test_size=0.2, random_state=12345, stratify=strat
    )

    df_train_X = df_train.loc[:, df_train.columns != "product02"]
    df_train_y = df_train["product02"]
    df_test_X = df_test.loc[:, df_test.columns != "product02"]
    df_test_y = df_test["product02"]

    return df_train_X, df_train_y, df_test_X, df_test_y, df_train, df_test
