# The codes have been tested with python 3.7.9

# import all neccessary libraries
import mlflow.models.signature
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, date
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import confusion_matrix, roc_auc_score
from hyperopt import hp
from data import preprocess_training_data
import mlflow


# feature engineering
def calculate_duration(lastVisit):
    """
    Calculate the duration in days between the last visit date and today.

    Args:
        lastVisit (str): A string representing the last visit date in the format "%m/%d/%Y %H:%M".

    Returns:
        int: The number of days between the last visit and today.
             Returns 0 if lastVisit is None or an empty string.
    """
    if lastVisit:
        lastVisit = datetime.strptime(lastVisit, "%m/%d/%Y %H:%M").date()
        today = date.today()
        return (today - lastVisit).days
    else:
        return 0


def optimize(
    trials,
    df_train_X,
    df_train_y,
    df_test_X,
    df_test_y,
    random_state=1998,
):
    """
    This is the optimization function that given a space (space here) of
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page:
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 1),
        "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        "max_depth": hp.choice("max_depth", np.arange(1, 14, dtype=int)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "gamma": hp.quniform("gamma", 0.5, 1, 0.05),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
        "eval_metric": "auc",
        "objective": "binary:logistic",
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        "nthread": 4,
        "booster": "gbtree",
        "tree_method": "exact",
        "silent": 1,
        "seed": random_state,
    }

    def score(params):
        print("Training with params: ")
        print(params)
        num_round = int(params["n_estimators"])
        del params["n_estimators"]
        dtrain = xgb.DMatrix(df_train_X, label=df_train_y)
        dvalid = xgb.DMatrix(df_test_X, label=df_test_y)
        watchlist = [(dvalid, "eval"), (dtrain, "train")]
        gbm_model = xgb.train(
            params, dtrain, num_round, evals=watchlist, verbose_eval=True
        )
        predictions = gbm_model.predict(
            dvalid, ntree_limit=gbm_model.best_iteration + 1
        )
        score = roc_auc_score(df_test_y, predictions)

        print("\tScore {0}\n\n".format(score))

        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(
        score,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=250,
    )
    return best


def train_model(df_train_X, df_train_y, df_test_X, df_test_y):
    mlflow.set_experiment("XGBoost Classifier")

    with mlflow.start_run():
        best_hyperparams = optimize(
            Trials(), df_train_X, df_train_y, df_test_X, df_test_y
        )
        print("The best hyperparameters are: ", "\n")
        print(best_hyperparams)

        # Log best hyperparameters
        mlflow.log_params(best_hyperparams)

        # train the best model
        best_params = best_hyperparams
        best_params.update({"eval_metric": "auc"})

        model = XGBClassifier(silent=True)
        model.set_params(**best_params)
        dtrain = xgb.DMatrix(df_train_X, label=df_train_y)
        dvalid = xgb.DMatrix(df_test_X, label=df_test_y)
        watchlist = [(dvalid, "eval"), (dtrain, "train")]
        model = xgb.train(best_params, dtrain, evals=watchlist, verbose_eval=True)

        # Log the model
        mlflow.xgboost.log_model(model, "xgboost_model")

    return model


def evaluate_model(model, df_test_X, df_test):
    with mlflow.start_run(run_name="model_evaluation"):
        test_set_prediction = model.predict(xgb.DMatrix(df_test_X))
        foo1 = pd.DataFrame({"predict": test_set_prediction}, index=df_test_X.index)
        df_test.loc[:, "predict"] = foo1["predict"]
        df_test.sort_values("predict", ascending=False, inplace=True)

        # 10% of the popultion is selected for target campaigns
        p = 0.1
        df_test.loc[:, "predict_label"] = 0
        cutoff_val = df_test.iloc[int(p * len(df_test)),]["predict"]
        df_test.loc[df_test.predict >= cutoff_val, "predict_label"] = 1

        df_test.predict_label.value_counts().sort_index()

        a = confusion_matrix(
            df_test["product02"], df_test["predict_label"], labels=[0, 1]
        )

        print(a)

        print("\n\n")
        actual_postive = len(df_test.loc[df_test.loc[:, "product02"] == 1, :])
        total_num = len(df_test)
        predicted_and_actual_postive = len(
            df_test.loc[
                (df_test.loc[:, "product02"] == 1) & (df_test.loc[:, "predict_label"]),
                :,
            ]
        )
        predicted_postive = sum(df_test["predict_label"])
        model_lift = (predicted_and_actual_postive / predicted_postive) / (
            actual_postive / total_num
        )

        print(
            f"{actual_postive} positives out of total {total_num} customers, which is "
            + f"{ (100 * (actual_postive) / total_num):.02f} %"
        )
        print(
            f"{predicted_and_actual_postive} actual positives out of total {predicted_postive} "
            + "predicted positives, which is "
            + f"{ (100 * (predicted_and_actual_postive) / predicted_postive):.02f} %"
        )

        print(f"MODEL LIFT = {model_lift:.03f}")

        print(
            f"Precision: { (100 * predicted_and_actual_postive / predicted_postive):.02f} %"
        )
        print(
            f"Recall   : { (100 * predicted_and_actual_postive / actual_postive):.02f} %"
        )

        print(
            f"By contacting {(100 * predicted_postive / total_num):.02f} % of the base, you hit "
            + f"{(100 * predicted_and_actual_postive / actual_postive):.02f} % of your target group"
        )

        # log the metrics reported
        mlflow.log_metric("model_lift", model_lift)
        mlflow.log_metric(
            "precision", 100 * predicted_and_actual_postive / predicted_postive
        )
        mlflow.log_metric("recall", 100 * predicted_and_actual_postive / actual_postive)
        mlflow.log_metric(
            "target_hit_rate", 100 * predicted_and_actual_postive / actual_postive
        )

        model.save_model("best_model_xgb.json")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=mlflow.models.infer_signature(df_test_X, test_set_prediction),
            registered_model_name="sk-learn-logistic-regression-model",
        )


if __name__ == "__main__":
    raw_data = pd.read_csv("telecom.csv")
    df_train_X, df_train_y, df_test_X, df_test_y, df_train, df_test = (
        preprocess_training_data(raw_data)
    )
    model = train_model(df_train_X, df_train_y, df_test_X, df_test_y)
    evaluate_model(model, df_test_X, df_test)
