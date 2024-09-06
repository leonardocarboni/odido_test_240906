import pandas as pd
import numpy as np
import xgboost as xgb
from model import calculate_duration
import mlflow
from utils import send_email


def main():
    with mlflow.start_run():
        df_test = pd.read_csv("telecom_test.csv")
        # Prepare the scoring data
        # fill missing values
        df_test[["gender", "house_type"]] = df_test[["gender", "house_type"]].fillna(
            value="missing"
        )
        df_test["income"] = df_test["income"].fillna((df_test["income"].mean()))
        df_test["var1"] = df_test["var1"].fillna((df_test["var1"].mean()))
        df_test["age"] = df_test["age"].fillna((df_test["age"].mean()))

        # log transformation
        df_test["income"] = np.log(df_test["income"])

        df_test["duration"] = df_test["lastVisit"].apply(calculate_duration)
        df_test = df_test.drop(["lastVisit"], axis=1)

        df_test_onehot = pd.get_dummies(df_test, dummy_na=True)
        df_test_onehot = df_test_onehot.drop(["subscriber"], axis=1)

        # load the saved model and use it to predict
        dtest = xgb.DMatrix(df_test_onehot)
        model_xgb_pred = xgb.Booster()
        model_xgb_pred.load_model("best_model_xgb.json")
        print(model_xgb_pred)
        scores = model_xgb_pred.predict(dtest)

        # log prediction metrics
        mlflow.log_metric("mean_score", np.mean(scores))
        mlflow.log_metric("median_score", np.median(scores))
        mlflow.log_metric("max_score", np.max(scores))
        mlflow.log_metric("min_score", np.min(scores))

        # save the output in csv
        df_test["score"] = scores
        output = df_test[["subscriber", "score"]]
        output.to_csv("output.csv", index=False)

        # log the output
        mlflow.log_artifact("output.csv")

        # send email to stakeholder # if not with ci/cd pipeline
        send_email(np.mean(scores))


if __name__ == "__main__":
    main()
