import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

import config
from common.utils import set_seed
from model_dispatcher import (
    CustomModel,
    DecisionTreeModel,
    LogisticRegressionModel,
    XGBoost,
    LightGBM,
    CatBoost,
    Lasso,
)

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


def run(fold: int, model: CustomModel) -> Tuple[float, np.ndarray]:
    # load the full training data with folds
    df = pd.read_csv(config.TRAIN_FOLDS)
    df_test = pd.read_csv(config.PREPROCESSED_TEST_DATA)

    # let drop columns that doesn't hold information
    df.drop(columns=["id"], inplace=True)
    df_test.drop(columns=["id"], inplace=True)

    # all columns are features except target and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold")]
    cat_features = []
    num_features = features

    # initialize model
    custom_model = model(
        df, fold, config.TARGET, cat_features, num_features, test=df_test
    )

    # encode all features
    custom_model.encode()

    # fit model on training data
    custom_model.fit()

    # predict on validation data and get rmse score
    auc = custom_model.predict_and_score()

    # print rmse
    print(f"Fold = {fold}, AUC = {auc}")

    # predict on test
    preds = custom_model.predict_test()

    return (auc, preds)


def parse_model(code: str) -> CustomModel:
    model = None
    if code == "lr":
        model = LogisticRegressionModel
    elif code == "rf":
        model = DecisionTreeModel
    elif code == "xgb":
        model = XGBoost
    elif code == "lgbm":
        model = LightGBM
    elif code == "cb":
        model = CatBoost
    elif code == "ls":
        model = Lasso
    else:
        raise argparse.ArgumentError(
            argument=models_arg,
            message=(
                "Only ",
                "'lr' (logistic regression)"
                ", 'rf' (random forest)"
                ", 'svd' (random forest with truncate svd)"
                ", 'xgb' (XGBoost)"
                ", 'lgbm (LightGBM)'"
                ", 'cb' (CatBoost)"
                ", 'ls' (Lasso)"
                " models are supported",
            ),
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    models_arg = parser.add_argument("--models", type=str, default=None)

    args = parser.parse_args()

    model_list = ["lr", "rf", "xgb", "lgbm", "cb", "ls"]
    if args.models is not None:
        model_list = args.models.split(",")

    models = [parse_model(m) for m in model_list]

    set_seed(config.SEED)
    validation_scores = []
    preds = []

    for model in models:
        for fold_ in range(config.FOLDS):
            score, predictions = run(fold_, model)
            validation_scores.append(score)
            preds.append(predictions)

    valid_auc = np.mean(validation_scores)
    print(f"Validation AUC = {valid_auc}")

    # create submission
    pred = np.mean(np.array(preds), axis=0)

    df_sub = pd.read_csv(config.SUBMISSION_SAMPLE)
    df_sub[config.TARGET] = pred

    ensemble_desc = str.join(".", model_list)
    dt = datetime.now().strftime("%y%m%d.%H%M")
    submission_file = Path(config.OUTPUTS) / f"{dt}-{ensemble_desc}-{valid_auc}.csv"
    submission_file.parent.mkdir(exist_ok=True)
    df_sub.to_csv(submission_file, index=False)
