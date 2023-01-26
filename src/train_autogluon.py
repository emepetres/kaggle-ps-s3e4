from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn import metrics

import config


def run(fold: int) -> Tuple[float, np.ndarray]:
    # load the full training data with folds
    df = pd.read_csv(config.TRAIN_FOLDS)
    df_test = pd.read_csv(config.PREPROCESSED_TEST_DATA)

    # let drop columns that doesn't hold information
    df.drop(columns=["Over18"], inplace=True)
    df_test.drop(columns=["Over18"], inplace=True)

    # get training & validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True).drop(columns="kfold")
    df_valid = df[df.kfold == fold].reset_index(drop=True).drop(columns="kfold")

    predictor = TabularPredictor(
        label=config.TARGET,
        path=config.INTERMEDIATE + "/autogluon",
        eval_metric="roc_auc",
    ).fit(
        TabularDataset(df_train),
        tuning_data=TabularDataset(df_valid),
        use_bag_holdout=True,
        presets="best_quality",
    )

    # predict on validation data and get rmse score
    valid_preds = predictor.predict_proba(TabularDataset(df_valid)).iloc[:, 1]
    auc = metrics.roc_auc_score(df_valid[config.TARGET].values, valid_preds)

    # print rmse
    print(f"Fold = {fold}, AUC = {auc}")

    # predict on test
    preds = predictor.predict_proba(TabularDataset(df_test)).iloc[:, 1]

    return (auc, preds)


if __name__ == "__main__":
    validation_scores = []
    preds = []
    for fold_ in range(config.FOLDS):
        score, predictions = run(fold_)
        validation_scores.append(score)
        preds.append(predictions)

    valid_auc = np.mean(validation_scores)
    print(f"Validation AUC = {valid_auc}")

    # create submission
    pred = np.mean(np.array(preds), axis=0)

    df_sub = pd.read_csv(config.SUBMISSION_SAMPLE)
    df_sub[config.TARGET] = pred

    dt = datetime.now().strftime("%y%m%d.%H%M")
    submission_file = Path(config.OUTPUTS) / f"{dt}-autogluon-{valid_auc}.csv"
    submission_file.parent.mkdir(exist_ok=True)
    df_sub.to_csv(submission_file, index=False)
