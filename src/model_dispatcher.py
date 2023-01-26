from typing import List
import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, linear_model
import xgboost as xgb

from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgbm
from catboost import CatBoostClassifier

from common.encoding import (
    encode_to_onehot,
    reduce_dimensions_svd,
    encode_to_values,
    scale_values,
)

import config


class CustomModel:
    def __init__(
        self,
        data: pd.DataFrame,
        fold: int,
        target: str,
        cat_features: List[str],
        ord_features: List[str],
        test: pd.DataFrame = None,
    ):
        self.data = data
        self.fold = fold
        self.target = target
        self.cat_features = cat_features
        self.num_features = ord_features
        self.test = test

        self.features = cat_features + ord_features

    def encode(self):
        """Transforms data into x_train & x_valid"""
        pass

    def fit(self):
        """Fits the model on x_valid and train target"""
        pass

    def predict_and_score(self) -> float:
        """Predicts on x_valid data and score using AUC"""
        # we need the probability values as we are calculating RMSE
        # we will use the probability of 1s
        valid_preds = self.model.predict_proba(self.x_valid)[:, 1]

        return metrics.roc_auc_score(self.df_valid[self.target].values, valid_preds)

    def predict_test(self) -> np.ndarray:
        """Predicts on x_test data"""

        if self.test is None:
            return None

        # we will use the probability of 1s
        return self.model.predict_proba(self.x_test)[:, 1]


class LogisticRegressionModel(CustomModel):
    def encode(self):
        # scale numerical values
        scale_values(self.data, self.num_features, self.test)

        # get training & validation data using folds
        self.df_train = self.data[self.data.kfold != self.fold].reset_index(drop=True)
        self.df_valid = self.data[self.data.kfold == self.fold].reset_index(drop=True)

        if self.cat_features:
            # get encoded dataframes with new categorical features
            df_cat_train, df_cat_valid, df_cat_test = encode_to_onehot(
                self.df_train, self.df_valid, self.cat_features, self.test
            )

            # we have a new set of categorical features
            encoded_features = df_cat_train.columns.to_list() + self.num_features

            dfx_train = pd.concat(
                [df_cat_train, self.df_train[self.num_features]], axis=1
            )
            dfx_valid = pd.concat(
                [df_cat_valid, self.df_valid[self.num_features]], axis=1
            )
            dfx_test = None
            if self.test is not None:
                dfx_test = pd.concat(
                    [df_cat_test, self.test[self.num_features]], axis=1
                )

            self.x_train = dfx_train[encoded_features].values
            self.x_valid = dfx_valid[encoded_features].values
            if dfx_test is not None:
                self.x_test = dfx_test[encoded_features].values
        else:
            encoded_features = self.num_features

            self.x_train = self.df_train[self.num_features].values
            self.x_valid = self.df_valid[self.num_features].values
            if self.test is not None:
                self.x_test = self.test[self.num_features].values

    def fit(self):
        self.model = linear_model.LogisticRegression()

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class DecisionTreeModel(CustomModel):
    def encode(self):
        if self.cat_features:
            encode_to_values(self.data, self.cat_features, test=self.test)

        # get training & validation data using folds
        self.df_train = self.data[self.data.kfold != self.fold].reset_index(drop=True)
        self.df_valid = self.data[self.data.kfold == self.fold].reset_index(drop=True)

        self.x_train = self.df_train[self.features].values
        self.x_valid = self.df_valid[self.features].values
        if self.test is not None:
            self.x_test = self.test[self.features].values
        else:
            self.x_test = None

    def fit(self):
        self.model = ensemble.RandomForestClassifier(n_jobs=-1)

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class DecisionTreeModelSVD(DecisionTreeModel):
    def encode(self):
        super().encode()

        # FIX: We are not doing one hot encoding before svd!
        self.x_train, self.x_valid, self.x_test = reduce_dimensions_svd(
            self.x_train, self.x_valid, 120, x_test=self.x_test
        )


class XGBoost(DecisionTreeModel):
    def fit(self):
        self.model = xgb.XGBClassifier(
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,  # , max_depth=5, n_estimators=200
        )

        # fit model on training data
        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)


class LightGBM(DecisionTreeModel):
    def fit(self):
        params = {
            # # "n_estimators": 150,
            # # "categorical_feature": cat_indexes,
            # # "num_leaves": 107,
            # # "min_child_samples": 19,
            # # "learning_rate": 0.004899729720251191,
            # # # "log_max_bin": 10,
            # # "colsample_bytree": 0.5094776453903889,
            # # "reg_alpha": 0.007603254267129311,
            # # "reg_lambda": 0.008134379186044243,
            "random_state": config.SEED,
            "metric": "auc",
        }

        self.model = LGBMClassifier(**params)

        # fit model on training data
        self.model.fit(
            self.x_train,
            self.df_train.loc[:, self.target].values,
            eval_set=[(self.x_valid, self.df_valid[self.target].values)],
            callbacks=[lgbm.early_stopping(100, verbose=False)],
            verbose=False,
        )


class CatBoost(DecisionTreeModel):
    def fit(self):
        # https://www.kaggle.com/code/alexandershumilin/playground-series-s3-e1-catboost-xgboost-lgbm
        params = {
            # # "depth": 3,
            # # "learning_rate": 0.01,
            # # "rsm": 0.5,
            # # "subsample": 0.931,
            # # "l2_leaf_reg": 69,
            # # "min_data_in_leaf": 20,
            # # "random_strength": 0.175,
            "random_seed": config.SEED,
            "use_best_model": True,
            "task_type": "CPU",
            "bootstrap_type": "Bernoulli",
            "grow_policy": "SymmetricTree",
            "loss_function": "Logloss",
            "eval_metric": "AUC",
        }

        self.model = CatBoostClassifier(
            **params, num_boost_round=10000, early_stopping_rounds=500
        )

        # fit model on training data
        self.model.fit(
            self.x_train,
            self.df_train.loc[:, self.target].values,
            eval_set=[(self.x_valid, self.df_valid[self.target].values)],
            early_stopping_rounds=500,
            verbose=False,
        )


class Lasso(LogisticRegressionModel):
    def fit(self):
        self.model = linear_model.LassoCV(cv=10, random_state=config.SEED)

        self.model.fit(self.x_train, self.df_train.loc[:, self.target].values)

    def predict_and_score(self) -> float:
        valid_preds = self.model.predict(self.x_valid)

        return metrics.roc_auc_score(self.df_valid[self.target].values, valid_preds)

    def predict_test(self) -> np.ndarray:
        if self.test is None:
            return None

        return self.model.predict(self.x_test)
