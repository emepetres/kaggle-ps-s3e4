from pathlib import Path
import pandas as pd
from sklearn import model_selection

from common.utils import set_seed
from common.kaggle import download_competition_data
import config

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    set_seed(config.SEED)

    # Download data if necessary
    download_competition_data(config.COMPETITION, config.INPUTS)

    # Read training data
    df = pd.read_csv(config.TRAIN_DATA)
    df_test = pd.read_csv(config.TEST_DATA)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(
        n_splits=config.FOLDS, random_state=config.SEED, shuffle=True
    )

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df[config.TARGET].values)):
        df.loc[v_, "kfold"] = f

    # save the new csv with kfold column
    Path(config.TRAIN_FOLDS).parent.mkdir(exist_ok=True)
    df.to_csv(config.TRAIN_FOLDS, index=False)
    df_test.to_csv(config.PREPROCESSED_TEST_DATA, index=False)
