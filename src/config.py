DATA_STORAGE_PATH = "/run/media/jcarnero/linux-data"
COMPETITION = "playground-series-s3e4"
SEED = 42

DATA_PATH = DATA_STORAGE_PATH + "/kaggle/" + COMPETITION
INPUTS = DATA_PATH + "/input"
PREPROCESSED = DATA_PATH + "/preprocess"
INTERMEDIATE = DATA_PATH + "/intermediate"
OUTPUTS = DATA_PATH + "/output"

FOLDS = 5
TARGET = "Class"
TRAIN_DATA = INPUTS + "/train.csv"
TEST_DATA = INPUTS + "/test.csv"
SUBMISSION_SAMPLE = INPUTS + "/sample_submission.csv"

PREPROCESSED_TRAIN_DATA = PREPROCESSED + "/train.csv"
PREPROCESSED_TEST_DATA = PREPROCESSED + "/test.csv"
TRAIN_FOLDS = PREPROCESSED + "/train_folds.csv"
