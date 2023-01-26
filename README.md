# kaggle-ps-s3e4

Kaggle Playground Series 3, Episode 4 competition

NOTES:

Tasks to obtain the best model:

* [x] Basic eda
* [ ] Cross-validation
* [ ] First training
* [ ] Detailed eda
* [ ] Feature engineering
* [ ] Use ten folds
* [ ] Use AutoGluon framework
* [ ] Merge original dataset
* [ ] Hyperparameter tunning
* [ ] Ensemble of best tunned models
* [ ] Ensemble of multiple algorithms
* [ ] Ensemble of multiple seeds, ¿or without setting it?
* [ ] Predict with fastai tabular

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [rf|svd|xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)
