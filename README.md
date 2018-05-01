# KKBOX_CHURN_PREDICTION

Done as a group project for a Kaggle competition - https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
Training and testing data can be gotten from the link. use create_compiled_data to compile train and test data into one csv each.

## Prerequisites

```
python3.6
pip install sklearn, pandas, numpy, xgboost, catboost

```

## Preprocessing the dataset

all the datasets should be stored in data/
```
python data_processing.py
python scratch.py
```

this should create two final csv - df_testfinal and df_trainfinal

## Training

```
python churn_prediction.py {option}
```

{option}:
1 - multilayer perceptron (deprecated because focus on boosters)
3 - xgboost
4 - catboost
5 - lightgbm
6 - ensemble

trained model will be saved in model/

## Testing

```
python churn_prediction.py 2 {model}
```
{model}:
xgb - xgboost
mlp - multilayer perceptron
lgb - lightgbm
cat - catboost

this will save the csv to be submitted to kaggle in results/{model}.csv

 