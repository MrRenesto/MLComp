#basic tools
import os
import numpy as np
import pandas as pd
import warnings

#tuning hyperparameters
from bayes_opt import BayesianOptimization
# from skopt  import BayesSearchCV

#graph, plots
import matplotlib.pyplot as plt
import seaborn as sns

#building models
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import time
import sys
from sklearn.metrics import f1_score

#metrics
from sklearn.metrics import roc_auc_score, roc_curve
import shap
warnings.simplefilter(action='ignore', category=FutureWarning)


features_df = pd.read_csv('../../Data/ModelBuilding_train_data.csv')

test_data = pd.read_csv('../../Data/ModelBuilding_test_data.csv')

# Merge dataframes based on 'Id' column
# data = pd.merge(features_df, labels_df, on='Id')

# data = data.drop('Id', axis=1)
X = features_df.drop('label', axis=1)
y = features_df['label']

X_train = X
y_train = y


def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6,n_estimators=10000, output_process=False):

    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    n_folds = 5
    random_seed = 420

    def lgb_eval(learning_rate, num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,
                 min_sum_hessian_in_leaf, subsample):
        params = {'application': 'binary', 'metric': 'auc'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_depth))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        params['subsample'] = max(min(subsample, 1), 0)

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True,
                           metrics=['auc'])
        print(cv_result)
        return max(cv_result['valid auc-mean'])

    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),
                                            'num_leaves': (24, 80),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 30),
                                            'max_bin': (20, 90),
                                            'min_data_in_leaf': (20, 80),
                                            'min_sum_hessian_in_leaf': (0, 100),
                                            'subsample': (0.01, 1.0)}, random_state=200)

    # n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    # init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    model_auc = []
    for model in range(len(lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])

    # return best parameters
    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'], lgbBO.res[pd.Series(model_auc).idxmax()]['params']


opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=100, n_folds=5, random_seed=6, n_estimators=10000)
opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))
opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))
opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))
opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))
opt_params[1]['objective']='binary'
opt_params[1]['metric']='auc'
opt_params[1]['is_unbalance']=True
opt_params[1]['boost_from_average']=False
opt_params=opt_params[1]
print(opt_params)

# Use the optimal hyperparameters obtained from Bayesian Optimization
optimal_params = opt_params

# Additional parameters if needed
additional_params = {
    'num_boost_round': 1000,  # Adjust as needed
    'early_stopping_rounds': 50,  # Adjust as needed
    'verbose_eval': 100,  # Adjust as needed
}

# Merge the parameters
lgb_params = {**optimal_params, **additional_params}


train_data = pd.read_csv('../../Data/ModelBuilding_train_data.csv')
test_data = pd.read_csv('../../Data/ModelBuilding_test_data.csv')
y_traind = train_data['label']
X_traind = train_data.drop('label', axis=1)

# Split your data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_traind, y_traind, test_size=0.2, random_state=42)

y_test = test_data['label']
X_test = test_data.drop('label', axis=1)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# Train the LightGBM model
model = lgb.train(lgb_params, train_data, valid_sets=[train_data, valid_data])

# Assuming 'test_data' is your test dataset
test_predictions = model.predict(X_test, num_iteration=model.best_iteration)

print(test_predictions)

binary_predictions = (test_predictions > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_test, binary_predictions)

print(f"F1 Score: {f1}")

