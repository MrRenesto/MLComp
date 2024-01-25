import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel
from Classification.src.ResultHandler import upload_result

# Load feature data
random_state = 69

# Define multiple models
model1 = RandomForestClassifier(random_state=random_state)
model2 = GradientBoostingClassifier(learning_rate=0.0868414542196764,
                                      max_depth=11,
                                      min_samples_leaf=9,
                                      min_samples_split=20,
                                      n_estimators=199,
                                      subsample=0.9318711362027603)
model3 = CatBoostClassifier(depth=8, l2_leaf_reg=10, learning_rate=0.18, num_trees=121)


param = {'num_leaves': 1455, 'objective': 'binary',
             'n_estimators': 174,
             'max_depth': 17,
             'bagging_fraction': 0.39542184201392894,
             'feature_fraction': 0.809330431307996,
             'min_data_in_leaf': 15
             }

model4 = LGBMClassifier()
model4.set_params(**param)

# Create a Voting Classifier
ensemble_model = StackingClassifier(estimators=[('rf', model1), ('gb', model2), ('svm', model3), ('lgbm', model4)])

buildmodel(ensemble_model,True,False,True)
