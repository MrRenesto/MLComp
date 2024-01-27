from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from Excecutable.Classification.ModelBuilding import buildmodel


# This created my Best Score on the Public Leaderboard: 0.81706
# Local Cross Validation Score: 0.808049
# The Ensamble includes the best Models i got with the Algorythm
# Press Y at the end to create Prediction. Do you want to build Model and Submit Results? Y/N:


# Define multiple models
model1 = RandomForestClassifier(max_depth=26, max_features=8, min_samples_leaf=3, min_samples_split=7, n_estimators=193)
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
model5 = CatBoostClassifier()

ensemble_model = StackingClassifier(estimators=[('rf', model1), ('gb', model2), ('svm', model3),
                                                ('lgbm', model4), ('md5', model5)])

buildmodel(ensemble_model,True,False,True)
