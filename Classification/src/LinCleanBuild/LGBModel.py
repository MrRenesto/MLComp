from lightgbm import LGBMClassifier

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel

# {'bagging_fraction': 0.39542184201392894, 'bagging_frequency': 8, 'feature_fraction': 0.809330431307996, 'max_depth': 17, 'min_data_in_leaf': 15, 'n_estimators': 174, 'num_leaves': 1455}


# buildmodel(LGBMClassifier(), True, True)


param = {'num_leaves': 1455, 'objective': 'binary',
             'n_estimators': 174,
             'max_depth': 17,
             'bagging_fraction': 0.39542184201392894,
             'feature_fraction': 0.809330431307996,
             'min_data_in_leaf': 15
             }

classifier = LGBMClassifier()
classifier.set_params(**param)
buildmodel(classifier, True, False, True)

