from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel


#Best Hyperparameters: {'learning_rate': 0.0868414542196764, 'max_depth': 11, 'min_samples_leaf': 9, 'min_samples_split': 20, 'n_estimators': 199, 'subsample': 0.9318711362027603}


buildmodel(BaggingClassifier(n_estimators=79, max_samples=1304), True, False, True)
#{'max_samples': 1304, 'n_estimators': 79}