from sklearn.ensemble import  ExtraTreesClassifier

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel


#Best Hyperparameters: {'learning_rate': 0.0868414542196764, 'max_depth': 11, 'min_samples_leaf': 9, 'min_samples_split': 20, 'n_estimators': 199, 'subsample': 0.9318711362027603}


buildmodel(ExtraTreesClassifier(), True, False, True)