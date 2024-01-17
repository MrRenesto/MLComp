from sklearn.ensemble import RandomForestClassifier

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel


# buildmodel(RandomForestClassifier(max_depth=26, max_features=8, min_samples_leaf=3, min_samples_split=7, n_estimators=193), False, False)
# [0.76028519 0.78427408 0.78708441 0.73439412 0.75686275]
# Mean F1_macro: 0.7645801086328853
# Standard Deviation of F1_macro: 0.019408726836141137

#buildmodel(RandomForestClassifier(), False, False)
# [0.76990234 0.74671841 0.79933492 0.75880253 0.77930618]
# Mean F1_macro: 0.7708128766949935
# Standard Deviation of F1_macro: 0.01795140247520668


buildmodel(RandomForestClassifier(max_depth=26, max_features=8, min_samples_leaf=3, min_samples_split=7, n_estimators=193), True, True)
# [0.80053143 0.74979887 0.82984692 0.79837398 0.77454545]
# Mean F1_macro: 0.7906193326859692
# Standard Deviation of F1_macro: 0.02691418801513422


#buildmodel(RandomForestClassifier(), True, True)
# [0.75778816 0.7597649  0.80838863 0.77772554 0.79085431]
# Mean F1_macro: 0.7789043068607037
# Standard Deviation of F1_macro: 0.01910876966895419