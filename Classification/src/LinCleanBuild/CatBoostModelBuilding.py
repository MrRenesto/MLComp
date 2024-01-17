from catboost import CatBoostClassifier

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel

#buildmodel(CatBoostClassifier(depth=8, l2_leaf_reg=10, learning_rate=0.18, num_trees=121), False, False)
# [0.78710142 0.77301111 0.80411822 0.79069225 0.77618406]


buildmodel(CatBoostClassifier(depth=8, l2_leaf_reg=10, learning_rate=0.18, num_trees=121), True, True)
# [0.77400309 0.78504821 0.81308368 0.80106344 0.79617261]
# Mean F1_macro: 0.7938742065361675
# Standard Deviation of F1_macro: 0.01340694704179604


#buildmodel(CatBoostClassifier(depth=8, l2_leaf_reg=10, learning_rate=0.18, num_trees=121), True, False)
# [0.77800061 0.80931511 0.79158682 0.77779319 0.79212912]
# Mean F1_macro: 0.7897649688938688
# Standard Deviation of F1_macro: 0.011600342239140861

#buildmodel(CatBoostClassifier(), False, False)
# [0.78275845 0.80101896 0.79933492 0.76474908 0.78435177]
# Mean F1_macro: 0.7864426378771844
# Standard Deviation of F1_macro: 0.013169930550282517


# buildmodel(CatBoostClassifier(), True, True)
# [0.76491204 0.79227892 0.81624709 0.80961541 0.77851869]
# Mean F1_macro: 0.792314430255671
# Standard Deviation of F1_macro: 0.019043689488203817
