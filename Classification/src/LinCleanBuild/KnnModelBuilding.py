from sklearn.neighbors import KNeighborsClassifier

from Classification.src.LinCleanBuild.ModelBuilding import buildmodel

#buildmodel(KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'), True, True)
# [0.71800474 0.75419338 0.74773604 0.74122652 0.74138709]
# Mean F1_macro: 0.7405095511737869
# Standard Deviation of F1_macro: 0.012224727227685153


#buildmodel(KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'), False, True)
# [0.72855123 0.76058507 0.75764735 0.74348841 0.76042082]
# Mean F1_macro: 0.7501385757673893
# Standard Deviation of F1_macro: 0.012501788048134324


buildmodel(KNeighborsClassifier(), True, True)
# [0.72726682 0.74597314 0.76487514 0.72945757 0.71358177]
# Mean F1_macro: 0.7362308895991192
# Standard Deviation of F1_macro: 0.01763303595606785