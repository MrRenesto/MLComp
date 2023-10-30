import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from DataProcessing import *
from sklearn.model_selection import cross_val_score
from sklearn import svm
from ResultHandler import *

X_train, X_test, y_train, y_test = get_train_and_test_data()

y_test_array = y_test[:, 1]
y_train_array = y_train[:, 1]

knn = svm.SVC()
knn.fit(X_train, y_train_array)
"""
import matplotlib.pyplot as plt

plt.plot(k_values,scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.grid(True)
plt.show()
"""

y_pred_val_test = knn.predict(X_test)
y_pred_labels = y_pred_val_test

report = classification_report(y_test_array, y_pred_labels, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)

validation_data, X_val = get_validation_data()
X_val = preprocess(X_val)
"""
y_pred_val = knn.predict(X_val)
y_pred_val = [pred[1] for pred in y_pred_val]

upload_result(validation_data, y_pred_val, "kNN Predictions with K = " + str(best_k) + " Report: " + report)
"""
