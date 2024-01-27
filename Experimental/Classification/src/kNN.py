from sklearn.metrics import classification_report
from DataProcessing import *
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from Excecutable.Classification.ResultHandler import *

X_train, X_test, y_train, y_test = get_train_and_test_data()



correlation_matrix = np.corrcoef(X_train, rowvar=False)

# Identify highly correlated features
highly_correlated = set()  # Store the indices of highly correlated features
threshold = 0.95  # Set the correlation threshold

num_features = X_train.shape[1]

for i in range(num_features):
    for j in range(i):
        corr = correlation_matrix[i, j]
        if abs(corr) > threshold:
            highly_correlated.add(i)  # Store the index of the highly correlated feature

# Drop the highly correlated features from your dataset
X_train = np.delete(X_train, list(highly_correlated), axis=1)
X_test = np.delete(X_test, list(highly_correlated), axis=1)


y_test_array = y_test[:, 1]
y_train_array = y_train[:, 1]

print(X_train.shape)
print(X_test.shape)
print(y_test_array.shape)
print(y_train_array.shape)

k_values = [i for i in range (1,21)]
scores = []
best_f1_score = 0  # Track the best F1 score
best_k = 0  # Track the best k value




for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train_array, cv=10, scoring='f1_macro')
    mean_f1 = np.mean(score)

    print(str(k) + ": has f1 score: " + str(mean_f1))

    if mean_f1 > best_f1_score:
        best_f1_score = mean_f1
        best_k = k

print("Best Index: " + str(best_k) + " with: " + str(best_f1_score))


knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
"""
import matplotlib.pyplot as plt

plt.plot(k_values,scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.grid(True)
plt.show()
"""

y_pred_val_test = knn.predict(X_test)
y_pred_labels = y_pred_val_test[:, 1]

report = classification_report(y_test_array, y_pred_labels, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)

validation_data, X_val = get_validation_data()
X_val = preprocess_training_data(X_val)


X_val = np.delete(X_val, list(highly_correlated), axis=1)

y_pred_val = knn.predict(X_val)
y_pred_val = [pred[1] for pred in y_pred_val]

upload_result(validation_data, y_pred_val, "kNN Predictions with K = " + str(best_k) + " Report: " + report)
