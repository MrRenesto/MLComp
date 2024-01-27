from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from Excecutable.Classification.ResultHandler import *

# Load feature data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

# Merge dataframes based on 'Id' column
data = pd.merge(features_df, labels_df, on='Id')

data = data.drop('Id', axis=1)
data = data.drop('feature_2', axis=1)

X = data.drop('label', axis=1)  # Assuming 'label' is the column with your labels
y = data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

rs_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (200,), (100,75), (100,100)],
    #'activation': ['relu', 'tanh'],
    #'alpha': loguniform(1e-5, 1e0),  # alpha is usually better to sample from a log scale
    'learning_rate_init': (0.0001, 0.01),
    'max_iter': [300, 400, 500]
}

random_search = RandomizedSearchCV(estimator=MLPClassifier(),
                                   param_distributions=rs_params,
                                   scoring='f1_macro',
                                   cv=5,
                                   n_iter=10,  # Number of random combinations to try
                                   verbose=2,
                                   n_jobs=-1)


# Fit the grid search to the data
random_search.fit(X, y)

# Print the best parameters and corresponding F1 score
print("Best Parameters: ", random_search.best_params_)
print("Best F1 Score: ", random_search.best_score_)
'''
rf_classifier = LGBMClassifier()

# You can also calculate and print the average F1 score across all classes
f1_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='f1_macro')
print("Average F1 Score: " + str(f1_scores))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", report)

labels_df = pd.read_csv('..\\..\\Data\\test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

labels_df2 = scaler.fit_transform(labels_df2)

labels_df_pred = rf_classifier.predict(labels_df2)

upload_result(labels_df, labels_df_pred, "RandomForest F1 Report: " + str(report))
'''
