import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from keras_tuner import RandomSearch

from Classification.src.ResultHandler import upload_result

# Load your data
features_df = pd.read_csv('../../Data/train_features.csv')
labels_df = pd.read_csv('../../Data/train_label.csv')

data = pd.merge(features_df, labels_df, on='Id')
data = data.drop(['Id', 'feature_2'], axis=1)

X = data.drop('label', axis=1)
y = data['label']

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=hp.Int('units_hidden', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Instantiate the Keras Tuner RandomSearch tuner
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=50, executions_per_trial=1, directory='my_tuner_dir5', project_name='my_tuner_project')


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Use SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_2, y_train_2 = smote.fit_resample(X_train_2, y_train_2)

# Perform the hyperparameter search
tuner.search(
    x=X_train_2,  # Training data
    y=y_train_2,  # Training labels
    epochs=20,
    validation_data=(X_test_2, y_test_2),  # Your own validation data
)


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# Use SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Build the model with the best hyperparameters and train it
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=20, validation_split=0.1)

# Evaluate the model on the test set
test_preds = (best_model.predict(X_test) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Print the best hyperparameters
print("Best Hyperparameters:")
print(f"Units in Input Layer: {best_hps.get('units_input')}")
print(f"Units in Hidden Layer: {best_hps.get('units_hidden')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")


labels_df = pd.read_csv('../../Data/test_features.csv')

labels_df2 = labels_df.drop('Id', axis=1)
labels_df2 = labels_df2.drop('feature_2', axis=1)

scaler = StandardScaler()
labels_df2 = scaler.fit_transform(labels_df2)

#labels_df_pred = best_model.predict(labels_df2)
labels_df_pred = (best_model.predict(labels_df2) > 0.5).astype(int)


array = labels_df_pred.tolist()
from itertools import chain
flat_list = list(chain.from_iterable(array))

print(flat_list)

upload_result(labels_df, flat_list, "tensorflow")
