import joblib
import numpy as np
import pandas as pd
import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import StandardScaler

# Load the AutoGluon model from the pickle file
#model = ag.load('../AutoML/predictions/AutoGluonModels/models/LightGBM_BAG_L2/model.pkl')
features_df = pd.read_csv('../../Data/test_features.csv')

model = TabularPredictor.load(path="../AutoML/predictions/AutoGluonModels/")
# Alternatively, if it's a TabularPredictor, you can use:
# predictor = TabularPredictor.load('your_model_directory')  # Load from the model directory

#scaler = StandardScaler()
#data_scaled = scaler.fit_transform(features_df)
#data_scaled = pd.DataFrame(data=data_scaled, columns=features_df.columns)
# Make predictions
predictions = model.predict(features_df, model='LightGBM_r130_BAG_L2')

# Print or use predictions
print("Predictions:", predictions)
predictions.to_csv("predictions.csv")
