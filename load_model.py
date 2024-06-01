import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

# Import Database
data = pd.read_csv("fake_data.csv")
X = data.drop(columns=["date", "demand"])
X = X.astype('float')

# Define MLflow Model path
run_id = 'b3eab1750d9e41ed91373502f9e77483'
experiment_id = '630945390445534026'
model_path = f'/home/ubuntu/MLflow/mlruns/{experiment_id}/{run_id}/artifacts/rf_apples'

# Load model with sklearn flavor
model = mlflow.sklearn.load_model(model_path)

# Make predictions
predictions = model.predict(X)

# Calculate the mean prediction
mean_prediction = predictions.mean()
print(mean_prediction)