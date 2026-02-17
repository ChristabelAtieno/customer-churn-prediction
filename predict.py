import mlflow
import mlflow.sklearn
import pandas as pd
from scripts.data_preprocessing import load_data, pre_process_data
from scripts.feature_engineering import feature_engineering, split_data

data_path = 'data/Telco_Customer_Churn.csv'

#---load the data
data = load_data(data_path)

#---preprocess the data
data = pre_process_data(data)

#---feature engineering
df = feature_engineering(data)

X, y, X_train, X_test, y_train, y_test = split_data(df)


loaded_model = mlflow.sklearn.load_model("runs:/3efee8ec134643c89f393e7ca65e1835/logreg_churn_model")

prediction = loaded_model.predict(X_test)

feature_names = X.columns

result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = prediction

print(result.head())
result.to_csv("predictions.csv", index=False)


