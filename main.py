from scripts.data_preprocessing import load_data, pre_process_data
from scripts.feature_engineering import feature_engineering
from scripts.log_reg_model import log_regression_model
from scripts.xgboost_model import xgbclassifier_model
import mlflow

if __name__== "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn_model")

    data_path = 'data/Telco_Customer_Churn.csv'

    #---load the data
    data = load_data(data_path)

    #---preprocess the data
    data = pre_process_data(data)

    #---feature engineering
    df = feature_engineering(data)

    #---logistic regression model
    lr_model, lr_pred, lr_features, lr_cv = log_regression_model(df)

    #---xgboost model
    xgb_model, xgb_pred, xgb_features, xgb_cv = xgbclassifier_model(df)
    