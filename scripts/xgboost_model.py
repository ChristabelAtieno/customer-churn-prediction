import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score

def xgbclassifier_model(df):

    #---split the data into X and y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    #---train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #model
    xgb = XGBClassifier(objective='binary:logistic',
                        eval_metrics='logloss',
                        use_label_encoder=False,
                        random_state=42)
    
    #---handle class imbalanceness by giving more wight to the minority class
    scale_pos_weight = (y==0).sum() / (y==1).sum()

    #parameters
    param_grid = {
        'n_estimators':[100,200,300],
        'max_depth':[3,5,7],
        'learning_rate':[0.01,0.05,0.1],
        'subsample':[0.8,1.0],
        'colsample_bytree':[0.8,1.0],
        'scale_pos_weight':[scale_pos_weight]
    }

    #---hyperparameter tune
    rs = RandomizedSearchCV(xgb, 
                            param_distributions=param_grid,
                            scoring='f1',
                            cv=3,
                            verbose=2,
                            n_jobs=-1)
    
    #start mlflow
    with mlflow.start_run(run_name="xgboost_run"):

        #---fit model
        rs.fit(X_train, y_train)

        #---predict
        y_pred_xgb = rs.predict(X_test)

        acc=accuracy_score(y_test, y_pred_xgb)
        rec=recall_score(y_test, y_pred_xgb)
        prec=precision_score(y_test, y_pred_xgb)
        f1=f1_score(y_test, y_pred_xgb)

        print("Accuracy: ", acc)
        print("Recall: ", rec)
        print("Precision: ", prec)
        print("F1 score: ", f1)
        print("\nThe confusion matrix: \n",confusion_matrix(y_test, y_pred_xgb))
        print("\nThe classification report: \n",classification_report(y_test, y_pred_xgb))

        #---feature importance
        best_estimators =rs.best_estimator_
        importance = best_estimators.feature_importances_
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False).head(10)

        print('Important features: \n',feature_importance)

        #---cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(rs,
                                X,
                                y,
                                cv=skf,
                                scoring='recall')
        print("The CV scores: \n", scores)
        print("Mean: ", scores.mean())

        #===mlflow parameters amd metrics==
        mlflow.log_params(rs.best_params_)
        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("recall",rec)
        mlflow.log_metric("precision",prec)
        mlflow.log_metric("f1_score",f1)
        mlflow.log_metric("cv_mean_recall",scores.mean())

        #===log and register the model
        mlflow.xgboost.log_model(best_estimators, 
                                  "xgboost_churn_model",
                                  registered_model_name="xgboost_churn_model")

        return rs, y_pred_xgb, feature_importance, scores