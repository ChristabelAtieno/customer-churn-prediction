import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score

def log_regression_model(df):
    #--split the data into X and y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    #---train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

    #---model
    lr = LogisticRegression(max_iter=1000,
                            solver='liblinear',
                            class_weight='balanced',
                            C=1.0,
                            penalty='l2',
                            random_state=42)

    #===start mlflow
    with mlflow.start_run(run_name="log_reg_run"):

        #---fit model
        lr.fit(X_train, y_train)

        #---predict
        y_pred_lr = lr.predict(X_test)

        acc=accuracy_score(y_test, y_pred_lr)
        prec=precision_score(y_test, y_pred_lr)
        rec=recall_score(y_test, y_pred_lr)
        f1=f1_score(y_test, y_pred_lr)

        print("Accuracy: ", acc)
        print("Recall: ", rec)
        print("Precision: ", prec)
        print("f1-score: ", f1)
        print("The confusion matrix: \n",confusion_matrix(y_test, y_pred_lr))
        print("The classification report: \n",classification_report(y_test, y_pred_lr))

        #---feature importance
        coefficient = lr.coef_[0]

        feature_importance = pd.DataFrame({
            'Features':X.columns,
            'Coefficients': coefficient
        }).sort_values(by='Coefficients', ascending=False).head(10)
        print("Feature importance: \n", feature_importance)
        
        #--cross validation
        sfk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(lr, X, y, cv=sfk, scoring='recall')
        print("The scores: \n", scores)
        print("Mean: ", scores.mean())

        #===mlflow log metrics and parameter
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("recall",rec)
        mlflow.log_metric("precision",prec)
        mlflow.log_metric("f1_score",f1)
        mlflow.log_metric("cv_mean_recall",scores.mean())

        #---log model and register
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="logreg_churn_model",
            registered_model_name="logreg_churn_model"
        )

        return lr, y_pred_lr, feature_importance, scores