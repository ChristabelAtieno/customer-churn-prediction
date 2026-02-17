import pandas as pd
from sklearn.model_selection import train_test_split

def feature_engineering(data):

    #---categorical columsn
    categorical_cols = ['gender', 'Partner', 'Dependents','SeniorCitizen',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    #---get dummies for te categorical columns
    df = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    #---encode the churn column
    df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})

    #---drop unnecessary colums
    col_to_drop = ['customerID','gender_Male','PhoneService_Yes','TotalCharges','OnlineSecurity_No internet service', 'OnlineBackup_No internet service', 'DeviceProtection_No internet service',
       'TechSupport_No internet service', 'StreamingTV_No internet service', 'StreamingMovies_No internet service','InternetService_Fiber optic','MultipleLines_Yes','StreamingMovies_Yes','StreamingTV_Yes'  ]
    df = df.drop(columns=col_to_drop)

    return df

def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    #---train test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    return X, y, X_train, X_test, y_train, y_test
