import pandas as pd

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
