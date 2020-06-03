#!/usr/bin/env python
# coding: utf-8

# ## Customer Churn Prediction for TELCO Company
# #### We have customer data of TELCO company with several features.
# #### Now, because lots of customers are leaving this company, so as part of customer retention program we need to predict customer churn before they decide to leave.<br/>In order to do that we need to use this data and create machine learning model for customer churn prediction.

import pandas as pd
import numpy as np
import pickle

print("Start training model Customer Churn Prediction...")

pd.set_option('display.max_columns', 50)

print("Reading in the Teclco-Customer-Churn.csv file...")
df=pd.read_csv('Telco-Customer-Churn.csv')
# df.head()

# #### We have 7043 records with 21 different features including customer id and churn
df.shape

# #### Most of the columns are object type, but for developing model we need numeric values and some label encoding as well, so now we will change datatype of some columns and change string values to numeric values by encoding them.
df.dtypes

def changeColumnsToString(df):
    columnsNames=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
    for col in columnsNames:
        df[col]=df[col].astype('str').str.replace('Yes','1').replace('No','0').replace('No internet service','0').replace('No phone service',0)

changeColumnsToString(df)

df['SeniorCitizen']=df['SeniorCitizen'].astype(bool)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

# df.head(2)


# #### There are some categorical values which can be encoded as numbers, so we will take a look at unique values present as categories and convert these fields as category and encode them.

print("Payment methods: ",df.PaymentMethod.unique())
print("Contract types: ",df.Contract.unique())
print("Gender: ",df.gender.unique())
print("Senior Citizen: ",df.SeniorCitizen.unique())
print("Internet Service Types: ",df.InternetService.unique())

df['gender']=df['gender'].astype('category')
df['PaymentMethod']=df['PaymentMethod'].astype('category')
df['Contract']=df['Contract'].astype('category')
df['SeniorCitizen']=df['SeniorCitizen'].astype('category')
df['InternetService']=df['InternetService'].astype('category')

# df.dtypes


# #### Here we have encoded fields with numbers by pandas build-in get_dummies method, and using that method we need to give prefix for new fields which will be generated.
# #### This method will generate new fields with prefix and category name as column name and 0 or 1 will be their value.
# #### As we can see below, we got all the new fields with values as 0 or 1.

dfPaymentDummies = pd.get_dummies(df['PaymentMethod'], prefix = 'payment')
dfContractDummies = pd.get_dummies(df['Contract'], prefix = 'contract')
dfGenderDummies = pd.get_dummies(df['gender'], prefix = 'gender')
dfSeniorCitizenDummies = pd.get_dummies(df['SeniorCitizen'], prefix = 'SC')
dfInternetServiceDummies = pd.get_dummies(df['InternetService'], prefix = 'IS')

# print(dfPaymentDummies.head(3))
# print(dfContractDummies.head(3))
# print(dfGenderDummies.head(3))
# print(dfSeniorCitizenDummies.head(3))
# print(dfInternetServiceDummies.head(3))


# #### Now we have new dataframes by label encoding, so we will concat them with our existing dataframe, but before that we will remove category fields as we don't need them right!

df.drop(['gender','PaymentMethod','Contract','SeniorCitizen','InternetService'], axis=1, inplace=True)

df = pd.concat([df, dfPaymentDummies], axis=1)
df = pd.concat([df, dfContractDummies], axis=1)
df = pd.concat([df, dfGenderDummies], axis=1)
df = pd.concat([df, dfSeniorCitizenDummies], axis=1)
df = pd.concat([df, dfInternetServiceDummies], axis=1)
# df.head(2)


# #### For a bit of simplicity, we'll rename some column names

df.columns = ['customerID', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No']


# #### We'll convert all fields to number type in dataframe for our model.
# #### Here we are wrapping up data preparation phase.

numericColumns=np.array(['Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No'])

for columnName in numericColumns:
    df[columnName]=pd.to_numeric(df[columnName],errors='coerce')
# df.dtypes


# #### We'll save our model data to new csv file without customerID, as we won't be using that in our model development.

modelData = df.loc[:, df.columns != 'customerID']
modelData.to_csv('modelData.csv')


# ### <b>Model Development</b>
# #### After reading our model data, we'll take our training and target data in numpy arrays

modelData=pd.read_csv('modelData.csv')

modelData[modelData==np.inf]=np.nan
modelData.fillna(modelData.mean(), inplace=True)

x=np.asarray(modelData.loc[:,modelData.columns != 'Churn'])
y=np.asarray(modelData['Churn'])

# print(x[:2])
# print(y[:2])


# #### Here we'll normalize our data by using sklearn's StandardScaler
from sklearn import preprocessing

x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:2]


# #### It is recommended practice of splitting data in training and testing before using it in model, in our case we are keeping 80/20 data for training and testing respectively.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=72)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# #### Before fitting data to our model, feature selection is very essential part of model development.
# #### Here we are using sklearn's RandomForestClassifier with ensemble learning to choose most relevent features for our model. It will iteratively select most relevent features and eliminate least relevent features and threshold will be median for feature selection.

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=37), threshold='median')
select.fit(X_train, y_train)

X_train_s = select.transform(X_train)

print('The shape of x_train: ',X_train.shape)
print('The shape of x_train_s: ',X_train_s.shape)

# #### We're fitting our training data to LogicalRegression and making prediction on our test data.
# #### Accuracy of our model is around 79%, and that means 79/100 times we can make correct prediction.

X_test_s = select.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
score = lr.fit(X_train_s,y_train).score(X_test_s, y_test)
print('The score of Logistic Regerssion for customer churn: {:.3f}'.format(score))

# save the model to disk
print("Saving model to disk...")
pkl_filename = 'customer_churn_lr_v0.1.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(lr, file)


print("Model saved successfully...")
print("Done.")

