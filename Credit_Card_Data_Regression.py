# IMPORT NECESSARY LIBRARIES

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# IMPORT DATASET

credit_card_dataset = pd.read_excel('H:/DATA SCIENCE/Real-Time Datasets/Logistic_Regression/Credit_Card_data/default of credit card clients.xls')

# DATA PREPARATION 
    
    ## DATA CLEANING
    
credit_card_dataset.drop(index = [0], inplace = True)
credit_card_dataset.columns = ['Limit_balance','Sex','Education','Marriage','Age','Pay_0','Pay_1','Pay_2','Pay_3','Pay_4','Pay_5','Pay_6','Bill_amt_1','Bill_amt_2','Bill_amt_3','Bill_amt_4','Bill_amt_5','Bill_amt_6','Pay_amt_1','Pay_amt_2','Pay_amt_3','Pay_amt_4','Pay_amt_5','Pay_amt_6','Default_next_month_payment']

    ## INITIAL ANALYSIS

data_shape         = credit_card_dataset.shape
data_description   = credit_card_dataset.describe(include='all')
data_dtypes        = credit_card_dataset.dtypes
data_null_check    = credit_card_dataset.isna().sum()

    ## DATA TRANSFORMATION

credit_card_dataset['Limit_balance'] = credit_card_dataset['Limit_balance'].astype(float)
credit_card_dataset['Sex'] = credit_card_dataset['Sex'].astype(int)
credit_card_dataset['Education'] = credit_card_dataset['Education'].astype(int)
credit_card_dataset['Marriage'] = credit_card_dataset['Marriage'].astype(int)
credit_card_dataset['Age'] = credit_card_dataset['Age'].astype(int)
credit_card_dataset['Pay_0'] = credit_card_dataset['Pay_0'].astype(int)
credit_card_dataset['Pay_1'] = credit_card_dataset['Pay_1'].astype(int)
credit_card_dataset['Pay_2'] = credit_card_dataset['Pay_2'].astype(int)
credit_card_dataset['Pay_3'] = credit_card_dataset['Pay_3'].astype(int)
credit_card_dataset['Pay_4'] = credit_card_dataset['Pay_4'].astype(int)
credit_card_dataset['Pay_5'] = credit_card_dataset['Pay_5'].astype(int)
credit_card_dataset['Pay_6'] = credit_card_dataset['Pay_6'].astype(int)







# MODEL BUILDING

X = credit_card_dataset.drop(labels = ['Default_next_month_payment'],axis=1)
y = credit_card_dataset[['Default_next_month_payment']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25 , random_state = 12, stratify = y)

# MODEL TRAINING

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
