# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:35:44 2017

@author: Fawad
"""

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Function to create dummy variables

def create_dummies(df, col_name):
    dummies = pd.get_dummies(data[col_name], prefix= col_name)

    if (data[col_name].isnull().values.any()) == False:

        df = pd.concat([data, dummies.iloc[:, :-1]], axis=1)
        df.drop(col_name, axis=1, inplace=True)
    else:

        df = pd.concat([data, dummies], axis=1)
        df.drop(col_name, axis=1, inplace=True)

    return df

#Name the different types of variable

categorical = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig',
                  'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                  'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 'SaleType', ]


ordinal = ['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',
           'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive', 'GarageCond',
           'PoolQC', 'Fence']


ord_dict = {'Functional': {'Sal':0, 'Sev':1, 'Mod': 4, 'Maj1': 3, 'Typ': 7, 'Maj2': 2, 'Min1': 6, 'Min2': 5},
            'GarageQual': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'ExterCond': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'BsmtFinType2': {'Unf': 1, 'ALQ': 5, 'LwQ': 2, 'GLQ': 6, 'BLQ': 4, 'Rec': 3},
            'Utilities': {'AllPub':3, 'NoSewr':2, 'NoSeWa': 1, 'ELO':0},
            'PoolQC': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'BsmtFinType1': {'Unf': 1, 'ALQ': 5, 'LwQ': 2, 'GLQ': 6, 'BLQ': 4, 'Rec': 3},
            'FireplaceQu': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
            'PavedDrive': {'P': 1, 'Y': 2, 'N': 0},
            'Electrical': {'Mix':0, 'FuseP': 1, 'FuseA': 3, 'SBrkr': 4, 'FuseF': 2},
            'BsmtCond': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'BsmtQual': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'KitchenQual': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'LotShape': {'Reg':3, 'IR1':2, 'IR2': 1, 'IR3':0 },
            'ExterQual': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'BsmtExposure': {'Av': 4, 'Gd': 5, 'Mn': 3, 'No': 2, 'NA': 1},
            'Fence': {'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'GdPrv': 4},
            'HeatingQC': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5},
            'GarageFinish': {'Unf': 1, 'Fin': 2, 'RFn': 1},
            'GarageCond': {'Fa': 2, 'Gd': 4, 'TA': 3, 'Po': 1, 'Ex': 5}}

age = ['YearBuilt', 'YearRemod/Add']

data = pd.read_csv("Training data.csv")

response=['SalePrice']

predictors= [x for x in list(data.columns) if x not in response]

data = data[response+predictors]


#replace NA in ordinal columns with 0
for columns in ord_dict:
    data[columns].fillna(0, inplace=True)

#replace ordinal columns with numeric ordinal values
for columns in ord_dict:
    for keys in ord_dict[columns]:
        data[columns].replace(keys, ord_dict[columns][keys], inplace=True)

#categorical columns replaced with
for categories in categorical:
    data = create_dummies(data, categories)

#replace the age columns with age
data['age_built'] = data['YrSold'] - data['YearBuilt'] + 1
data['age_remodelled'] = data['YrSold'] - data['YearRemod/Add'] + 1
data.drop('YearBuilt', axis=1, inplace=True)
data.drop('YearRemod/Add', axis=1, inplace=True)



#Store Column names in a list
column_names = data.columns


from sklearn.preprocessing import Imputer
imputer= Imputer(strategy= 'mean')
data_imp = imputer.fit_transform(data)

#load X and Y variables
Y= data_imp[:,0]
X= data_imp[:,1:]
Y = Y.ravel()


#Scale feature variables
from sklearn.preprocessing import StandardScaler
scalar= StandardScaler()
X = scalar.fit_transform(X)


from sklearn.feature_selection import f_regression, mutual_info_regression


f_test, _ = f_regression(X, Y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, Y)
mi /= np.max(mi)

print(mi.shape)

threshold = 0.1
selected_columns = []
selected_column_names = []



for i in range(0,len(mi)):
    if mi[i]>threshold:
        print(column_names[1+i],"F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]))
        selected_columns.append(i)
        selected_column_names.append(column_names[1+i])
    elif f_test[i]>threshold:
        print(column_names[1 + i], "F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]))
        selected_columns.append(i)
        selected_column_names.append(column_names[1 + i])


# Use this section to obtain the scatter plots for each variable

for i in selected_columns:

     plt.scatter(X[:, i], Y, edgecolor='black', s=5)
     plt.xlabel(column_names[1+i], fontsize=8)
     if i == 0:
         plt.ylabel("$y$", fontsize=8)
     plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
               fontsize=8)
     plt.show()




data_updated = data[selected_column_names]

import seaborn as sns



# calculate the correlation matrix
corr = data_updated.corr()

# plot the heatmap
sns.set(font_scale=0.7)
sns.heatmap(corr)

plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.show()


selected_column_names.remove('Exterior2nd_VinylSd')
selected_column_names.remove('Fireplaces')
selected_column_names.remove('GarageCars')
selected_column_names.remove('GarageCond')
selected_column_names.remove('TotRmsAbvGrd')

data_updated = data[selected_column_names]

#calculate the correlation matrix
corr = data_updated.corr()

# plot the heatmap
sns.set(font_scale=0.7)
sns.heatmap(corr)

plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.show()


data_imp = imputer.fit_transform(data_updated)

X= data_imp[:,:]
X = scalar.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2, test_size=0.2)
alpha_range = 10. ** np.arange(-10, 10)
l1_range = np.arange(0,1,0.1)
mse = []

# for i in l1_range:
#     elascv = ElasticNetCV(l1_ratio = i, alphas=alpha_range, cv=10, max_iter=10000, selection="random", n_jobs=-1)
#     elascv.fit(X_train, y_train)
#     preds_elascv = elascv.predict(X_test)
#     mse.append(np.sqrt(mean_squared_error(y_test, preds_elascv)))
#
#
# plt.plot(l1_range, mse)
# plt.show()
#


# # OLS prediction and error measure block
# lm = LinearRegression()
# lm.fit(X_train, y_train)
# preds_ols_test = lm.predict(X_test)
# print("OLS:  {:0.4f}".format(np.sqrt(mean_squared_error(y_test, preds_ols_test))))
#
# # Initialising Ridge Regression and alpha range code block
# alpha_range = 10. ** np.arange(-10, 10)
# rregcv = RidgeCV(alphas=alpha_range, cv=5, scoring='neg_mean_squared_error')
# rregcv.fit(X_train, y_train)
# ridge = Ridge(alpha=rregcv.alpha_)
#
# preds_ridge = rregcv.predict(X_test)
# print("RIDGE error: {:0.4f}".format(np.sqrt(mean_squared_error(y_test, preds_ridge))))
#
#
# # Initialising Lasso code block
# lascv = LassoCV(normalize=True, alphas=alpha_range, cv=5, max_iter=1000)
# lascv.fit(X_train, y_train)
# preds_lassocv = lascv.predict(X_test)
# print("LASSO error: {:0.4f}".format(np.sqrt(mean_squared_error(y_test, preds_lassocv))))
# print("LASSO Lambda: {:0.4f}".format(lascv.alpha_))
#
#

#ElasticnNet Code Block
elascv = ElasticNetCV(l1_ratio = 0.3, alphas=alpha_range, cv=5, max_iter=10000, selection="random", n_jobs=-1)
elascv.fit(X_train, y_train)
preds_elascv = elascv.predict(X_test)
print("Elastic Net RMSE: {:0.4f}".format(np.sqrt(mean_squared_error(y_test, preds_elascv))))
print("Elastic Net Lambda: {:0.4f}".format(elascv.alpha_))


# mse.append(np.sqrt(mean_squared_error(y_test, preds_elascv)))
#
#
# #RandomForest Regressor
# regr = RandomForestRegressor(n_estimators=1000, max_depth=2, random_state=0)
#
# regr.fit(X_train, y_train)
# preds_rfreg = regr.predict(X_test)
# print("Random Forest RMSE: {:0.4f}".format(np.sqrt(mean_squared_error(y_test, preds_rfreg))))
# mse.append(np.sqrt(mean_squared_error(y_test, preds_rfreg)))





#
#
# # from sklearn import decomposition
# # pca = decomposition.PCA(n_components=197)
# # pca.fit(X)
# #
# # plt.figure(1, figsize=(4, 3))
# # plt.clf()
# # plt.axes([.2, .2, .7, .7])
# # plt.plot(pca.explained_variance_, linewidth=2)
# # plt.axis('tight')
# # plt.xlabel('n_components')
# # plt.ylabel('explained_variance_')
# # plt.show()
#
#


## FINAL PREDICTIONS

# Validation Data import and preprocessing
data = pd.read_csv("test.xls")




#replace NA in ordinal columns with 0
for columns in ord_dict:
    data[columns].fillna(0, inplace=True)

#replace ordinal columns with numeric ordinal values
for columns in ord_dict:
    for keys in ord_dict[columns]:
        data[columns].replace(keys, ord_dict[columns][keys], inplace=True)

#categorical columns replaced with
for categories in categorical:
    data = create_dummies(data, categories)

#replace the age columns with age
data['age_built'] = data['YrSold'] - data['YearBuilt'] + 1
data['age_remodelled'] = data['YrSold'] - data['YearRemod/Add'] + 1
data.drop('YearBuilt', axis=1, inplace=True)
data.drop('YearRemod/Add', axis=1, inplace=True)





validation_data = data[selected_column_names]

validation_data = imputer.fit_transform(validation_data)
X_val= validation_data[:,:]
X_val = scalar.fit_transform(X_val)


#ElasticnNet Prediction Output
elascv = ElasticNetCV(l1_ratio = 0.3, alphas=alpha_range, cv=5, max_iter=10000, selection="random", n_jobs=-1)
elascv.fit(X, Y)

preds_elascv_test = elascv.predict(X_val)
preds_elascv_test = pd.DataFrame(elascv.predict(X_val))
preds_elascv_test.index += 1
preds_elascv_test.to_csv('Fake Data Scientists-Elastic Net prediction.csv', index=True, index_label='ID',
                         header=['Prediction'])


# #OLS Prediction Output
# lm = LinearRegression()
# lm.fit(X, Y)
# preds_ols_test = pd.DataFrame(lm.predict(X_val))
# preds_ols_test.index += 1
# preds_ols_test.to_csv('Fake Data Scientists-OLS prediction.csv', index=True, index_label='ID', header=['Prediction'])
