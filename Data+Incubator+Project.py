
# coding: utf-8

# ## Lets load our libraries and perform initial analysis of the main application dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:


train = pd.read_csv('application_train.csv').sort_values('SK_ID_CURR').reset_index(drop=True)
test = pd.read_csv('application_test.csv').sort_values('SK_ID_CURR').reset_index(drop=True)
#bureau = pd.read_csv('bureau.csv').sort_values(['SK_ID_CURR','SK_ID_BUREAU']).reset_index(drop = True)
#bureau_balance = pd.read_csv('bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True)
#credit = pd.read_csv('credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True)
#installments = pd.read_csv('installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True)
#cash = pd.read_csv('POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True)
#previous = pd.read_csv('previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True)


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


train['TARGET'].value_counts() # our data is imbalanced, only 8.78% data are those who defaulted. 


# In[6]:


train.describe()


# In[7]:


train.dtypes.value_counts()


# ###### There are 16 categorical variables which will need encoding

# In[8]:


#Lets observe missing values
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[9]:


missing_table = missing_values_table(train)


# # Lets make 2 visualizations (only) of the dataset for the challenge

# In[10]:


train['DAYS_BIRTH']= train['DAYS_BIRTH']*-1



# In[11]:


train['YEAR_BIRTH']=train['DAYS_BIRTH']/365
train['YEAR_BIRTH'].head()


# ## Lets see how age affects repayment 

# In[12]:



plt.figure(figsize = (8, 6))
sns.kdeplot(train.loc[train['TARGET']==1,'YEAR_BIRTH'],shade=True, label='target = 1')
sns.kdeplot(train.loc[train['TARGET']==0,'YEAR_BIRTH'],shade=True, label='target = 0')
                  


# ### The above graph shows younger people tend to default more than olders as the graph of defaulted customers are skewed to the left. Next we check gender map with year and create categories of age classes to see exactly which group is defaulting the most

# ### Lets dig deep and see what age categories are most likely to default 

# In[13]:


# Lets take Age information, target and gender into a separate dataframe
df_age = train[['TARGET', 'YEAR_BIRTH','CODE_GENDER']]

# Bin the age data
df_age['YEARS_BINNED'] = pd.cut(df_age['YEAR_BIRTH'], bins = np.linspace(20, 70, num = 10))
df_age.head(10)


# In[14]:


age_mean = df_age.groupby('YEARS_BINNED').mean()
age_mean


# In[17]:


sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
sns.barplot(age_mean.index.astype(str), 100 * age_mean['TARGET'], data =age_mean, palette='rocket')
#sns.barplot(age_mean.index.astype(str), df_age['CODE_GENDER'], data = df_age)
# Plot labeling
plt.xticks(rotation = 'vertical'); plt.xlabel('Age Group (years)'); plt.ylabel('Default percentage')
plt.title('Ranked Default age group categories');


# # Rough draft of the machine learning process for finding AOC

# In[18]:


categorical =train.select_dtypes(include = ['object'])


# In[20]:


categorical.apply(pd.Series.nunique,axis=0) #seeing how many unique categorical values are in each column 


# In[21]:


#making dummy variable transformations
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)


# In[22]:


print(train.shape)
print(test.shape)


# In[23]:


y = train['TARGET']


# In[28]:


#lets align
train, test = train.align(test, join= 'inner', axis=1)
print('Training data shape is ',(train.shape))
print('Testing data shape is ',(test.shape))


# In[29]:


#putting target back in training data
train['TARGET']= y

print(train.shape)
print(test.shape)


# In[31]:


from sklearn.preprocessing import MinMaxScaler, Imputer

train_1 = train.copy()
test_1 = test.copy()


# In[32]:


test_1.shape


# In[35]:


x = train.drop(['TARGET'], axis=1) #training data withot target


# In[37]:


imputer = Imputer(strategy = 'median')
imputer.fit(x)
x =imputer.transform(x)


imputer = Imputer(strategy = 'median')
imputer.fit(test)
test =imputer.transform(test)


# In[38]:


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(x)
x = scaler.transform(x)

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(test)
test = scaler.transform(test)


# ## Lets bulid our first "Naive" model without any feature selection

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


logistic_reg = LogisticRegression(C=0.00001)


# In[41]:


logistic_reg.fit(x,y)


# In[42]:


prediction_logistic = logistic_reg.predict_proba(test)[:,1]


# In[ ]:


# Submission dataframe
submit = test_1[['SK_ID_CURR']]
submit['TARGET'] = prediction_logistic

submit.shape


# In[ ]:


# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index = False)


# In[43]:


from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, n_jobs = -1)


# In[44]:


# Train on the training data
random_forest.fit(x, y)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
#feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[:, 1]

