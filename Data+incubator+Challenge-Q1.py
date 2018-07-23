
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[4]:


data = pd.read_excel('ccrb.xlsx')


# In[5]:


data.head()


# How many unique complaints (identified by 'UniqueComplaintId') with complete information (i.e. there are no missing values) appear in the dataset?

# In[62]:


unique = data['UniqueComplaintId'].nunique() #no missing values
print(unique)


# In[33]:


data['Borough of Occurrence'].value_counts(dropna=True)


# In[32]:


total_borough = data['Borough of Occurrence'].value_counts().sum()
print(data['Borough of Occurrence'].value_counts(dropna=True)['Brooklyn']/total_borough)


# How many complaints per 100k residents were there in the borough with the highest number of complaints per capita resulting from incidents in 2016? Find the 2016 population estimates of each borough on Wikipedia. Ignore complaints from "Outside NYC". 
# For this question, only consider unique complaints with complete information.

# In[79]:


unique_ids = data.drop_duplicates(subset = ['UniqueComplaintId']) #creating a set of unique complaint Ids only


# In[81]:


year_data_unique2016 = unique_ids[data['Incident Year']==2016] # filtering year 2016 with unique values


# In[99]:


year_data_unique2016.groupby('Borough of Occurrence').count()
#borough_2016


# In[97]:





# In[100]:


#highest per capita compplaint is in Bronx with 0.53
print('The number of complaints per 100k residents is in Bronx with a value of ', borough_2016['UniqueComplaintId'][0]/100)


# Calculating average number of years it takes for a complaint to be closed? 
# For this question, only consider unique complaints with complete information.

# In[65]:


unique_ids = data.drop_duplicates(subset = ['UniqueComplaintId'])


# In[75]:


diff_year_mean = unique_ids['Close Year']- unique_ids['Received Year']
diff_year_mean.mean()


# #### Complaints about stop and frisk have been declining. Use linear regression from the year complaints about stop and frisk peaked through 2016 (inclusive) to predict how many stop and frisk incidents in 2018 will eventually lead to a complaint. For this question, only consider unique complaints with complete information. Remember that the count of complaints must be an integer (round to nearest).

# In[107]:


# Lets first look at the plot of decline
plt.figure(figsize=(12,6))
sns.countplot(x='Received Year',data=unique_ids,hue='Complaint Contains Stop & Frisk Allegations',palette='viridis')
# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ###### The Green bar peaked in 2007 suggesting this is our starting point for linear regression to 2016
# 
# Our y variable is count of True and X is received year from 2007 to 2016. Lets create the dataframe

# In[115]:


unique_ids.groupby(['Complaint Contains Stop & Frisk Allegations','Received Year']).count()


# In[162]:


dictionary = {'Years':[0,1,2,3,4,5,6,7,8,9],'Count':[2557,2285,2276,1905,1655,1493,1258,1003,883,736]}
#labels = ['Year','Count of True']
regress = pd.DataFrame.from_dict(data=dictionary)


# In[164]:


sns.lmplot(x='Years',y='Count',data=regress)


# In[168]:


from sklearn.linear_model import LinearRegression
#model =LinearRegression()
#model.fit(regress['Years'],regress['Count'])


# In[172]:


import statsmodels.api as sm
x= regress['Years']
x=sm.add_constant(x)
y= regress['Count']
model = sm.OLS(y,x).fit()
model.summary()


# Our prediction for 2018 is 2550.6727 - 210.1273 * 11 = 239 (round off)

# ##### Calculate the chi-square test statistic for testing whether a complaint is more likely to receive a full investigation when it has video evidence. 
# For this question, only consider unique complaints with complete information

# In[173]:


import scipy.stats as stats


# In[174]:


cont = pd.crosstab(unique_ids["Is Full Investigation"],unique_ids["Complaint Has Video Evidence"])
cont


# In[176]:


stats.chi2_contingency(cont)

