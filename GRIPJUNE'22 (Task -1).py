#!/usr/bin/env python
# coding: utf-8

# # Tanya Aggarwal
# 
# # GRIP JUNE'22

# # Prediction of student's study hours by linear regression
# 

# # LINEAR REGRESSION:- 
# Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.
# 
# #simple linear regression:- In this task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studies.
# 
# As only two variables are involved i.e 1 dependent and 1 independent variable ,this is Simple linear regresion
# 
# 

# In[2]:


#importing Libraries
import pandas as pd                  #for manipulationg and analyse the data
import numpy as np                   #for numerical data
import matplotlib.pyplot as plt      #for plotting the data
#%matplotlib inline                   # for inline plotting(below the commands)


# In[3]:


#Importing Data
url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
df1=pd.read_csv(url)                             #to read the data  
print("Data imported successfully")
print(df1)


# In[4]:


#if we want to print the limited values
df1.head(2)           #to get upper values
df1.tail(2)            #to get below values


# In[5]:


#Plotting the distribution of scores
df1.plot(x='Hours', y='Scores',style='o')        #ploptting(we can change style *,1)
plt.title('Hours vs Percentage')                 #title of graph
plt.xlabel('Hours Studied')                      #label  x axis
plt.ylabel('Percentage Score')                   #label y axis
plt.show()


# In[6]:


#As we can see the above graph, we can conclude that as hours studied increases ,percentafe increases. So, we can say that 
#there's a positive linear relation between two variables


# # Preparing the data

# In[7]:


#Step-1:-In the next step we're going to divide data into "attributes"(inputs) and"labels"(Outputs)
#Independent and Deoendent features
x=df1.iloc[:,:-1].values             #iloc() function enables us to select a particular cell of the dataset
#print(x)
y=df1.iloc[:,-1].values 
print(y)


# In[9]:


#step 2-Split the data into training and testing sets by Using Scikit -learn's built in  train_test_split()method
# train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets:
#for training data and for testing data. With this function, you don't need to divide the dataset manually.
#By default, Sklearn train_test_split will make random partitions for the two subsets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)   #,test_size=0.2=20% for testing


# # Training the Algorithm
# 

# In[10]:


#step-3:- Train the Algorithm

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("Training Done")


# In[11]:


#step 4:- Plotting the training line
line = regressor.coef_*x+regressor.intercept_

#Plotting the test data
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# # Prediction

# In[12]:


print(x_test)                       #testing data in hours
y_pred=regressor.predict(x_test)    #predicting the scores
print(y_pred)


# In[13]:


#Comparing Actual Vs prediction
df2=pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
df2


# In[14]:


#Checking own data
hours=9.25
own_pred=regressor.predict([[hours]])
print('Predicted score if student study 9.25 hours/day')
print('No. of hours={}'.format(hours))
print('Predicted Score={}'.format(own_pred[0]))


# In[15]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




