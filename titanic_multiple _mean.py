# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:34:37 2019

@author: Sathya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the data
train = pd.read_csv('C:\\python\\logistic\\titanic_train.csv')
train.info()
train.head()
print("Total no of passengers in the data set: ",len(train.index))

#Analyze the data by plotting
sns.countplot(x='Survived',data=train)

#Here we can see that more people died than survived. so we take Survived as y(dependent variable) 
#and plot against each independent variable to see their dependencies
sns.countplot(x='Survived',data=train,hue='Sex')
sns.countplot(x='Survived',data=train,hue='Pclass')
sns.countplot(x='Survived',data=train,hue='Age')
sns.countplot(x='Survived',data=train,hue='SibSp')
sns.countplot(x='Survived',data=train,hue='Cabin')
#Data cleaning
#cabin have 687 NAs out of 800 records. so no need to take it.
#'Embarked','Name','Ticket','Parch','Fare' doesnt have much effect on the survival. so drop

train.drop(['Embarked','Name','Ticket','Parch','Fare','Cabin'],axis =1,inplace = True)
train.info()
train.isna().sum()

# Age has more NAs. We can simply replace NAs with mean of Age but it will give less accuracy
#instead we can take into account both gender and Pclass and find the mean value
sns.boxplot(x = 'Sex',y='Age',data = train)

train.groupby(['Pclass','Sex']).mean()

# We'll use these average age values to impute based on Pclass and Sex for Age.
def imputeage(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age): 
        
        if (Pclass == 1 and Sex == 'female'):
            return 34
        elif (Pclass == 1 and Sex == 'male'):
            return 41
        elif (Pclass == 2 and Sex == 'female'):
            return 28
        elif (Pclass == 2 and Sex == 'male'):
            return 30
        elif (Pclass == 3 and Sex == 'female'):
            return 21
        else:
            return 26
    else:   
         return Age
     
        
train['Age'] = train[['Age','Pclass','Sex']].apply(imputeage,axis = 1)

#replacing Sex column with binary values
d = {'male':0,'female':1}
train['Sex'] = train['Sex'].map(d)
train.head()

# Building a Logistic Regression model by splitting our data into a training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train.drop('Survived',axis =1),train['Survived'],test_size = 0.2)

#Training and Predicting
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

#We can check precision,recall,f1-score using classification report!
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))


    

    
    
    
    
    