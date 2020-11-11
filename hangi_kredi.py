# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 00:10:52 2020

@author: Hamza
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# dataset import edildi
dataset = pd.read_csv("term-deposit-marketing-2020.csv")

# input ve output ayrıldı
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# strings to int
# Normally i wouldnt do such repatative code like below but im having
# a problem hich i couldnt solve while using multiple columns
labelencoder = LabelEncoder() 
X["default"] = labelencoder.fit_transform(X["default"])
X["housing"] = labelencoder.fit_transform(X["housing"])
X["loan"] = labelencoder.fit_transform(X["loan"])
y = labelencoder.fit_transform(y)


# transforming categorical data with pandas
X_dummy_variable=pd.get_dummies(X, columns=[
        "job","marital", "education", "contact", "month"])

# avoiding the dummy variable trap
del X_dummy_variable["month_jan"]
del X_dummy_variable["job_unknown"]
del X_dummy_variable["marital_divorced"]
del X_dummy_variable["education_unknown"]
del X_dummy_variable["contact_unknown"]


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

scores = cross_val_score(classifier, X_dummy_variable, y, cv=5)
print("5-Folf cross val accuracy score is: {:.2f}%".format(scores.mean()*100))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset_for_importance = pd.read_csv("term-deposit-marketing-2020.csv")


labelencoder = LabelEncoder() 
dataset_for_importance["default"] = labelencoder.fit_transform(
        dataset_for_importance["default"])
dataset_for_importance["housing"] = labelencoder.fit_transform(
        dataset_for_importance["housing"])
dataset_for_importance["loan"] = labelencoder.fit_transform(
        dataset_for_importance["loan"])
dataset_for_importance["y"] = labelencoder.fit_transform(
        dataset_for_importance["y"])


# transforming categorical data with pandas
dataset_for_importance=pd.get_dummies(dataset_for_importance, columns=[
        "job","marital", "education", "contact", "month"])

# avoiding the dummy variable trap
del dataset_for_importance["month_jan"]
del dataset_for_importance["job_unknown"]
del dataset_for_importance["marital_divorced"]
del dataset_for_importance["education_unknown"]
del dataset_for_importance["contact_unknown"]
#get correlations of each features in dataset
corrmat = dataset_for_importance.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset_for_importance[
        top_corr_features].corr(),annot=True,cmap="RdYlGn")

# Duration is the most important parameter as it looks









