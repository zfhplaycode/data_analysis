#!/usr/bin/env python
# coding: utf-8

# ## 判断泰坦尼克号乘客能否获救

# In[1]:


import pandas
titanic = pandas.read_csv("titanic_train.csv")
#titanic.head(5)
print(titanic.describe())


# ## 观察数据可知，年龄这一特征有缺失值，故需要进行数值的填充

# In[2]:


#使用均值填充数据
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe())


# ## 性别特征为字符，故需要转化为可计算的数值

# In[7]:


#性别特征包含的内容
print(titanic["Sex"].unique())

#将性别映射为数值
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#titanic["Sex"] = titanic["Sex"].replace["male", 0]


# ### 上船地点
# * 有缺失值，需要进行填充
# * 是字符类型，需要转换为数值类型

# In[8]:


print(titanic["Embarked"].unique())
#填充缺失值
titanic["Embarked"] = titanic["Embarked"].fillna("S")
#数值映射
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


# ## 使用线性回归模型

# In[9]:


from sklearn.linear_model import LinearRegression
#使用KFold进行数据分割以进行交叉验证
from sklearn.model_selection import KFold

#The columns we'll use to predict the target
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#Initialize our algorithm class
alg = LinearRegression()

# Generate cross validation folds for the titanic dataset.  It return the row indices
#corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
#n_splits : int, default=5、
    #Number of folds. Must be at least 2
kf = KFold(n_splits=3, random_state=1, shuffle=True)

predictions = []
#将全部数据集分成三份，训练集中用其中的两份(train)交叉验证另外一份(test)

for train, test in kf.split(titanic):
    #获取训练集中的特征数据
    train_features = (titanic[features].iloc[train])
    #获取训练集中的结果
    #iloc：根据索引下标定位数据
    train_target = titanic["Survived"].iloc[train]
    #根据训练集中的特征值和目标值构建模型
    alg.fit(train_features, train_target)
    #使用模型对测试集中的特征数据进行预测
    test_predictions = alg.predict(titanic[features].iloc[test])
    predictions.append(test_predictions)


# In[10]:


#对测试集的预测值进行处理得出精度
import numpy as np
# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

#Map predictions to outcomes(only possible outcomes are 1 or 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
#计算测试集预测值的精度
accuracy = len(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)


# ## 使用逻辑回归进行预测

# In[11]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
#Initialize algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = model_selection.cross_val_score(alg, titanic[features], titanic["Survived"], cv=3)
#Tak the mean of the socre (each fold have their socres)
print(scores.mean())


# In[12]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#Initialize our algorithm with the default paramters
#n_estimators is the number of trees we want to make
#min_samples_split is the minimum number of rows we need to make a split
#min_samples)leaf is the minimum number of samples we can have at the palce where a tree branch ends(the 
#bottom points of the treee)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
#Computer the accracy socre for all the cross validation folds .(much simpler than before we did)
kf = KFold(n_splits=3, random_state=1, shuffle=True)
scores = cross_val_score(alg, titanic[features], titanic["Survived"], cv=kf.split(titanic))

#take the mean of the scores(because we have one for each fold)
print(scores.mean())


# In[ ]:


print(kf.split(titanic))


# ## 发现上述结果并不是很理想，故需要进行参数优化

# In[14]:


alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
#Computer the accracy socre for all the cross validation folds .(much simpler than before we did)
kf = KFold(n_splits=3, random_state=1, shuffle=True)
scores = cross_val_score(alg, titanic[features], titanic["Survived"], cv=kf.split(titanic))
#take the mean of the scores(because we have one for each fold)
print(scores.mean())


# ## 通过对原始数据进行处理，优化预测结果

# In[17]:


#Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

#The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))


# In[21]:


import re

#A function to get the title from a name
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters,
    #and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #if the title exists,extract and return it
    if title_search:
        return title_search.group(1)
    return ""

#Get all the titles and print how often each one occurs
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

#Verify that we converted everything
print(pandas.value_counts(titles))

#Add in the title column
titanic["Title"] = titles


# ## 判断所有特征的重要程度

# In[27]:


import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

#Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[features], titanic["Survived"])

#Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

#Plot the scores. See how "Pclass", "Sex", "Title", and "Fare" are best
plt.bar(range(len(features)), scores)
plt.xticks(range(len(predictors)), features, rotation='vertical')
plt.show()

#Pick only the four best features
features = ["Pclass", "Sex", "Fare", "Title"]
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
#Computer the accracy socre for all the cross validation folds .(much simpler than before we did)
kf = KFold(n_splits=3, random_state=1, shuffle=True)
scores = cross_val_score(alg, titanic[features], titanic["Survived"], cv=kf.split(titanic))
#take the mean of the scores(because we have one for each fold)
print(scores.mean())


# ## 使用多个分类器，使用继承算法获取结果

# In[36]:


from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

#The algorithms we want to ensemble.
#We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

#Initialize the crossvalidation folds
kf = KFold(n_splits=3, random_state=1, shuffle=True)

predictions = []
for train, test in kf.split(titanic):
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    
    #Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        #Fit the algorithm on the training data
        alg.fit(titanic[features].iloc[train, :], train_target)
        #Select and predict on the test fold
        #The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error
        #predict_proba(self, X):Predict class probilities for X
        test_predictions = alg.predict_proba(titanic[features].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    #Use a simple ensembling scheme -- just average the predictions to get the final classification
    test_predictions = (full_test_predictions[0] + full_test_predictions[1])/2
    #Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

#Pt all the prediction together into one array
predictions = np.concatenate(predictions, axis=0)
#Compute accuracy by comparing to the training data.
accuracy = len(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)


# In[31]:


print(help(GradientBoostingClassifier.predict_proba))

