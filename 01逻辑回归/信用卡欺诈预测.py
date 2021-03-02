#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv("creditcard.csv")
data.head()


# In[10]:


#对class中不同类别的数量进行可视化展示
#pd.value_counts:  Compute a histogram(直方图) of the counts of non-null values.
    #values: ndarray
    #sort : boolean, default True Sort by values
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel('Class')
plt.ylabel('Frequency')


# In[21]:


#由于Amount列数据跨度较大，故对其进行标准化处理
from sklearn.preprocessing import StandardScaler

#用values方法将Series对象转化成numpy的ndarray，再用ndarray的reshape方法
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis=1)
data.head()


# ## 下采样
# 使0和1的数据一样少

# In[40]:


#对数据集进行下采样
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

#统计Class=1的数量和索引
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
#统计Class=0的索引
normal_indices = data[data.Class == 0].index

#根据Class=1的数量随机选择Class=0的索引，并转化为ndarray格式
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

#将筛选出的数据整合到一起
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = data.iloc[under_sample_indices, :]

#X_undersample：特征，y_undersample：目标
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# ## 交叉验证
# 将原始数据进行切分，用于调参和预测

# In[45]:


from sklearn.model_selection import train_test_split

#whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

#Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
X_undersample, y_undersample, test_size = 0.3, random_state = 0)

print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))


# ### 模型评估标准
# 当样本不均衡时，精度会存在很大的误差。故引入recall的评估标准。

# In[73]:


#Recall = TP/(TP+FN)
from sklearn.linear_model import LogisticRegression
#KFold：切分训练集
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report


# In[77]:


def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)
    
    #设置正则惩罚参数
    c_param_range = [0.01, 0.1, 10, 100]
    
    #range(start, stop[, step]),创建一个4*2的矩阵，并指定列名
    #results_table保存每个C参数下的recall score平均值，并由其计算出最大recall score的C参数
    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns=['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range
    
    j = 0 
    #选择不同的正则惩罚参数
    for c_param in c_param_range:
        print('-------------------------')
        print('C parameter: ', c_param)
        print('-------------------------')
        print('')
        
        recall_accs = []
        
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        #同时列出数据和数据下标
        #sequence -- 一个序列、迭代器或其他支持迭代对象。start -- 下标起始位置。
        for iteration, indices in enumerate(fold.split(x_train_data)):
            
            #Call the logistic regression model with a certain C parameter, and choose the type of 正则惩罚
            lr = LogisticRegression(C = c_param, penalty='l1', solver='liblinear')
            
            #the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            
            #Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)
            
            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ' : recall score = ', recall_acc)
        # The mean value of those recall scores is the metric we want to save and get hold of.
        #.ix is deprecated. Please use
        #.loc for label based indexing or
        #.iloc for positional indexing
        results_table.loc['Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')
        
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c


# In[79]:


best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)


# ## 混淆矩阵
# * 通过混淆矩阵可以计算出精度，Recall的值。

# In[83]:


def plot_confusion_matrix(cm, classes,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[87]:


#利用下采样模型对下采样数据进行预测
import itertools
lr = LogisticRegression(C = best_c, penalty = 'l1', solver='liblinear')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

#Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

#Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[89]:


#利用下采样模型对所有数据进行预测
lr = LogisticRegression(C = best_c, penalty = 'l1', solver="liblinear")
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[91]:


best_c = printing_Kfold_scores(X_train, y_train)


# 会发现其Recall score并不理想

# In[92]:


lr = LogisticRegression(C = best_c, penalty = 'l1', solver="liblinear")
lr.fit(X_train,y_train.values.ravel())
y_pred_undersample = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# ## 阈值对逻辑回归的影响

# In[97]:


lr = LogisticRegression(C = 0.01, penalty= 'l1', solver="liblinear")
#使用下采样数据建立模型
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
#使用下采样数据进行预测
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10,10))

j = 1
for i in thresholds:
    #设定阈值
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
    
    plt.subplot(3, 3, j)
    j += 1
    
    #Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    
    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    
    #Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 


# # 过采样
# ## SMOTE算法生成新的样本点

# In[4]:


import pandas as pd
#需要安装imblearn库
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[6]:


credit_cards = pd.read_csv('creditcard.csv')

columns = credit_cards.columns
#移除class，获取特征列
features_columns = columns.delete(len(columns)-1)

features = credit_cards[features_columns]
labels = credit_cards['Class']


# In[7]:


#将原始数据分为训练集和测试集
features_train, feature_test, labels_train, labels_test = train_test_split(features, labels,
                                                                          test_size=0.2, random_state=0)


# In[ ]:


#在原始数据的训练集中产生过采样数据
oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_smaple(features_train, labels_trian)


# In[ ]:


#观察是否成功产生过采样样本点
len(os_labels[os_labels==1])


# In[ ]:


#对过采样的训练集进行训练,选出最优参数
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
best_c = printing_Kfold_scores(os_features, os_labels)


# In[ ]:


#使用逻辑回归对过采样训练集进行预测
lr = LoigisticRegression(C = best_c, penalty = 'l1')
lr.fit(os_features, os_labels.values.ravel())
#使用测试集中的原始数据进行预测
y_pred = lr.predict(features_test.values)

#计算测试集中的原始数据的混淆矩阵
cnf_matrix = confusion_matrix(labels_test, y_pred)
#确定浮点数的位数
np.set_printoptions(precision=2)

print('Recall metric in the testing dataset:', cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[ ]:




