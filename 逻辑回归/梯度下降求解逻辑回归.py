#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# # The data

# 建立一个逻辑回归模型预测一个学生是否被大学录取。假设你是一个大学的系管理员，要根据两次考试的结果来决定每个申请人的录取机会。你有以前的申请人的历史数据，可以用它作为逻辑回归的训练集。对于每一个培训的例子，由两个申请人的考试分数和录取决定，为了做到这一点，将建立一个分类模型，根据考试成绩估计入学的概率。

# In[6]:


#三大件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import os
path = 'LogiReg_data.txt'
#读入数据，指定列名
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
pdData.head()


# In[8]:


pdData.shape


# In[9]:


#将被录取和不被录取的数据筛选出来
positive = pdData[pdData['Admitted']==1]
#print(pdData['Admitted']==1)
#pint(positive)
negative = pdData[pdData['Admitted']==0]

#指定画图区域的大小
fig, ax = plt.subplots(figsize=(10,5))
#s:散点的大小
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# # The logistic regression
# * 目标：建立分类器（求解三个参数$\theta_{0},\theta_{1},\theta_{2}$）
# * 设定阈值，根据阈值判断录取结果

# ## 要完成的模块
# * sigmoid：映射到概率的函数
# * model：返回预测结果值
# * cost：根据参数计算损失
# * gradient：计算每个参数的梯度方向
# * descent：进行参数更新
# * accuracy：计算精度

# ### sigmoid 函数
# ## <center>$g(z)=\frac{1}{1+e^{-z}}$

# In[10]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[11]:


nums = np.arange(-10, 10, step=1)#create a vector containing 20 equally spaced values from -10 to 10
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(nums, sigmoid(nums), 'r')


# ### sigmoid()
# * $g:\mathbb{R} \to [0,1]$
# * $g(0)=0.5$
# * $g(- \infty)=0$
# * $g(+ \infty)=1$

# In[12]:


#得到（100,1）的矩阵
def model(X, theta):
    #np.dot(A,B) 返回矩阵A*B
    return sigmoid(np.dot(X, theta.T))


# $$
# \begin{array}{ccc}
# \begin{pmatrix}\theta_{0} & \theta_{1} & \theta_{2}\end{pmatrix} & \times & \begin{pmatrix}1\\
# x_{1}\\
# x_{2}
# \end{pmatrix}\end{array}=\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}
# $$

# In[15]:


#Insert column into DataFrame at specified location.
pdData.insert(0, 'Ones', 1)
#print(help(pdData.insert))
#pdData.head()

#set X(training data) and y(target variable)
orig_data = pdData.values
cols = orig_data.shape[1]
X = orig_data[:, 0:cols-1]
y = orig_data[:, cols-1:cols]
#print(X)

#convert to numpy arrays and initalize the parameter array theta
#X = np.matrix(X.values)
#y = np.matrix(data.iloc[:, 3:4].values) #np.array(y.values)
#Return a new array of given shape and type, filled with zeros
theta = np.zeros([1,3])


# In[18]:


print(X[:5])
pdData.head()


# In[19]:


y[:5]


# In[20]:


theta


# In[21]:


X.shape, y.shape, theta.shape


# ### 损失函数
# 将对数似然函数转化为求梯度下降的函数
# 
# $$
# D(h_\theta(x), y) = -y\log(h_\theta(x)) - (1-y)\log(1-h_\theta(x))
# $$
# 求平均损失
# $$
# J(\theta)=\frac{1}{n}\sum_{i=1}^{n} D(h_\theta(x_i), y_i)
# $$

# In[22]:


def cost(X, y, theta):
    #np.multiply()：对应元素相乘，得到(100,1)的矩阵
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1-y, np.log(1-model(X, theta)))
    #np.sum()：所有元素的和
    return np.sum(left-right)/(len(X))


# In[23]:


cost(X, y, theta)


# ### 计算梯度
# 
# 
# $$
# \frac{\partial J}{\partial \theta_j}=-\frac{1}{m}\sum_{i=1}^n (y_i - h_\theta (x_i))x_{ij}
# $$

# In[24]:


def gradient(X, y, theta):
    #新的梯度
    grad = np.zeros(theta.shape)
    #计算误差
    error = (model(X, theta)-y).ravel()
    #theta.ravel()扁平化操作
    for j in range(len(theta.ravel())):
        #np.multiply(A,B)：A,B为同型矩阵，对应元素相乘
        #更新j,则取X的第J列
        term = np.multiply(error, X[:,j])
        #len()：Return the number of items in a container(list, dic, touple)
        grad[0, j]=np.sum(term)/len(X)
    
    return grad


# ### Gradient descent
# 
# 比较三种不同梯度下降方法

# In[33]:


#根据迭代次数结束
STOP_ITER = 0
#根据损失值结束
STOP_COST = 1
#根据梯度值结束
STOP_GRAD = 2

def stopCriterion(Type, value, threshold):
    #设定三种不同的停止策略
    if Type==STOP_ITER: return value > threshold
    if Type==STOP_COST: return abs(value[-1]-value[-2]) < threshold
    #np.linalg.norm：求范数
    if Type==STOP_GRAD: return np.linalg.norm(value) <threshold


# In[26]:


import numpy.random
#洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    
    return X, y


# In[27]:


import time
#batchSize：样本量
def descent(data, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解
    
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)#计算的梯度
    costs = [cost(X, y, theta)]#损失值
    
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize #取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)#重新洗牌
        theta = theta - alpha*grad#参数更新
        costs.append(cost(X, y, theta))#计算新的损失
        i += 1
        
        if stopType == STOP_ITER: value = i
        elif stopType == STOP_COST: value = costs
        elif stopType == STOP_GRAD: value = grad
        
        if stopCriterion(stopType, value, thresh): break
    
    return theta, i-1, costs, grad, time.time() - init_time
            


# In[28]:


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter_num, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += "data - learning rate: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1: strDescType = "Stopchastic"
    else: strDescType = "Mini-batch({})".format(batchSize)
    name += strDescType + "descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("**{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter_num, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    #np.arange返回一个有终点和起点的固定步长的排列
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta    


# In[29]:


#选择的梯度下降方法是基于所有样本的
n = 100
runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)


# ### 根据损失值停止
# 设定阈值1E-6，差不多需要110000此迭代

# In[34]:


runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)


# ### 根据梯度变化停止
# 设定阈值0.05，差不多需要40 000次迭代

# In[35]:


runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)


# ## 对比不同的梯度下降方法
# ### Stochastic descent

# In[36]:


runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)


# 不稳定，将学习率调小

# In[37]:


runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)


# 速度快，但稳定性差，需要很小的学习率

# ### Min-batch descent

# In[38]:


runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)


# 浮动比较大，尝试对数据进行标准化，将数据按其属性（按列进行）减去其均值，然后除以其方差。最后得到的结果是，对每个属性来说所有的数据都聚集在0附近，方差值为1

# In[39]:


from sklearn import preprocessing as pp


# In[42]:


#orig_data[:5]
scaled_data = orig_data.copy()
scaled_data[:,1:3] = pp.scale(orig_data[:, 1:3])

runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)


# 结果有了明显改善，原始数据只能达到0.61，在这里能达到0.38，突出了数据预处理的重要性

# In[43]:


runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)


# 更多的迭代次数会使得损失值下降的更多

# In[49]:


theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)


# 随机梯度波动过大

# In[51]:


runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)


# ## 精度

# In[52]:


#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]


# In[58]:


scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
correct = [1 if (a==1 and b==1) or (a==0 and b==0) else 0 for (a, b) in zip(predictions, y)]
#map() 会根据提供的函数对指定序列做映射
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))


# In[ ]:





# In[ ]:





# In[ ]:




