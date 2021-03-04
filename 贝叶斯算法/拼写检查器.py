#!/usr/bin/env python
# coding: utf-8

# ## 贝叶斯拼写检查器

# In[2]:


#collections：将元素数量统计，然后计数返回一个字典，键为元素，值为元素个数
import re, collections

#获取文档中所有的单词（转化为小写）
def words(text): return re.findall('[a-z]+', text.lower())

#统计每个单词出现的次数，对于文档中没有的单词
def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(open('big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def editsl(word):
    n = len(word)
    #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
                #word = "hello"  print([word[0:i]+word[i+1:] for i in range(5)])
                #out：['ello', 'hllo', 'helo', 'helo', 'hell']
    return set([word[0:i]+word[i+1:] for i in range(n)] + 
               #out：['ehllo', 'hlelo', 'hello', 'helol']
               [word[0:i] + word[i+1]+word[i]+word[i+2:] for i in range(n-1)] +
               #替换
               [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] +
               #插入
               [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet]
    )

def known_edits2(word):
    #e2为根据editsl的返回值任意组合的单词
    return set(e2 for e1 in editsl(word) for e2 in editsl(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    #从左向右，满足一个known就不会继续执行
    candidates = known([word]) or known(editsl(word)) or known_edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])

 


# In[3]:


correct('thi')


# ### 上述代码通过简单的字符替换/删除/替换等产生新的单词后检索在文件中是否存在该新产生的单词

# ### 求解：argmaxc P(c|w) -> argmaxc P(w|c) P(c) / P(w) ###
# 
# * P(c), 文章中出现一个正确拼写词 c 的概率, 也就是说, 在英语文章中, c 出现的概率有多大
# * P(w|c), 在用户想键入 c 的情况下敲成 w 的概率. 因为这个是代表用户会以多大的概率把 c 敲错成 w
# * argmaxc, 用来枚举所有可能的 c 并且选取概率最大的

# In[35]:





# In[ ]:





# In[ ]:




