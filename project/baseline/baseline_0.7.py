#!/usr/bin/env python
# coding: utf-8

# # 2019看山杯专家发现算法大赛 Baseline(0.7632)

# ## 1. 赛题分析

# 我们先给出最简单的赛题说明：将一个问题Q推荐给用户U，计算用户U会回答这个问题Q的概率。
# 
# 具体来说，比赛提供了问题信息（question_info_0926.txt，可以为问题Q创建额外的特征信息）、用户画像（member_info_0926.txt，可以为用户U创建额外的特征信息）、回答信息（answer_info_0926.txt，可以同时为问题Q和用户U创建额外的特征信息）
# 
# 数据集字段详细介绍，参考官网数据集说明（非常详细）ref:https://www.biendata.com/competition/zhihu2019/data/

# ## 2. 数据分析
# 

# ### 2.1 用户数据集和问题数据集

# In[1]:


# In[7]:
from pandas import DataFrame

import pandas as pd
user_info = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
for col in user_info.columns:
    print(col, len(user_info[col].unique()))
    
question_info = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
for col in question_info.columns:
    print(col, len(question_info[col].unique()))

# 从上面的数据分析可以看出，用户数据中有21个特征，其中5个特征（创作关键词、创作数量等级、创作热度等级、注册类型、注册平台）在数据集中只有一个取值，说明这5个特征是完全无用的，可以直接去掉。

# ### 2.2 数据集合并

# 为了分析上述两个数据集中的特征是否有对预测结果有影响（或者说这些特征是否是有区分度的强特征），我们首先将这两个数据集和训练集（invite_info_0926.txt）合并, 然后通过图表来对部分特征进行分析。

# In[8]:


train = pd.read_csv('invite_info_0926.txt', header=None, sep='\t')
train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
train = pd.merge(train, user_info, how='left', on='用户id')
train = pd.merge(train, question_info, how='left', on='问题id')
print(train.columns)

# **性别特征：**性别特征有三个类别，分别是 男性，女性和未知，下面的柱状图可以看出，男性和女性分布非常相似，未知的分布相比较之下有 较大的区别，显然，该特征具有较好的区分度。

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
# % matplotlib inline
sns.set_style('whitegrid')

sns.countplot(x='性别',hue='是否回答',data=train)

# **访问频率特征：**该特征总计5中类别，[weekly, monthly, daily, new, unknown]，从下面的柱状图可以看出，不同的类别具有完全不同的分布，这特征显然是一种很有区分度很强的特征。

# In[10]:


sns.countplot(x='访问评率',hue='是否回答',data=train)

# **用户二分类特征a：**该特征是二分类特征，下图表示该特征具备很好的区分度（剩下的二分类和多分类特征也是同理，不赘述）。

# In[11]:


sns.countplot(x='用户二分类特征a',hue='是否回答',data=train)

# **盐值：**我们先对盐值进行分桶，然后查看不同区间盐值的分布情况。下图表示不同区间盐值的用户具有很有的区分度，在处理这个特征时，至于是否分桶，如何通过更加详细的数据分析自由发挥，给出的baseline对该特征未做处理。

# In[12]:


def trans(x):
    if x <= 0:
        return x
    if 1 <= x <= 10:
        return 1
    if 10 < x <= 100:
        return 2
    if 100 < x <= 200:
        return 3
    if 200 < x <= 300:
        return 4
    if 400 < x <= 500:
        return 5
    if 500 < x <= 600:
        return 6
    if x > 600:
        return 7
train['盐值'] = train['盐值'].apply(lambda x: trans(x))
sns.countplot(x='盐值',hue='是否回答',data=train)

# **时间：**数据集中的时间都采用“D×-H×”的格式，D代表天数，H代表小时，我们需要将这一个特征转化为两个特征，天和小时。

# ## 3. 数据处理
# 

# 从这一部分开始都是baseline的全部代码（前面仅仅是简单的数据分析，不作为baseline的代码）。

# In[1]:


import pandas as pd

# 导入数据
user_info = pd.read_csv('./data/member_info_0926.txt', header=None, sep='\t')
question_info = pd.read_csv('./data/question_info_0926.txt', header=None, sep='\t')
train = pd.read_csv('./data/invite_info_0926.txt', header=None, sep='\t')
test = pd.read_csv('./data/invite_info_evaluate_1_0926.txt', header=None, sep='\t')

user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
user_info  = user_info.drop(['创作关键词','创作数量等级','创作热度等级','注册类型','注册平台'], axis=1)
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']

train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
train = pd.merge(train, user_info, how='left', on='用户id')
train = pd.merge(train, question_info, how='left', on='问题id')

test.columns = ['问题id', '用户id', '邀请创建时间']
test = pd.merge(test, user_info, how='left', on='用户id')
test = pd.merge(test, question_info, how='left', on='问题id')

# 数据合并
data = pd.concat([train, test], axis=0, sort=True)

# 用于保存提交结果
result_append = data[['问题id', '用户id', '邀请创建时间']][train.shape[0]:]

data['邀请创建时间-day'] = data['邀请创建时间'].apply(lambda x:x.split('-')[0].split('D')[1])
data['邀请创建时间-hour'] = data['邀请创建时间'].apply(lambda x:x.split('-')[1].split('H')[1])

data['问题创建时间-day'] = data['问题创建时间'].apply(lambda x:x.split('-')[0].split('D')[1])
data['问题创建时间-hour'] = data['问题创建时间'].apply(lambda x:x.split('-')[1].split('H')[1])

# 删除的特征并非不重要，相反这部分的数据很重要，如何处理这部分特征有很大的发挥空间，本baseline不涉及这些特征。
drop_feat = ['问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题', '关注话题','感兴趣话题','问题创建时间','邀请创建时间']
data  = data.drop(drop_feat, axis=1)

print(data.columns)


# ## 4. 特征处理
# 

# **编码：**将离散型的特征通过LabelEncoder进行数字编码。
# 非数值特征-->数字类别码
# In[2]:


from sklearn.preprocessing import LabelEncoder
class_feat =  ['用户id','问题id','性别', '访问评率','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e']
encoder = LabelEncoder()
for feat in class_feat:
    encoder.fit(data[feat])
    data[feat] = encoder.transform(data[feat])

# **构造计数特征：**对具有很好区分度的特征进行单特征计数(有明显提升)。

# In[4]:


for feat in ['用户id','问题id','性别', '访问评率','用户二分类特征a', '用户二分类特征b', '用户二分类特征c', '用户二分类特征d',
       '用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e']:
    col_name = '{}_count'.format(feat)      # 转为字符串XXX_count
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int)) # value_counts查看表格某列中有多少个不同值的快捷方法，并计算每个不同值有在该列中有多少重复值
    data.loc[data[col_name] < 2, feat] = -1 # loc以行标签索引
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min()) # 归一化


# ## 5. 模型训练和预测

# In[6]:


from lightgbm import LGBMClassifier

# 划分训练集和测试集
y_train = data[:train.shape[0]]['是否回答'].values
X_train = data[:train.shape[0]].drop(['是否回答'], axis=1).values

X_test = data[train.shape[0]:].drop(['是否回答'], axis=1).values

model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)
# 建议使用CV的方式训练预测。
model_lgb.fit(X_train, y_train, 
                  eval_names=['train'],
                  eval_metric=['logloss','auc'],
                  eval_set=[(X_train, y_train)],
                  early_stopping_rounds=10)
y_pred = model_lgb.predict_proba(X_test)[:, 1]
result_append['是否回答'] = y_pred
result_append.to_csv('result.txt', index=False, header=False, sep='\t')

# 压缩提交结果文件result.txt，可以得到得分 0.763213863070066
