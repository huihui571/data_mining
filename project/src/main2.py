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

# cd ../data

# In[7]:


# import pandas as pd
# user_info = pd.read_csv('member_info_0926.txt', header=None, sep='\t')
# user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台','访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d','用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e','盐值','关注话题','感兴趣话题']
# for col in user_info.columns:
#     print(col, len(user_info[col].unique()))
#
# question_info = pd.read_csv('question_info_0926.txt', header=None, sep='\t')
# question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
# for col in question_info.columns:
#     print(col, len(question_info[col].unique()))

# 从上面的数据分析可以看出，用户数据中有21个特征，其中5个特征（创作关键词、创作数量等级、创作热度等级、注册类型、注册平台）在数据集中只有一个取值，说明这5个特征是完全无用的，可以直接去掉。

# ### 2.2 数据集合并

# 为了分析上述两个数据集中的特征是否有对预测结果有影响（或者说这些特征是否是有区分度的强特征），我们首先将这两个数据集和训练集（invite_info_0926.txt）合并, 然后通过图表来对部分特征进行分析。

# In[8]:


# train = pd.read_csv('invite_info_0926.txt', header=None, sep='\t')
# train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
# train = pd.merge(train, user_info, how='left', on='用户id')
# train = pd.merge(train, question_info, how='left', on='问题id')
# print(train.columns)

# **性别特征：**性别特征有三个类别，分别是 男性，女性和未知，下面的柱状图可以看出，男性和女性分布非常相似，未知的分布相比较之下有 较大的区别，显然，该特征具有较好的区分度。

# In[9]:


# import matplotlib.pyplot as plt
# import seaborn as sns
# % matplotlib inline
# sns.set_style('whitegrid')
#
# sns.countplot(x='性别',hue='是否回答',data=train)

# **访问频率特征：**该特征总计5中类别，[weekly, monthly, daily, new, unknown]，从下面的柱状图可以看出，不同的类别具有完全不同的分布，这特征显然是一种很有区分度很强的特征。

# In[10]:


# sns.countplot(x='访问评率',hue='是否回答',data=train)

# **用户二分类特征a：**该特征是二分类特征，下图表示该特征具备很好的区分度（剩下的二分类和多分类特征也是同理，不赘述）。

# In[11]:


# sns.countplot(x='用户二分类特征a',hue='是否回答',data=train)

# **盐值：**我们先对盐值进行分桶，然后查看不同区间盐值的分布情况。下图表示不同区间盐值的用户具有很有的区分度，在处理这个特征时，至于是否分桶，如何通过更加详细的数据分析自由发挥，给出的baseline对该特征未做处理。

# In[12]:


# def trans(x):
#     if x <= 0:
#         return x
#     if 1 <= x <= 10:
#         return 1
#     if 10 < x <= 100:
#         return 2
#     if 100 < x <= 200:
#         return 3
#     if 200 < x <= 300:
#         return 4
#     if 400 < x <= 500:
#         return 5
#     if 500 < x <= 600:
#         return 6
#     if x > 600:
#         return 7
# train['盐值'] = train['盐值'].apply(lambda x: trans(x))
# sns.countplot(x='盐值',hue='是否回答',data=train)

# **时间：**数据集中的时间都采用“D×-H×”的格式，D代表天数，H代表小时，我们需要将这一个特征转化为两个特征，天和小时。

# ## 3. 数据处理
# 

# 从这一部分开始都是baseline的全部代码（前面仅仅是简单的数据分析，不作为baseline的代码）。

# In[1]:

import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))

def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))

def parse_str(d):
    return np.array(list(map(float, d.split())))

def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))

def parse_list_2(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))

def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))

def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]
# 导入数据
train = pd.read_csv('../data/data_set_0926/invite_info_0926.txt', header=None, sep='\t')
test = pd.read_csv('../data/data_set_0926/invite_info_evaluate_1_0926.txt', header=None, sep='\t')

train.columns = ['问题id', '用户id', '邀请创建时间','是否回答']
print("invite %s",train.shape)
# train = pd.merge(train, user_info, how='left', on='用户id')
# train = pd.merge(train, question_info, how='left', on='问题id')

test.columns = ['问题id', '用户id', '邀请创建时间']
print("test %s",test.shape)

result_append = test.copy()
result_size = len(result_append)

train['邀请创建时间-day'] = extract_day(train['邀请创建时间'])
train['邀请创建时间-hour'] = extract_hour(train['邀请创建时间'])

test['邀请创建时间-day'] = extract_day(test['邀请创建时间'])
test['邀请创建时间-hour'] = extract_hour(test['邀请创建时间'])

del train['邀请创建时间'], test['邀请创建时间']
# test = pd.merge(test, user_info, how='left', on='用户id')
# test = pd.merge(test, question_info, how='left', on='问题id')

# 加载问题
question_info = pd.read_csv('../data/data_set_0926/question_info_0926.txt', header=None, sep='\t')
question_info.columns = ['问题id','问题创建时间','问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题']
del question_info['问题标题单字编码'], question_info['问题标题切词编码'], question_info['问题描述单字编码'], question_info['问题描述切词编码']

question_info['问题创建时间-day'] = extract_day(question_info['问题创建时间'])
question_info['问题创建时间-hour'] = extract_hour(question_info['问题创建时间'])
del question_info['问题创建时间']

# 加载回答
answer_info = pd.read_csv('../data/data_set_0926/answer_info_0926.txt', header=None,sep='\t')
answer_info.columns = ['回答id','问题id','用户id','回答创建时间','回答内容的单字编码序列','回答内容的切词编码序列','回答是否被标优',
                       '回答是否被推荐','回答是否被收入圆桌','是否包含图片','是否包含视频','回答字数','点赞数','取赞数','评论数','收藏数',
                       '感谢数','举报数','没有帮助数','反对数']
del answer_info['回答内容的单字编码序列'],answer_info['回答内容的切词编码序列']

answer_info['回答创建时间-day'] = extract_day(answer_info['回答创建时间'])
answer_info['回答创建时间-hour'] = extract_hour(answer_info['回答创建时间'])
del answer_info['回答创建时间']

answer_info = pd.merge(answer_info, question_info, on='问题id')
del question_info

# 回答距提问的天数
answer_info['回答距提问的天数'] = answer_info['回答创建时间-day'] - answer_info['问题创建时间-day']

# 时间窗口划分
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6

train_label_feature_end = label_end - 7
train_label_feature_start = train_label_feature_end - 22

train_ans_feature_end = label_end - 7
train_ans_feature_start = train_ans_feature_end - 50

val_label_feature_end = val_start - 1
val_label_feature_start = val_label_feature_end - 22

val_ans_feature_end = val_start - 1
val_ans_feature_start = val_ans_feature_end - 50

train_label_feature = train[(train['邀请创建时间-day'] >= train_label_feature_start) & (train['邀请创建时间-day'] <= train_label_feature_end)]

val_label_feature = train[(train['邀请创建时间-day'] >= val_label_feature_start) & (train['邀请创建时间-day'] <= val_label_feature_end)]

train_label = train[(train['邀请创建时间-day'] > train_label_feature_end)]

# 确定answer的时间范围
# 3807~3874
train_ans_feature = answer_info[(answer_info['回答创建时间-day'] >= train_ans_feature_start) & (answer_info['回答创建时间-day'] <= train_ans_feature_end)]

val_ans_feature = answer_info[(answer_info['回答创建时间-day'] >= val_ans_feature_start) & (answer_info['回答创建时间-day'] <= val_ans_feature_end)]

feature = ['回答是否被推荐','回答是否被收入圆桌','是否包含图片','是否包含视频','回答字数','点赞数','取赞数','评论数','收藏数',
                       '感谢数','举报数','没有帮助数','反对数','回答距提问的天数']

def extract_feature(target, label_feature, answer_feature):
    # 问题特征
    t1 = label_feature.groupby('问题id')['是否回答'].agg(['mean','sum','std','count']).reset_index()
    t1.columns = ['问题id','问题邀请平均值','问题邀请总和','问题邀请标准差','问题邀请数']
    target = pd.merge(target, t1, on='问题id', how='left')

    # 用户特征
    t1 = label_feature.groupby('用户id')['是否回答'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['用户id', '用户邀请平均值', '用户邀请总和', '用户邀请标准差', '用户邀请数']
    target = pd.merge(target, t1, on='用户id', how='left')

    # 回答的部分特征
    t1 = answer_feature.groupby('问题id')['回答id'].count().reset_index()
    t1.columns = ['问题id','问题回答数']
    target = pd.merge(target, t1, on='问题id',how='left')

    t1 = answer_feature.groupby('用户id')['回答id'].count().reset_index()
    t1.columns = ['用户id','用户回答数']
    target = pd.merge(target, t1, on='用户id', how='left')

    for feat in feature:
        t1 = answer_feature.groupby('用户id')[feat].agg(['sum', 'max','mean']).reset_index()
        t1.columns = ['用户id',f'用户{feat}总和',f'用户{feat}最大值',f'用户{feat}平均值']
        target = pd.merge(target, t1, on='用户id', how='left')
        t1 = answer_feature.groupby('问题id')[feat].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['问题id', f'问题{feat}总和', f'问题{feat}最大值', f'问题{feat}平均值']
        target = pd.merge(target, t1, on='问题id', how='left')

    return target

train_label = extract_feature(train_label,train_label_feature,train_ans_feature)
test = extract_feature(test,val_label_feature,val_ans_feature)

assert len(test) == result_size

# 加载用户
user_info = pd.read_csv('../data/data_set_0926/member_info_0926.txt', header=None, sep='\t')
user_info.columns = ['用户id','性别','创作关键词','创作数量等级','创作热度等级','注册类型','注册平台',
                     '访问评率','用户二分类特征a','用户二分类特征b','用户二分类特征c','用户二分类特征d',
                     '用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d',
                     '用户多分类特征e','盐值','关注话题','感兴趣话题']
answer_info.head()

user_info['关注话题'] = user_info['关注话题'].apply(parse_list_1)
user_info['感兴趣话题'] = user_info['感兴趣话题'].apply(parse_map)
answer_info.head()

# 用户关注和感兴趣的话题数
user_info['关注话题数'] = user_info['关注话题'].apply(len)
user_info['感兴趣话题数'] = user_info['感兴趣话题'].apply(len)

# 用户最感兴趣的话题
user_info['最感兴趣的话题'] = user_info['感兴趣话题'].apply(most_interest_topic)
user_info['最感兴趣的话题'] = LabelEncoder().fit_transform(user_info['最感兴趣的话题'])

del user_info['关注话题'],user_info['感兴趣话题']
del user_info['创作关键词'],user_info['创作数量等级'],user_info['创作热度等级'],user_info['注册类型'],user_info['注册平台']

# 删除的特征并非不重要，相反这部分的数据很重要，如何处理这部分特征有很大的发挥空间，本baseline不涉及这些特征。
# drop_feat = ['问题标题单字编码','问题标题切词编码','问题描述单字编码','问题描述切词编码','问题绑定话题', '关注话题','感兴趣话题','问题创建时间','邀请创建时间']
# data  = data.drop(drop_feat, axis=1)

# print(data.columns)


# 对离散型的特征进行数字编码
class_feat = ['用户id','问题id','性别', '访问评率','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e']
encoder = LabelEncoder()
t = user_info.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['关注话题','感兴趣话题','用户id']]

for d in cats:
    user_info[d] = encoder.fit_transform(user_info[d])

# 对问题id进行数字编码
question_encoder = LabelEncoder()
question_encoder.fit(list(train_label['问题id'].astype(str).values) + list(test['问题id'].astype(str).values))
train_label['问题id_编码'] = question_encoder.transform(train_label['问题id'])
test['问题id_编码'] = question_encoder.transform(test['问题id'])

# 对用户id进行数字编码
user_encoder = LabelEncoder()
user_encoder.fit(user_info['用户id'])
train_label['用户id_编码'] = user_encoder.transform(train_label['用户id'])
test['用户id_编码'] = user_encoder.transform(test['用户id'])

# 数据合并
train_label = pd.merge(train_label, user_info, on='用户id', how='left')
test = pd.merge(test, user_info, on='用户id', how='left')

print("train shape %s, test shape %s",train_label.shape,test.shape)

data = pd.concat((train_label, test), axis=0, sort=True)

# 构造计数特征
count_feat = ['用户id_编码','问题id_编码','性别', '访问评率','用户二分类特征a', '用户二分类特征b', '用户二分类特征c', '用户二分类特征d',
        '用户二分类特征e','用户多分类特征a','用户多分类特征b','用户多分类特征c','用户多分类特征d','用户多分类特征e']
for feat in count_feat:
    col_name = f'{feat}_count'
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())

# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

data['邀请创建时间-week'] = data['邀请创建时间-day'] % 7

feature_cols = [x for x in data.columns if x not in ('是否回答', '用户id', '问题id', '邀请创建时间', '邀请创建时间-day')]

# 划分训练集和测试集
X_train_all = data.iloc[:len(train_label)][feature_cols]
y_train_all = data.iloc[:len(train_label)]['是否回答']
test = data.iloc[len(train_label):]
del data
assert len(test) == result_size

# 5折交叉验证
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    break

X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols].values, X_train_all.iloc[val_idx][feature_cols].values, \
                                 y_train_all.iloc[train_idx].values, \
                                 y_train_all.iloc[val_idx].values
del X_train_all

# X_train = X_train_all.values
# y_train = y_train_all.values
# 模型训练和预测

model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=5000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)
model_lgb.fit(X_train, y_train, 
                  eval_names=['train'],
                  eval_metric=['logloss','auc'],
                  eval_set=[(X_val,y_val)],
                  early_stopping_rounds=50)

y_pred = model_lgb.predict_proba(test[feature_cols])[:, 1]
result_append['是否回答'] = y_pred
result_append.to_csv('result_swc.txt', index=False, header=False, sep='\t')

# 压缩提交结果文件result.txt
