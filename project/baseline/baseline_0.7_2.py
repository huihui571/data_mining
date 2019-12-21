#!/usr/bin/env python
# coding: utf-8

# ## 智源 - 看山杯 专家发现算法大赛 baseline 0.7
# 
# 
# **分数（AUC）**：线下 0.719116，线上 0.701722
# 
# **方法**：构造用户、问题特征，构造用户问题交叉特征，5折交叉
# 
# **模型**：Catboost
# 
# **测试环境**：Ubuntu18，CPU32核，内存125G *(实际内存使用峰值约30%)*，显卡RTX2080Ti 
# 
# ---
# 
# #### 特征说明
# 
# **1. 用户特征**
# 
# |特征 | 特征说明 
# | :------:| :------: | 
# | 'gender', 'freq', 'A1', 'B1', 'C1' ... | 用户原始特征 | 
# | 'num_atten_topic', 'num_interest_topic' | 用户关注和感兴趣的topic数 | 
# | 'most_interest_topic' | 用户最感兴趣的topic | 
# | 'min_interest_values', 'max...', 'std...', 'mean...' | 用户topic兴趣值的统计特征 | 
# 
# ---
# 
# **2. 问题特征**
# 
# | 特征 | 特征说明 
# | :------:| :------: | 
# | 'num_title_sw', 'num_title_w' | 标题 词计数 | 
# | 'num_desc_sw', 'num_desc_w' | 描述 词计数 | 
# | 'num_qtopic' | topic计数 | 
# 
# ---
# 
# **3. 用户问题交叉特征**
# 
# | 特征 | 特征说明 
# | :------:| :------: | 
# | 'num_topic_attent_intersection' | 关注topic交集计数 | 
# | 'num_topic_interest_intersection' | 兴趣topic交集计数 | 
# | 'min_topic_interest...', 'max...', 'std...', 'mean...' | 交集topic兴趣值统计 | 
# 
# ---
# 
# #### 代码及说明
# 
# **1. preprocess**: 数据预处理，包括解析列表，重编码id，pickle保存。
# 
# &ensp;&ensp;运行时间 1388s，内存占用峰值 125G * 30%
# 
# **2. gen_feat**: 构造特征，特征说明如上述。
# 
# &ensp;&ensp;运行时间（32核）1764s，内存占用峰值 125G * 20%
# 
# &ensp;&ensp;*(注：这里为了加快运算，所以用了多进程 ，windows上 multiprocessing + jupyter可能有bug，建议linux上跑。)*
# 
# **3. baseline**: 模型训练预测。

# In[ ]:


#####################

# ## 1. preprocess

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm_notebook
import os
import time

# In[ ]:


tic = time.time()

# In[ ]:


# 减少内存占用
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# In[ ]:


# 解析列表， 重编码id
# 处理vector 转成ndarray
def parse_str(d):
    return np.array(list(map(float, d.split())))
# 处理W T,得到数字 list
def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))
# 处理SW
def parse_list_2(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))
# 处理感兴趣话题 得到dict {132 ,1.546302}
def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))

# In[ ]:


PATH = '../data'
SAVE_PATH = './pkl'
if not os.path.exists(SAVE_PATH):
    print('create dir: %s' % SAVE_PATH)
    os.mkdir(SAVE_PATH)

# ### single word

# In[ ]:


single_word = pd.read_csv(os.path.join(PATH, 'single_word_vectors_64d.txt'), 
                          names=['id', 'embed'], sep='\t')
single_word.head()
#print(single_word['embed'].head())

# In[ ]:


single_word['embed'] = single_word['embed'].apply(parse_str)  # apply 对dataframe row或者column进行计算
single_word['id'] = single_word['id'].apply(lambda x: int(x[2:])) # 转换成 1 2 3 4
single_word.head()

# In[ ]:


with open('./pkl/single_word.pkl', 'wb') as file:
    pickle.dump(single_word, file)

del single_word
gc.collect()

# ### word

# In[ ]:


word = pd.read_csv(os.path.join(PATH, 'word_vectors_64d.txt'), 
                          names=['id', 'embed'], sep='\t')
word.head()

# In[ ]:


word['embed'] = word['embed'].apply(parse_str)
word['id'] = word['id'].apply(lambda x: int(x[1:]))
word.head()

# In[ ]:


with open('./pkl/word.pkl', 'wb') as file:
    pickle.dump(word, file)
    
del word
gc.collect()

# ### topic

# In[ ]:


topic = pd.read_csv(os.path.join(PATH, 'topic_vectors_64d.txt'), 
                          names=['id', 'embed'], sep='\t')
topic.head()

# In[ ]:


topic['embed'] = topic['embed'].apply(parse_str)
topic['id'] = topic['id'].apply(lambda x: int(x[1:]))
topic.head()

# In[ ]:


with open('./pkl/topic.pkl', 'wb') as file:
    pickle.dump(topic, file)
    
del topic
gc.collect()

# ### invite

# In[ ]:


invite_info = pd.read_csv(os.path.join(PATH, 'invite_info_0926.txt'), 
                          names=['question_id', 'author_id', 'invite_time', 'label'], sep='\t')
invite_info_evaluate = pd.read_csv(os.path.join(PATH, 'invite_info_evaluate_1_0926.txt'), 
                          names=['question_id', 'author_id', 'invite_time'], sep='\t')
invite_info.head()

# In[ ]:


invite_info['invite_day'] = invite_info['invite_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
invite_info['invite_hour'] = invite_info['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)

# In[ ]:


invite_info_evaluate['invite_day'] = invite_info_evaluate['invite_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
invite_info_evaluate['invite_hour'] = invite_info_evaluate['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)

# In[ ]:


invite_info = reduce_mem_usage(invite_info)

# In[ ]:


with open('./pkl/invite_info.pkl', 'wb') as file:
    pickle.dump(invite_info, file)
    
with open('./pkl/invite_info_evaluate.pkl', 'wb') as file:
    pickle.dump(invite_info_evaluate, file)
    
del invite_info, invite_info_evaluate
gc.collect()

# ### member

# In[ ]:


member_info = pd.read_csv(os.path.join(PATH, 'member_info_0926.txt'), 
                          names=['author_id', 'gender', 'keyword', 'grade', 'hotness', 'reg_type','reg_plat','freq',
                                 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2',
                                 'score', 'topic_attent', 'topic_interest'], sep='\t')
member_info.head()

# In[ ]:


member_info['topic_attent'] = member_info['topic_attent'].apply(parse_list_1)
member_info['topic_interest'] = member_info['topic_interest'].apply(parse_map)

# In[ ]:


member_info = reduce_mem_usage(member_info)

# In[ ]:


with open('./pkl/member_info.pkl', 'wb') as file:
    pickle.dump(member_info, file)
    
del member_info
gc.collect()

# ### question

# In[ ]:


question_info = pd.read_csv(os.path.join(PATH, 'question_info_0926.txt'),
                          names=['question_id', 'question_time', 'title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series', 'topic'], sep='\t')
question_info.head()

# In[ ]:


question_info['title_sw_series'] = question_info['title_sw_series'].apply(parse_list_2)#.apply(sw_lbl_enc.transform).apply(list)
question_info['title_w_series'] = question_info['title_w_series'].apply(parse_list_1)#.apply(w_lbl_enc.transform).apply(list)
question_info['desc_sw_series'] = question_info['desc_sw_series'].apply(parse_list_2)#.apply(sw_lbl_enc.transform).apply(list)
question_info['desc_w_series'] = question_info['desc_w_series'].apply(parse_list_1)#.apply(w_lbl_enc.transform).apply(list)
question_info['topic'] = question_info['topic'].apply(parse_list_1)# .apply(topic_lbl_enc.transform).apply(list)
question_info.head()

# In[ ]:


question_info['question_day'] = question_info['question_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
question_info['question_hour'] = question_info['question_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)
del question_info['question_time']
gc.collect()

# In[ ]:


question_info = reduce_mem_usage(question_info)

# In[ ]:


with open('./pkl/question_info.pkl', 'wb') as file:
    pickle.dump(question_info, file)
    
del question_info
gc.collect()

# ### answer

# In[ ]:


# %%time
answer_info = pd.read_csv(os.path.join(PATH, 'answer_info_0926.txt'), 
                          names=['answer_id', 'question_id', 'author_id', 'answer_time', 'content_sw_series', 'content_w_series', 
                                 'excellent', 'recommend', 'round_table', 'figure', 'video', 
                                 'num_word', 'num_like', 'num_unlike', 'num_comment',
                                 'num_favor', 'num_thank', 'num_report', 'num_nohelp', 'num_oppose'], sep='\t')
answer_info.head()

# In[ ]:


answer_info['content_sw_series'] = answer_info['content_sw_series'].apply(parse_list_2) 
answer_info['content_w_series'] = answer_info['content_w_series'].apply(parse_list_1) 
answer_info.head()

# In[ ]:


answer_info['answer_day'] = answer_info['answer_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
answer_info['answer_hour'] = answer_info['answer_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)
del answer_info['answer_time']
gc.collect()

# In[ ]:


answer_info = reduce_mem_usage(answer_info)

# In[ ]:


with open('./pkl/answer_info.pkl', 'wb') as file:
    pickle.dump(answer_info, file)

del answer_info
gc.collect()

# In[ ]:


toc = time.time()

# In[ ]:


print('Used time: %d' % int(toc-tic))

# In[ ]:




# ## 2. gen_feat

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import gc
import os
import time
import multiprocessing as mp

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# In[ ]:


tic = time.time()

# In[ ]:


SAVE_PATH = './feats'
if not os.path.exists(SAVE_PATH):
    print('create dir: %s' % SAVE_PATH)
    os.mkdir(SAVE_PATH)

# ### member_info: 用户特征

# In[ ]:


with open('./pkl/member_info.pkl', 'rb') as file:
    member_info = pickle.load(file)
member_info.head(2)

# In[ ]:


# 原始类别特征
member_cat_feats = ['gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2']
for feat in member_cat_feats:
    member_info[feat] = LabelEncoder().fit_transform(member_info[feat])

# In[ ]:


# 用户关注和感兴趣的topic数
member_info['num_atten_topic'] = member_info['topic_attent'].apply(len)
member_info['num_interest_topic'] = member_info['topic_interest'].apply(len)

# In[ ]:


def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]

# In[ ]:


# 用户最感兴趣的topic
member_info['most_interest_topic'] = member_info['topic_interest'].apply(most_interest_topic)
member_info['most_interest_topic'] = LabelEncoder().fit_transform(member_info['most_interest_topic'])

# In[ ]:


def get_interest_values(d):
    if len(d) == 0:
        return [0]
    return list(d.values())

# In[ ]:


# 用户topic兴趣值的统计特征
member_info['interest_values'] = member_info['topic_interest'].apply(get_interest_values)
member_info['min_interest_values'] = member_info['interest_values'].apply(np.min)
member_info['max_interest_values'] = member_info['interest_values'].apply(np.max)
member_info['mean_interest_values'] = member_info['interest_values'].apply(np.mean)
member_info['std_interest_values'] = member_info['interest_values'].apply(np.std)

# In[ ]:


# 汇总
feats = ['author_id', 'gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2', 'score']
feats += ['num_atten_topic', 'num_interest_topic', 'most_interest_topic']
feats += ['min_interest_values', 'max_interest_values', 'mean_interest_values', 'std_interest_values']
member_feat = member_info[feats]

# In[ ]:


member_feat.head(3)

# In[ ]:


member_feat.to_hdf('./feats/member_feat.h5', key='data')

del member_feat, member_info
gc.collect()

# ### question_info: 问题特征

# In[ ]:


with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)
    
question_info.head(2)

# In[ ]:


# title、desc词计数，topic计数
question_info['num_title_sw'] = question_info['title_sw_series'].apply(len)
question_info['num_title_w'] = question_info['title_w_series'].apply(len)
question_info['num_desc_sw'] = question_info['desc_sw_series'].apply(len)
question_info['num_desc_w'] = question_info['desc_w_series'].apply(len)
question_info['num_qtopic'] = question_info['topic'].apply(len)

# In[ ]:


feats = ['question_id', 'num_title_sw', 'num_title_w', 'num_desc_sw', 'num_desc_w', 'num_qtopic', 'question_hour']
feats += []
question_feat = question_info[feats]

# In[ ]:


question_feat.head(3)

# In[ ]:


question_feat.to_hdf('./feats/question_feat.h5', key='data')

# In[ ]:


del question_info, question_feat
gc.collect()

# ### member_info & question_info: 用户和问题的交互特征

# In[ ]:


with open('./pkl/invite_info.pkl', 'rb') as file:
    invite_info = pickle.load(file)
with open('./pkl/invite_info_evaluate.pkl', 'rb') as file:
    invite_info_evaluate = pickle.load(file)
with open('./pkl/member_info.pkl', 'rb') as file:
    member_info = pickle.load(file)
with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

# In[ ]:


# 合并 author_id，question_id
invite = pd.concat([invite_info, invite_info_evaluate])
invite_id = invite[['author_id', 'question_id']]
invite_id['author_question_id'] = invite_id['author_id'] + invite_id['question_id']
invite_id.drop_duplicates(subset='author_question_id',inplace=True)
invite_id_qm = invite_id.merge(member_info[['author_id', 'topic_attent', 'topic_interest']], 'left', 'author_id').merge(question_info[['question_id', 'topic']], 'left', 'question_id')
invite_id_qm.head(2)

# #### 注：这里为了加快运算，所以用了多进程 multiprocessing，windows + multiprocessing + jupyter可能有bug，建议linux上跑。

# In[ ]:


# 分割 df，方便多进程跑
def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

# In[ ]:


def gc_mp(pool, ret, chunk_list):
    del pool
    for r in ret:
        del r
    del ret
    for cl in chunk_list:
        del cl
    del chunk_list
    gc.collect()

# In[ ]:


# 用户关注topic和问题 topic的交集
def process(df):
    return df.apply(lambda row: list(set(row['topic_attent']) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_attent_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

# In[ ]:


# 用户感兴趣topic和问题 topic的交集
def process(df):
    return df.apply(lambda row: list(set(row['topic_interest'].keys()) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_interest_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

# In[ ]:


# 用户感兴趣topic和问题 topic的交集的兴趣值
def process(df):
    return df.apply(lambda row: [row['topic_interest'][t] for t in row['topic_interest_intersection']],axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_interest_intersection_values'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

# In[ ]:


# 交集topic计数
invite_id_qm['num_topic_attent_intersection'] = invite_id_qm['topic_attent_intersection'].apply(len)
invite_id_qm['num_topic_interest_intersection'] = invite_id_qm['topic_interest_intersection'].apply(len)

# In[ ]:


# 交集topic兴趣值统计
invite_id_qm['topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(lambda x: [0] if len(x) == 0 else x)
invite_id_qm['min_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.min)
invite_id_qm['max_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.max)
invite_id_qm['mean_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.mean)
invite_id_qm['std_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.std)

# In[ ]:


feats = ['author_question_id', 'num_topic_attent_intersection', 'num_topic_interest_intersection', 'min_topic_interest_intersection_values', 'max_topic_interest_intersection_values', 'mean_topic_interest_intersection_values', 'std_topic_interest_intersection_values']
feats += []
member_question_feat = invite_id_qm[feats]
member_question_feat.head(3)

# In[ ]:


member_question_feat.to_hdf('./feats/member_question_feat.h5', key='data')

# In[ ]:


del invite_id_qm, member_question_feat
gc.collect()

# In[ ]:


toc = time.time()
print('Used time: %d' % int(toc-tic))

# In[ ]:




# ## 3. baseline

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# In[ ]:


import pandas as pd
import numpy as np
import gc
import pickle
import time

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# In[ ]:


from catboost import CatBoostClassifier, Pool

# In[ ]:


tic = time.time()

# In[ ]:


with open('./pkl/invite_info.pkl', 'rb') as file:
    invite_info = pickle.load(file)
with open('./pkl/invite_info_evaluate.pkl', 'rb') as file:
    invite_info_evaluate = pickle.load(file)

# In[ ]:


member_feat = pd.read_hdf('./feats/member_feat.h5', key='data')  # 0.689438
question_feat = pd.read_hdf('./feats/question_feat.h5', key='data')  # 0.706848

# In[ ]:


member_question_feat = pd.read_hdf('./feats/member_question_feat.h5', key='data')  # 719116 d12
invite_info['author_question_id'] = invite_info['author_id'] + invite_info['question_id']
invite_info_evaluate['author_question_id'] = invite_info_evaluate['author_id'] + invite_info_evaluate['question_id']

# In[ ]:


train = invite_info.merge(member_feat, 'left', 'author_id')
test = invite_info_evaluate.merge(member_feat, 'left', 'author_id')

# In[ ]:


train = train.merge(question_feat, 'left', 'question_id')
test = test.merge(question_feat, 'left', 'question_id')

# In[ ]:


train = train.merge(member_question_feat, 'left', 'author_question_id')
test = test.merge(member_question_feat, 'left', 'author_question_id')

# In[ ]:


del member_feat, question_feat, member_question_feat
gc.collect()

# In[ ]:


drop_feats = ['question_id', 'author_id', 'author_question_id', 'invite_time', 'label', 'invite_day']

used_feats = [f for f in train.columns if f not in drop_feats]
print(len(used_feats))
print(used_feats)

# In[ ]:


train_x = train[used_feats].reset_index(drop=True)
train_y = train['label'].reset_index(drop=True)
test_x = test[used_feats].reset_index(drop=True)

# In[ ]:


preds = np.zeros((test_x.shape[0], 2))
scores = []
has_saved = False
imp = pd.DataFrame()
imp['feat'] = used_feats

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for index, (tr_idx, va_idx) in enumerate(kfold.split(train_x, train_y)):
    print('*' * 30)
    X_train, y_train, X_valid, y_valid = train_x.iloc[tr_idx], train_y.iloc[tr_idx], train_x.iloc[va_idx], train_y.iloc[va_idx]
    cate_features = []
    train_pool = Pool(X_train, y_train, cat_features=cate_features)
    eval_pool = Pool(X_valid, y_valid,cat_features=cate_features)
    if not has_saved: 
        cbt_model = CatBoostClassifier(iterations=10000,
                           learning_rate=0.1,
                           eval_metric='AUC',
                           use_best_model=True,
                           random_seed=42,
                           logging_level='Verbose',
                           task_type='GPU',
                           devices='0',
                           early_stopping_rounds=300,
                           loss_function='Logloss',
                           depth=12,
                           )
        cbt_model.fit(train_pool, eval_set=eval_pool, verbose=100)
#         with open('./models/fold%d_cbt_v1.mdl' % index, 'wb') as file:
#             pickle.dump(cbt_model, file)
    else:
        with open('./models/fold%d_cbt_v1.mdl' % index, 'rb') as file:
            cbt_model = pickle.load(file)
    
    imp['score%d' % (index+1)] = cbt_model.feature_importances_
    
    score = cbt_model.best_score_['validation']['AUC']
    scores.append(score)
    print('fold %d round %d : score: %.6f | mean score %.6f' % (index+1, cbt_model.best_iteration_, score,np.mean(scores))) 
    preds += cbt_model.predict_proba(test_x)  
    
    del cbt_model, train_pool, eval_pool
    del X_train, y_train, X_valid, y_valid
    import gc
    gc.collect()
    
#     mdls.append(cbt_model)

# In[ ]:


imp.sort_values(by='score1', ascending=False)

# In[ ]:


result = invite_info_evaluate[['question_id', 'author_id', 'invite_time']]
result['result'] = preds[:, 1] / 5
result.head()

# In[ ]:


result.to_csv('./result.txt', sep='\t', index=False, header=False)

# In[ ]:


toc = time.time()
print('Used time: %d' % int(toc - tic))
