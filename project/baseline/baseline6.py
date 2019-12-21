import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging
import pickle
import gc
import time
import multiprocessing as mp

log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


base_path = 'data_set_0926'

# 加载邀请回答数据

train = pd.read_csv(f'{base_path}/invite_info_0926.txt', sep='\t', header=None)
train.columns = ['qid', 'uid', 'dt', 'label']
logging.info("invite %s", train.shape)

test = pd.read_csv(f'{base_path}/invite_info_evaluate_2_0926.txt', sep='\t', header=None)
test.columns = ['qid', 'uid', 'dt']
logging.info("test %s", test.shape)

sub = test.copy()

sub_size = len(sub)

train['day'] = extract_day(train['dt'])
train['hour'] = extract_hour(train['dt'])

test['day'] = extract_day(test['dt'])
test['hour'] = extract_hour(test['dt'])

del train['dt'], test['dt']

# 解析列表， 重编码id
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


# 加载问题
ques = pd.read_csv(f'{base_path}/question_info_0926.txt', header=None, sep='\t')
ques.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
ques['topic'] = ques['topic'].apply(parse_list_1)# .apply(topic_lbl_enc.transform).apply(list)
del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']
logging.info("ques %s", ques.shape)

ques['q_day'] = extract_day(ques['q_dt'])
ques['q_hour'] = extract_hour(ques['q_dt'])
del ques['q_dt']

# 加载回答
ans = pd.read_csv(f'{base_path}/answer_info_0926.txt', header=None, sep='\t')
ans.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',
               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
               'reci_xxx', 'reci_no_help', 'reci_dis']
del ans['ans_t1'], ans['ans_t2']
logging.info("ans %s", ans.shape)

ans['a_day'] = extract_day(ans['ans_dt'])
ans['a_hour'] = extract_hour(ans['ans_dt'])
del ans['ans_dt']

ans = pd.merge(ans, ques, on='qid')
# del ques

# 回答距提问的天数
ans['diff_qa_days'] = ans['a_day'] - ans['q_day']

# 时间窗口划分
# train
# val
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

train_label_feature = train[(train['day'] >= train_label_feature_start) & (train['day'] <= train_label_feature_end)]
logging.info("train_label_feature %s", train_label_feature.shape)

val_label_feature = train[(train['day'] >= val_label_feature_start) & (train['day'] <= val_label_feature_end)]
logging.info("val_label_feature %s", val_label_feature.shape)

train_label = train[(train['day'] > train_label_feature_end)]

logging.info("train feature start %s end %s, label start %s end %s", train_label_feature['day'].min(),
             train_label_feature['day'].max(), train_label['day'].min(), train_label['day'].max())

logging.info("test feature start %s end %s, label start %s end %s", val_label_feature['day'].min(),
             val_label_feature['day'].max(), test['day'].min(), test['day'].max())

# 确定ans的时间范围
# 3807~3874
train_ans_feature = ans[(ans['a_day'] >= train_ans_feature_start) & (ans['a_day'] <= train_ans_feature_end)]

val_ans_feature = ans[(ans['a_day'] >= val_ans_feature_start) & (ans['a_day'] <= val_ans_feature_end)]

logging.info("train ans feature %s, start %s end %s", train_ans_feature.shape, train_ans_feature['a_day'].min(),
             train_ans_feature['a_day'].max())

logging.info("val ans feature %s, start %s end %s", val_ans_feature.shape, val_ans_feature['a_day'].min(),
             val_ans_feature['a_day'].max())

fea_cols = ['is_good', 'is_rec', 'is_dest', 'has_img', 'has_video', 'word_count',
            'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
            'reci_xxx', 'reci_no_help', 'reci_dis', 'diff_qa_days']


def extract_feature1(target, label_feature, ans_feature):
    # 问题特征
    t1 = label_feature.groupby('qid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['qid', 'q_inv_mean', 'q_inv_sum', 'q_inv_std', 'q_inv_count']
    target = pd.merge(target, t1, on='qid', how='left')

    # 用户特征
    t1 = label_feature.groupby('uid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['uid', 'u_inv_mean', 'u_inv_sum', 'u_inv_std', 'u_inv_count']
    target = pd.merge(target, t1, on='uid', how='left')
    #
    # train_size = len(train)
    # data = pd.concat((train, test), sort=True)

    # 回答部分特征

    t1 = ans_feature.groupby('qid')['aid'].count().reset_index()
    t1.columns = ['qid', 'q_ans_count']
    target = pd.merge(target, t1, on='qid', how='left')

    t1 = ans_feature.groupby('uid')['aid'].count().reset_index()
    t1.columns = ['uid', 'u_ans_count']
    target = pd.merge(target, t1, on='uid', how='left')

    for col in fea_cols:
        t1 = ans_feature.groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['uid', f'u_{col}_sum', f'u_{col}_max', f'u_{col}_mean']
        target = pd.merge(target, t1, on='uid', how='left')

        t1 = ans_feature.groupby('qid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['qid', f'q_{col}_sum', f'q_{col}_max', f'q_{col}_mean']
        target = pd.merge(target, t1, on='qid', how='left')
        logging.info("extract %s", col)

    return target

train_label = extract_feature1(train_label, train_label_feature, train_ans_feature)
test = extract_feature1(test, val_label_feature, val_ans_feature)
train_label = pd.merge(train_label, ques[['qid', 'topic']], on='qid', how='left')
test = pd.merge(test, ques[['qid', 'topic']], on='qid', how='left')

# 特征提取结束
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
assert len(test) == sub_size

# 加载用户
user = pd.read_csv(f'{base_path}/member_info_0926.txt', header=None, sep='\t')
user.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1', 'uf_b2',
                'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic',
                'inter_topic']
# del user['follow_topic'], user['inter_topic']

logging.info("user %s", user.shape)

unq = user.nunique()
logging.info("user unq %s", unq)

for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)

t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in ['follow_topic', 'inter_topic', 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)

q_lb = LabelEncoder()
q_lb.fit(list(train_label['qid'].astype(str).values) + list(test['qid'].astype(str).values))
train_label['qid_enc'] = q_lb.transform(train_label['qid'])
test['qid_enc'] = q_lb.transform(test['qid'])

u_lb = LabelEncoder()
u_lb.fit(user['uid'])
train_label['uid_enc'] = u_lb.transform(train_label['uid'])
test['uid_enc'] = u_lb.transform(test['uid'])

# merge user
user['follow_topic'] = user['follow_topic'].apply(parse_list_1)
user['inter_topic'] = user['inter_topic'].apply(parse_map)
train_label = pd.merge(train_label, user, on='uid', how='left')
test = pd.merge(test, user, on='uid', how='left')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

# 计算user关注和感兴趣的话题与问题的话题的交集
# 分割 df，方便多进程跑
def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
def gc_mp(pool, ret, chunk_list):
    del pool
    for r in ret:
        del r
    del ret
    for cl in chunk_list:
        del cl
    del chunk_list
    gc.collect()

atr = train_label.columns.values
atrv = train_label.iloc[2,]
ate = test.columns.values
atev = test.iloc[2,]
print(atr)
print(atrv)
print(ate)
print(atev)


# 用户关注topic和问题 topic的交集
def process1(df):
    return df.apply(lambda row: list(set(row['follow_topic']) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(train_label, 100)
ret = pool.map(process1, chunk_list)
train_label['follow_topic_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)
print("1 finished")
pool = mp.Pool()
chunk_list = split_df(test, 100)
ret = pool.map(process1, chunk_list)
test['follow_topic_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)
print("2 finished")

# 用户感兴趣topic和问题 topic的交集
def process2(df):
    return df.apply(lambda row: list(set(row['inter_topic'].keys()) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(train_label, 100)
ret = pool.map(process2, chunk_list)
train_label['inter_topic_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

pool = mp.Pool()
chunk_list = split_df(test, 100)
ret = pool.map(process2, chunk_list)
test['inter_topic_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

# 用户感兴趣topic和问题 topic的交集的兴趣值
def process3(df):
    return df.apply(lambda row: [row['inter_topic'][t] for t in row['inter_topic_intersection']],axis=1)

pool = mp.Pool()
chunk_list = split_df(train_label, 100)
ret = pool.map(process3, chunk_list)
train_label['inter_topic_intersection_values'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

pool = mp.Pool()
chunk_list = split_df(test, 100)
ret = pool.map(process3, chunk_list)
test['inter_topic_intersection_values'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
# 交集topic计数
train_label['num_follow_topic_intersection'] = train_label['follow_topic_intersection'].apply(len)
train_label['num_inter_topic_intersection'] = train_label['inter_topic_intersection'].apply(len)
test['num_follow_topic_intersection'] = test['follow_topic_intersection'].apply(len)
test['num_inter_topic_intersection'] = test['inter_topic_intersection'].apply(len)

# 交集topic兴趣值统计
train_label['inter_topic_intersection_values'] = train_label['inter_topic_intersection_values'].apply(lambda x: [0] if len(x) == 0 else x)
train_label['min_inter_topic_intersection_values'] = train_label['inter_topic_intersection_values'].apply(np.min)
train_label['max_inter_topic_intersection_values'] = train_label['inter_topic_intersection_values'].apply(np.max)
train_label['mean_inter_topic_intersection_values'] = train_label['inter_topic_intersection_values'].apply(np.mean)
train_label['std_inter_topic_intersection_values'] = train_label['inter_topic_intersection_values'].apply(np.std)

test['inter_topic_intersection_values'] = test['inter_topic_intersection_values'].apply(lambda x: [0] if len(x) == 0 else x)
test['min_inter_topic_intersection_values'] = test['inter_topic_intersection_values'].apply(np.min)
test['max_inter_topic_intersection_values'] = test['inter_topic_intersection_values'].apply(np.max)
test['mean_inter_topic_intersection_values'] = test['inter_topic_intersection_values'].apply(np.mean)
test['std_inter_topic_intersection_values'] = test['inter_topic_intersection_values'].apply(np.std)

del train_label['topic'], train_label['follow_topic'], train_label['inter_topic'], train_label['follow_topic_intersection'], train_label['inter_topic_intersection']
del test['topic'], test['follow_topic'], test['inter_topic'], test['follow_topic_intersection'], test['inter_topic_intersection']
del train_label['inter_topic_intersection_values'], test['inter_topic_intersection_values']
print('3 finished')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
data = pd.concat((train_label, test), axis=0, sort=True)
# del train_label, test

# count编码
count_fea = ['uid_enc', 'qid_enc', 'gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5']
for feat in count_fea:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
    # 
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

data['wk'] = data['day'] % 7

feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
logging.info("feature size %s", len(feature_cols))

X_train_all = data.iloc[:len(train_label)][feature_cols]
y_train_all = data.iloc[:len(train_label)]['label']
test = data.iloc[len(train_label):]
del data
assert len(test) == sub_size

logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    break

X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols], \
                                 y_train_all.iloc[train_idx], \
                                 y_train_all.iloc[val_idx]
del X_train_all

model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=64, learning_rate=0.01, n_estimators=5000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=-1, silent=True)
model_lgb.fit(X_train, y_train,
              eval_metric=['logloss', 'auc'],
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50)

sub['label'] = model_lgb.predict_proba(test[feature_cols])[:, 1]


sub.to_csv('submit2/result6.txt', index=None, header=None, sep='\t')