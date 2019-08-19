import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import catboost as cbt
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import warnings
from util import DefaultConfig
from util import reduce_mem_usage

warnings.filterwarnings('ignore')

import time

start = time.clock()
# 训练集
traindata = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.test_traindata_cache_path, mode='r', key='train'))
# 标签
label = pd.read_hdf(path_or_buf=DefaultConfig.test_label_cache_path, mode='r', key='label')
# 测试集
testdata = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.test_testdata_cache_path, mode='r', key='test'))

# 删除无关特征
train = traindata.drop(DefaultConfig.delete_columns, axis=1)
test = testdata.drop(DefaultConfig.delete_columns, axis=1)

train_label = pd.Series(label)


# 以下代码是5折交叉验证的结果 + lgb catboost xgb 最后使用logist进行回归预测
def get_stacking(clf, x_train, y_train, x_test, feature_name, n_folds=5):
    print('len_x_train:', len(x_train))

    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[feature_name].iloc[train_index], y_train[train_index]
        x_tst, y_tst = x_train[feature_name].iloc[test_index], y_train[test_index]

        clf.fit(x_tra[feature_name], y_tra, eval_set=(x_tst[feature_name], y_tst))

        second_level_train_set[test_index] = clf.predict(x_tst[feature_name])
        test_nfolds_sets[:, i] = clf.predict(x_test[feature_name])

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


def lgb_f1(labels, preds):
    score = f1_score(labels, np.round(preds))
    return 'f1', score, True


lgb_model = lgb.LGBMClassifier(random_seed=2019, n_jobs=-1, objective='binary', learning_rate=0.05, n_estimators=3000,
                               num_leaves=31, max_depth=-1, min_child_samples=50, min_child_weight=9, subsample_freq=1,
                               subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=5, eval_metric=lgb_f1,
                               early_stopping_rounds=400)

xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=3000, silent=False, objective='binary',
                             early_stopping_rounds=400, feval=lgb_f1)

cbt_model = cbt.CatBoostClassifier(iterations=3000, learning_rate=0.05, max_depth=11, l2_leaf_reg=1, verbose=10,
                                   early_stopping_rounds=400, task_type='GPU', eval_metric='F1', cat_features=train.columns)

train_sets = []
test_sets = []
for clf in [cbt_model, lgb_model, xgb_model]:
    print('begin train clf:', clf)
    train_set, test_set = get_stacking(clf, train, label, test, train.columns)
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

# 使用逻辑回归作为第二层模型
bclf = LogisticRegression()
bclf.fit(meta_train, label)
test_pred = bclf.predict_proba(meta_test)[:, 1]

# 提交结果
submit = pd.Series(range(test.shape[0]))
submit['label'] = (test_pred >= 0.5).astype(int)
print(submit['label'].value_counts())
submit.to_csv("A_Simple_Stacking_Model.csv", index=False)

# 打印预测地概率 方便以后使用融合模型
df_sub = pd.concat([pd.Series(range(test.shape[0])), pd.Series(test_pred)], axis = 1)
df_sub.columns = ['sid', 'label']
df_sub.to_csv('A_Simple_Stacking_Model_proba.csv', sep=',', index=False)
