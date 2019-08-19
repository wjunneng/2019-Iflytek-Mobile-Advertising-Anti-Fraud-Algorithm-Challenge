import pandas
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from itertools import product, chain
from util import *
from config import *
import pandas as pd


def using_best_param(train, test, label):
    """
    使用最好的参数训练模型
    :param params:
    :return:
    """
    model = XGBClassifier(n_estimators=946,
                          max_depth=8,
                          learning_rate=0.05,
                          custom_metric='auc',
                          eval_metric='auc',
                          random_seed=2019,
                          objective='binary',
                          l2_leaf_reg=5.0,
                          silent=True,
                          nthread=11)
    model.fit(train, label)
    y_pred = model.predict(test).tolist()

    judge_df = pd.DataFrame()
    judge_df['sid'] = range(test.shape[0])
    judge_df['label'] = y_pred
    judge_df['label'] = judge_df['label'].apply(lambda x: 1 if x >= 0.5 else 0)

    return judge_df[['sid', 'label']]


if __name__ == '__main__':
    import time

    start = time.clock()
    # 训练集
    traindata = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.traindata_cache_path, mode='r', key='train'))
    # 标签
    label = pd.read_hdf(path_or_buf=DefaultConfig.label_cache_path, mode='r', key='label')
    # 测试集
    testdata = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.testdata_cache_path, mode='r', key='test'))

    # 删除无关特征
    train = traindata.drop(DefaultConfig.delete_columns, axis=1)
    test = testdata.drop(DefaultConfig.delete_columns, axis=1)

    train_label = pd.Series(label)

    judge_by_catboost = using_best_param(train=train, test=test, label=train_label)
    judge_by_catboost.to_csv(DefaultConfig.project_path + '/data/submit/submit_xgboost.csv', index=False,
                             encoding='utf-8')
    print(time.clock() - start)