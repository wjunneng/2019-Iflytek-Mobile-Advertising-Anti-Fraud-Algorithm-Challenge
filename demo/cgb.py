import pandas
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from paramsearch import paramsearch
from itertools import product, chain
from util import *
from config import *
import pandas as pd

params = {'depth': [6, 5, 7, 8, 9, 10, 11],
          'iterations': [350, 200, 250, 300, 400, 500, 600, 700, 800, 900],
          'learning_rate': [0.03, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
          'l2_leaf_reg': [5, 3, 4, 6, 7, 8, 9, 10, 11, 20],
          'border_count': [32, 5, 4, 6, 8, 10, 20, 50, 100, 200],
          'thread_count': 10,
          'gpu_ram_part': 0.9,
          'custom_metric': 'F1',
          'eval_metric': 'F1',
          'random_seed': 2019,
          'logging_level': 'Silent',
          'task_type': "GPU"
          }


def crossvaltest(params, train_set, train_label, cat_dims, n_splits=3):
    """
    5折交叉验证
    :param params:
    :param train_set:
    :param train_label:
    :param cat_dims:
    :param n_splits:
    :return:
    """
    kf = KFold(n_splits=n_splits, shuffle=True)
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index, :]
        test = train_set.iloc[test_index, :]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = cb.CatBoostClassifier(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test) == np.ravel(test_labels)))
    return np.mean(res)


def catboost_param_tune(params, train_set, train_label, cat_dims=None, n_splits=5):
    """
    使用网格搜索参数
    :param params:
    :param train_set:
    :param train_label:
    :param cat_dims:
    :param n_splits:
    :return:
    """
    ps = paramsearch(params)
    # 独立寻找 'border_count', 'l2_leaf_reg' etc. 但是一起寻找 'iterations','learning_rate'
    for prms in chain(ps.grid_search(['border_count']),
                      ps.grid_search(['l2_leaf_reg']),
                      ps.grid_search(['iterations', 'learning_rate']),
                      ps.grid_search(['depth'])):
        res = crossvaltest(prms, train_set, train_label, cat_dims, n_splits)
        # 保存交叉验证结果，以便将来的迭代可以重用最佳参数
        ps.register_result(res, prms)
        print(res, prms, 'best:', ps.bestscore(), ps.bestparam())
        return ps.bestparam()


def using_best_param(train, test, label):
    """
    使用最好的参数训练模型
    :param params:
    :return:
    """
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, max_depth=7, cat_features=train.columns,verbose=100, custom_metric='F1', random_seed=2019,
                               early_stopping_rounds=200, task_type='CPU', thread_count=11,
                               eval_metric='F1')

    model.fit(train, label)
    y_pred = model.predict(test).tolist()

    judge_df = pd.DataFrame()
    judge_df['sid'] = range(test.shape[0])
    judge_df['label'] = y_pred
    judge_df['label'] = judge_df['label'].apply(lambda x: 1 if x >= 0.49 else 0)

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

    # 最好的参数
    # bestparams = catboost_param_tune(params=params, train_set=train, train_label=train_label, cat_dims=train.columns)
    #
    # # 利用训练集来调整参数
    # clf = cb.CatBoostClassifier(**bestparams)
    # clf.fit(train, np.ravel(pd.Series(label)), cat_features=train.columns)
    # res = clf.predict(test)
    # result = pd.DataFrame()
    # result['sid'] = range(len(testdata))
    # result['label'] = res
    # result.to_csv(DefaultConfig.project_path + '/data/submit/submit_catboost.csv', index=False, encoding='utf-8')
    # print(time.clock() - start)

    # iterations = 946, depth = 8, cat_features = features, learning_rate = 0.05,
    # custom_metric = 'F1', eval_metric = 'F1', random_seed = 2019,
    # l2_leaf_reg = 5.0, logging_level = 'Silent', thread_count = 11

    # params = {'task_type': 'GPU',
    #           'gpu_ram_part': 0.9,
    #           'thread_count': 10,
    #           'learning_rate': 0.03,
    #           'border_count': 5,
    #           'depth': 8,
    #           'random_seed': 2019,
    #           'l2_leaf_reg': 5,
    #           'iterations': 1000,
    #           'eval_metric': 'F1',
    #           'custom_metric': ['F1'],
    #           'logging_level': 'Silent',
    #           'loss_function': 'Logloss'}

    # using lgb feature
    select_feature = ['imeimd5',
                      'city',
                      'adidmd5',
                      'nginxtime_minute',
                      'ip2', 'nginxtime_hour',
                      'Fusion_attr_adunitshowid_model', 'ip3',
                      'macmd5',
                      'Fusion_attr_adunitshowid_ratio',
                      'nginxtime-begintime', 'nginxtime_day',
                      'Fusion_attr_make_adunitshowid',
                      'Fusion_attr_model_osv', 'mean_model_w', 'mean_model_h', 'model',
                      'var_model_h',
                      'adunitshowid', 'var_model_w', 'Fusion_attr_model_h', 'Fusion_attr_model_ratio',
                      'Fusion_attr_ratio_osv',
                      'Fusion_attr_make_model', 'var_adunitshowid_h', 'Fusion_attr_make_osv', 'mean_adunitshowid_h',
                      'Fusion_attr_make_ratio', 'var_adunitshowid_w', 'mean_adunitshowid_w', 'machine', 'mediashowid',
                      'begintime_minute', 'reqrealip3', 'reqrealip2', 'apptype', 'h', 'pkgname', 'osv_2', 'ntt',
                      'province',
                      'begintime_hour', 'size', 'mean_ver_h', 'mean_ver_w', 'carrier', 'px', 'osv_1', 'var_ver_h',
                      # 'openudidmd5',
                      'ver_2', 'var_ver_w', 'begintime_day', 'osv_3', 'make', 'ver_3', 'var_make_h', 'var_make_w',
                      'mean_make_h', 'mj',
                      'mean_make_w', 'w', 'ver_1', 'ppi', 'ratio', 'lan', 'dvctype',
                      # 'mean_adunitshowid_ratio',
                      'orientation'
                      # ,'mean_model_ratio'
                      ]
    train = train[select_feature]
    test = test[select_feature]

    train = train.astype(int)
    test = test.astype(int)
    judge_by_catboost = using_best_param(train=train, test=test, label=train_label)
    judge_by_catboost.to_csv(DefaultConfig.project_path + '/data/submit/submit_catboost.csv', index=False,
                             encoding='utf-8')
