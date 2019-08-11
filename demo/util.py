import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from demo.config import DefaultConfig
from sklearn import preprocessing


def get_testdata_feature(**params):
    """
    返回 testdata_feature 文件内容
    :param params:
    :return:
    """
    testdata_feature_path = DefaultConfig.testdata_feature_path
    testdata_feature_data = pd.read_table(testdata_feature_path, sep='\t')

    return testdata_feature_data


def get_traindata(**params):
    """
    返回 traindata 文件内容
    :param params:
    :return:
    """
    traindata_path = DefaultConfig.traindata_path
    traindata_data = pd.read_table(traindata_path, sep='\t')

    return traindata_data


def add_testdata_label(df, **params):
    """
    添加label 列
    :param df:
    :param params:
    :return:
    """
    df['label'] = -1

    return df


def add_begintime(df, **params):
    """
    添加begin time列  请求会话时间
    :param df:
    :param params:
    :return:
    """
    df['begintime'] = df['sid'].apply(lambda x: int(x.split('-')[-1]))

    return df


def add_nginxtime_begintime(df, **params):
    """
    获取请求到达服务时间-请求会话时间
    :param df:
    :param params:
    :return:
    """
    df['nginxtime-begintime'] = df['nginxtime'] - df['begintime']

    return df


def conversion_time(df, columns, **params):
    """
    对会话开始和结束时间进行标准化
    :param df:
    :param params:
    :return:
    """
    for column in columns:
        # 本题所给时间戳为毫秒级，故需除以1000转换为秒级：时间戳转成日期格式
        df[column] = df[column].apply(lambda x: pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(int(x) / 1000)))))
        df[column + '_hour'] = df[column].dt.hour.astype('int')
        df[column + '_quarter'] = df[column].dt.quarter.astype('int')
        df[column + '_day'] = df[column].dt.day.astype('int')
        df[column + '_month'] = df[column].dt.month.astype('int')
        df[column + '_year'] = df[column].dt.year.astype('int')

    return df


def merge_train_test_data(train, test):
    """
    合并训练和测试集
    :param train:
    :param test:
    :return:
    """
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    return df


def one_hot_col(df, **params):
    """
    标签编码
    :param df:
    :return:
    """
    object_cols = list(df.dtypes[df.dtypes == np.object].index)

    lbl = preprocessing.LabelEncoder()
    for col in object_cols:
        if col != 'sid':
            df[col] = lbl.fit(df[col].astype('str')).transform(df[col].astype('str'))

    return df


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def xgb_model(new_train, y, new_test, columns, **params):
    """
    xgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    xgb_params = {'booster': 'gbtree',
                  'eta': 0.01, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'silent': True,
                  }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    oof_xgb = np.zeros(new_train.shape[0])
    prediction_xgb = np.zeros(new_test.shape[0])
    feature_importance_df = pd.DataFrame()  # 存放特征重要性
    for i, (tr, va) in enumerate(skf.split(new_train, y)):
        print('fold:', i + 1, 'training')
        dtrain = xgb.DMatrix(new_train[tr], y[tr])
        dvalid = xgb.DMatrix(new_train[va], y[va])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
        bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=50, params=xgb_params)
        oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
        prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = columns
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('the roc_auc_score for train:', roc_auc_score(y, oof_xgb))
    prediction_xgb /= 5
    return oof_xgb, prediction_xgb, feature_importance_df


def lgb_model(new_train, y, new_test, columns, **params):
    """
    lgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_leaves': 1000,
        'verbose': -1,
        'max_depth': -1,
        'seed': 42,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    oof_lgb = np.zeros(new_train.shape[0])  # 用于存放训练集概率，由每折验证集所得
    prediction_lgb = np.zeros(new_test.shape[0])  # 用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame()  # 存放特征重要性
    for i, (tr, va) in enumerate(skf.split(new_train, y)):
        print('fold:', i + 1, 'training')
        dtrain = lgb.Dataset(new_train[tr], y[tr])
        dvalid = lgb.Dataset(new_train[va], y[va], reference=dtrain)
        # 训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid, verbose_eval=400,
                        early_stopping_rounds=200)
        # 预测验证集：
        oof_lgb[va] += bst.predict(new_train[va], num_iteration=bst.best_iteration)
        # 预测测试集：
        prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = columns
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('the roc_auc_score for train:', roc_auc_score(y, oof_lgb))  # 线下auc评分
    prediction_lgb /= 5
    return oof_lgb, prediction_lgb, feature_importance_df


def save_result(model, testdata, prediction, **params):
    """
    保存结果
    :param model:
    :param testdata:
    :param prediction:
    :param params:
    :return:
    """
    # 保存结果：
    sub = testdata[['sid']]
    sub['label'] = prediction
    # ∪概率大于0.5的置1，否则置0
    sub['label'] = sub['label'].apply(lambda x: 1 if x > 0.5 else 0)
    # 模型预测测试集的标签分布
    print('test pre_label distribution:\n', sub['label'].value_counts())

    if model is 'lgb':
        sub.to_csv(DefaultConfig.submit_lgb_path, index=None)

    elif model is 'xgb':
        sub.to_csv(DefaultConfig.submit_xgb_path, index=None)


def model_predict(traindata, label, testdata, **params):
    """
    模型预测与结果保存
    :param traindata:
    :param testdata:
    :param label:
    :param params:
    :return:
    """
    train = np.array(traindata.drop(DefaultConfig.delete_columns, axis=1))
    test = np.array(testdata.drop(DefaultConfig.delete_columns, axis=1))
    columns = [i for i in traindata.columns if i not in DefaultConfig.delete_columns]

    for model in list(DefaultConfig.select_model):
        if model is 'lgb':
            # 模型训练预测：
            oof_lgb, prediction_lgb, feature_importance_df = lgb_model(train, label, test, columns)

            # 保存feature_importance_df
            feature_importance_df.to_hdf(path_or_buf=DefaultConfig.feature_cache_path, key='lgb')

            # 保存结果
            save_result(model, testdata, prediction_lgb)

        elif model is 'xgb':
            # 模型训练预测：
            oof_lgb, prediction_xgb, feature_importance_df = xgb_model(train, label, test, columns)

            # 保存结果
            save_result(model, testdata, prediction_xgb)


def draw_feature(models, **params):
    """
    绘制特征重要度
    :param model:
    :param params:
    :return:
    """
    for model in models:
        if os.path.exists(DefaultConfig.feature_cache_path):
            # 读取feature_importance_df
            feature_importance_df = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.feature_cache_path, key=model, mode='r'))

            plt.figure(figsize=(8, 8))
            # 按照flod分组
            group = feature_importance_df.groupby(by=['fold'])

            result = []
            for key, value in group:
                value = value[['feature', 'importance']]

                result.append(value)

            result = pd.concat(result)
            # 5折数据取平均值
            result.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(40).plot.barh()
            plt.show()