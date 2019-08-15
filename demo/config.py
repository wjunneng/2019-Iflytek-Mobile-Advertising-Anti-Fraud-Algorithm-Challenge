# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os


class DefaultConfig(object):
    """
    参数配置
    """

    def __init__(self):
        pass

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    # round1_iflyad_anticheat_testdata_feature.txt文件路径
    testdata_feature_path = project_path + '/data/original/round1_iflyad_anticheat_testdata_feature.txt'
    # round1_iflyad_anticheat_traindata.txt文件路径
    traindata_path = project_path + '/data/original/round1_iflyad_anticheat_traindata.txt'

    # submit file
    submit_lgb_path = project_path + '/data/submit/submit_lgb.csv'
    submit_xgb_path = project_path + '/data/submit/submit_xgb.csv'

    # select_model
    select_model = ['lgb']

    # cache
    traindata_cache_path = project_path + '/data/cache/traindata.h5'
    testdata_cache_path = project_path + '/data/cache/testdata_feature.h5'
    label_cache_path = project_path + '/data/cache/label.h5'
    lgb_feature_cache_path = project_path + '/data/cache/feature_lgb.h5'
    xgb_feature_cache_path = project_path + '/data/cache/feature_xgb.h5'

    # foreign
    china_city_list_path = project_path + '/data/foreign/china_city_list.json'
    # train foreign
    traindata_ip_path = project_path + '/data/foreign/traindata_ip.json'
    # test foreign
    testdata_feature_ip_path = project_path + '/data/foreign/testdata_feature_ip.json'

    # save
    save = True

    # 去除的columns
    delete_columns = ['sid', 'label', 'ip', 'reqrealip', 'nginxtime', 'begintime']


if __name__ == '__main__':
    config = DefaultConfig()
