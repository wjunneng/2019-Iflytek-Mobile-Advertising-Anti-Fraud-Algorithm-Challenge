import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn import preprocessing

from demo.config import DefaultConfig


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

    # # 归一化函数
    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    #
    # df['nginxtime-begintime'] = df[['nginxtime-begintime']].apply(max_min_scaler)

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
        df[column] = df[column].apply(
            lambda x: pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(int(x) / 1000)))))
        df[column + '_hour'] = df[column].dt.hour.astype('int')
        df[column + '_day'] = df[column].dt.day.astype('int')

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


def deal_sid(df, **params):
    """
    处理sid
    :param df:
    :param params:
    :return:
    """
    df['sid'] = df['sid'].apply(lambda x: '-'.join(x.split('-')[:-1]))

    return df


def deal_ver(df, **params):
    """
    删除ver
    :param df:
    :param params:
    :return:
    """
    import re

    df['ver'] = df['ver'].apply(lambda x: ''.join(re.findall(r"\d+\.", str(x))))

    ver_1 = []
    ver_2 = []
    ver_3 = []
    for ver in df['ver'].values:
        tmp = ver.split('.')

        if len(tmp) > 0 and tmp[0].isdigit():
            ver_1.append(tmp[0])
        else:
            ver_1.append('-1')

        if len(tmp) > 1 and tmp[1].isdigit():
            ver_2.append(tmp[1])
        else:
            ver_2.append('-1')

        if len(tmp) > 2 and tmp[2].isdigit():
            ver_3.append(tmp[2])
        else:
            ver_3.append('-1')

    df['ver_1'] = ver_1
    df['ver_2'] = ver_2
    df['ver_3'] = ver_3

    del df['ver']

    return df


def deal_apptype(df, **params):
    """
    处理apptype
    :param df:
    :param params:
    :return:
    """
    df['apptype'] = df['apptype'].apply(lambda x: str(x))

    return df


def deal_city_province(df, **params):
    """
    处理城市和省份
    :param df:
    :param params:
    :return:
    """
    import json

    # 中国省市编码地址
    china_city_list_path = DefaultConfig.china_city_list_path

    # 省
    provinces = {}

    # 市
    cities = {}

    # 读取json
    with open(china_city_list_path, 'r') as file:
        # 读取所有省市数据
        china_city_list = json.load(file)

        for province in china_city_list:
            # key:province/value:code
            provinces[province['province']] = province['code']

            for city in province['cities']:
                # key:city/value:code
                cities[city['name']] = city['code']

    china_city_list = df['city']
    china_city_list = china_city_list.fillna(0)

    china_city_list_code = []
    china_province_list_code = []

    # 遍历获取code
    for city in china_city_list.values:
        # 是否在provinces中
        if city in dict.fromkeys(provinces.keys(), True):
            tmp = provinces[city]
            china_city_list_code.append(tmp)
            china_province_list_code.append(tmp[:2])

        # 是否在cities中
        elif city in dict.fromkeys(cities.keys(), True):
            tmp = cities[city]
            china_city_list_code.append(tmp)
            china_province_list_code.append(tmp[:2])

        # 不存在
        else:
            china_city_list_code.append('100000')
            china_province_list_code.append('10')

            if city != 0 and city != u'自治区直辖县级行政区划':
                print(city)

    # 替换
    df['city'] = china_city_list_code
    df['province'] = china_province_list_code

    return df


def deal_orientation(df, **params):
    """
    剔除非0和1的数据 【即非横/竖屏的数据】
    :param df:
    :param params:
    :return:
    """
    # 转化 orientation
    df['orientation'] = df['orientation'].replace(float(90.0), float(0.0))
    df['orientation'] = df['orientation'].replace(float(2.0), float(0.0))

    return df


def deal_idfamd5(df, **params):
    """
    处理idfamd5
    :param df:
    :param params:
    :return:
    """
    del df['idfamd5']

    return df


def deal_dvctype(df, **params):
    """
    返回设备类型
    :param df:
    :param params:
    :return:
    """
    df['dvctype'] = df['dvctype'].apply(lambda x: str(x))

    return df


def deal_model(df, **params):
    """
    处理model
    :param df:
    :param params:
    :return:
    """
    # 转换为小写
    df['model'] = df['model'].apply(lambda x: str(x).lower())

    result = []
    for model in df['model'].values:
        for type in [' ', '-', '+', '_', '%']:
            model = model.split(type)[-1].lower()

        makes = ['oppo', 'vivo', 'huawei', 'xiaomi', 'honor', 'meizu', 'smartisan', 'samsung', 'gionee', '360',
                 'oneplus',
                 'nubia', 'lenovo', 'meitu', 'zte', 'bbk', 'hisense', 'coolpad', 'generic', 'nokia', 'lemobile',
                 'letv', 'sony', 'cmcc', 'htc', 'xiaolajiao', 'blackshark', 'blephone', 'leeco', 'doov', 'gree', 'geli',
                 'koobee', 'motorola', 'hmd global', 'alps', 'k-touch', 'yulong', 'konka', 'rockchip',
                 'sugar', 'sharp', 'zuk', 'gree', 'changhong', 'lephone', 'tcl']

        for make in dict.fromkeys(makes, True):
            if make in model:
                model = model.replace(make, '')

        result.append(model)

    df['model'] = result

    return df


def deal_make(df, **params):
    """
    处理make
    :param df:
    :param params:
    :return:
    """
    # 转换为小写
    df['make'] = df['make'].apply(lambda x: str(x).lower())

    # 处理none
    tmp = []
    for index in range(df.shape[0]):
        make = str(df.ix[index, 'make'])
        if make != 'none':
            tmp.append(make)
        else:
            model = str(df.ix[index, 'model'])

            for type in [' ', '-', '+', '_', '%']:
                model = model.split(type)[0].lower()

            tmp.append(model.lower())

    # 替换
    df['make'] = tmp

    # 中文转英文
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'华为' else x)
    df['make'] = df['make'].apply(lambda x: '360' if x == u'360手机' else x)
    df['make'] = df['make'].apply(lambda x: 'gionee' if x == u'金立' else x)
    df['make'] = df['make'].apply(lambda x: 'samsung' if x == u'三星' else x)
    df['make'] = df['make'].apply(lambda x: 'xiaomi' if x == u'小米' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'荣耀' else x)
    df['make'] = df['make'].apply(lambda x: 'meizu' if x == u'魅族' else x)
    df['make'] = df['make'].apply(lambda x: 'coolpad' if x == u'酷派' else x)
    df['make'] = df['make'].apply(lambda x: 'nubia' if x == u'努比亚' else x)
    df['make'] = df['make'].apply(lambda x: '360' if x == u'奇酷' else x)
    df['make'] = df['make'].apply(lambda x: 'oneplus' if x == u'一加' else x)
    df['make'] = df['make'].apply(lambda x: 'zte' if x == u'中兴' else x)
    df['make'] = df['make'].apply(lambda x: 'meitu' if x == u'美图' else x)
    df['make'] = df['make'].apply(lambda x: 'smartisan' if x == u'锤子' else x)
    df['make'] = df['make'].apply(lambda x: 'blephone' if x == u'百立丰' else x)
    df['make'] = df['make'].apply(lambda x: 'xiaolajiao' if x == u'小辣椒' else x)
    df['make'] = df['make'].apply(lambda x: 'cmcc' if x == u'中国移动' else x)
    df['make'] = df['make'].apply(lambda x: 'lenovo' if x == u'联想' else x)
    df['make'] = df['make'].apply(lambda x: 'leeco' if x == u'乐视' else x)
    df['make'] = df['make'].apply(lambda x: 'gree' if x == u'格力' else x)
    df['make'] = df['make'].apply(lambda x: 'motorola' if x == u'摩托罗拉' else x)
    df['make'] = df['make'].apply(lambda x: 'doov' if x == u'朵唯' else x)
    df['make'] = df['make'].apply(lambda x: 'nokia' if x == u'诺基亚' else x)
    df['make'] = df['make'].apply(lambda x: 'hisense' if x == u'海信' else x)
    df['make'] = df['make'].apply(lambda x: 'sony' if x == u'索尼' else x)
    df['make'] = df['make'].apply(lambda x: 'sop' if x == u'赛博宇华' else x)
    df['make'] = df['make'].apply(lambda x: 'fox' if x == u'云狐' else x)
    df['make'] = df['make'].apply(lambda x: 'k-touch' if x == u'天语' else x)
    df['make'] = df['make'].apply(lambda x: 'yusun' if x == u'语信' else x)
    df['make'] = df['make'].apply(lambda x: 'lephone' if x == u'乐丰' else x)
    df['make'] = df['make'].apply(lambda x: 'bbk' if x == u'步步高' else x)
    df['make'] = df['make'].apply(lambda x: 'blackshark' if x == u'黑鲨' else x)
    df['make'] = df['make'].apply(lambda x: 'koobee' if x == u'酷比' else x)
    df['make'] = df['make'].apply(lambda x: 'realme' if x == u'真我' else x)

    # 型号  [考虑删除前后对比]
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'eml-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'par-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'cor-al10' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'hma-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'clt-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'ine-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'jsn-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'lya-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'jsn-al00a' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'ane-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'vce-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'lld-al20' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'pct-al10' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'ele-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'jkm-al00b' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'jkm-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'dub-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'evr-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'hry-al00a' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'lld-al30' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'clt-al01' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'lya-al10' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'are-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'vog-al10' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'rvl-al09' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'sne-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'ane-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'ine-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'vog-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'mar-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'ars-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'cor-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'jkm-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'bkk-al10' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'jsn-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'jkm-al00a' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'hry-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'ane-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'hma-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'tny-al00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'par-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if x == u'jat-tl00' else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if x == u'dut-tl00' else x)

    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pbam00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pbem00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pacm00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pbbm00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'paam00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'padm00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pact00' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pbcm10' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'pbet00' in str(x) else x)

    df['make'] = df['make'].apply(lambda x: 'xiaomi' if 'mi' in str(x) else x)

    df['make'] = df['make'].apply(lambda x: 'hisense' if 'h==ense' in str(x) else x)

    # 判断是否存在
    df['make'] = df['make'].apply(lambda x: 'oppo' if 'oppo' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'vivo' if 'vivo' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'huawei' if 'huawei' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'xiaomi' if 'xiaomi' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'honor' if 'honor' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'meizu' if 'meizu' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'smartisan' if 'smartisan' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'samsung' if 'samsung' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'gionee' if 'gionee' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: '360' if '360' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'oneplus' if 'oneplus' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'nubia' if 'nubia' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'lenovo' if 'lenovo' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'meitu' if 'meitu' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'zte' if 'zte' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'bbk' if 'bbk' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'hisense' if 'hisense' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'coolpad' if 'coolpad' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'generic' if 'generic' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'nokia' if 'nokia' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'lemobile' if 'lemobile' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'letv' if 'letv' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'sony' if 'sony' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'cmcc' if 'cmcc' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'htc' if 'htc' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'xiaolajiao' if 'xiaolajiao' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'blackshark' if 'blackshark' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'blephone' if 'blephone' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'leeco' if 'leeco' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'doov' if 'doov' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'gree' if 'gree' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'gree' if 'geli' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'koobee' if 'koobee' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'motorola' if 'motorola' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'hmd global' if 'hmd global' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'alps' if 'alps' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'k-touch' if 'k-touch' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'yulong' if 'yulong' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'konka' if 'konka' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'rockchip' if 'rockchip' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'sugar' if 'sugar' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'sharp' if 'sharp' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'zuk' if 'zuk' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'gree' if 'geli' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'changhong' if 'changhong' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'lephone' if 'lephone' in str(x) else x)
    df['make'] = df['make'].apply(lambda x: 'tcl' if 'tcl' in str(x) else x)

    make = ['oppo', 'vivo', 'huawei', 'xiaomi', 'honor', 'meizu', 'smartisan', 'samsung', 'gionee', '360', 'oneplus',
            'nubia', 'lenovo', 'meitu', 'zte', 'bbk', 'hisense', 'coolpad', 'generic', 'nokia', 'lemobile',
            'letv', 'sony', 'cmcc', 'htc', 'xiaolajiao', 'blackshark', 'blephone', 'leeco', 'doov', 'gree', 'geli',
            'koobee', 'motorola', 'hmd global', 'alps', 'k-touch', 'yulong', 'konka', 'rockchip',
            'sugar', 'sharp', 'zuk', 'gree', 'changhong', 'lephone', 'tcl']

    df['make'] = df['make'].apply(lambda x: 'none' if str(x) not in make else x)

    return df


def deal_ntt(df, **params):
    """
    处理ntt
    :param df:
    :param params:
    :return:
    """
    df['ntt'] = df['ntt'].apply(lambda x: str(x))
    # df['ntt_type'] = df['ntt'].apply(lambda x: str(2) if int(float(x)) in [2, 3, 4, 5, 6, 7] else x)

    return df


def deal_carrier(df, **params):
    """
    处理运营商
    :param df:
    :param params:
    :return:
    """
    df['carrier'] = df['carrier'].apply(lambda x: str(0) if int(x) == 0 else x)
    df['carrier'] = df['carrier'].apply(lambda x: str(46000) if int(x) == 46000 else x)
    df['carrier'] = df['carrier'].apply(lambda x: str(46001) if int(x) == 46001 else x)
    df['carrier'] = df['carrier'].apply(lambda x: str(46003) if int(x) == 46003 else x)
    df['carrier'] = df['carrier'].apply(
        lambda x: str(1) if str(x) not in [str(0), str(46000), str(46001), str(46003)] else x)

    return df


def deal_os(df, **params):
    """
    处理操作系统
    :param df:
    :param params:
    :return:
    """
    if 'os' in df.columns:
        # 先将所有值转化为小写
        df['os'] = df['os'].apply(lambda x: x.lower())
        # 如果所有值都相同，则剔除
        if len(set(df['os'])) is 1:
            del df['os']

    return df


def deal_osv(df, **params):
    """
    处理操作系统版本
    :param df:
    :param params:
    :return:
    """
    import re

    df['osv'] = df['osv'].apply(lambda x: ''.join(re.findall(r"\d+", str(x))))

    osv_1 = []
    osv_2 = []
    osv_3 = []
    for osv in df['osv'].values:
        if len(osv) < 3:
            tmp = osv.rjust(3, '*')
            osv_1.append(tmp[0])
            osv_2.append(tmp[1])
            osv_3.append(tmp[2])
        else:
            osv_1.append(osv[0])
            osv_2.append(osv[1])
            osv_3.append(osv[2])

    df['osv_1'] = osv_1
    df['osv_2'] = osv_2
    df['osv_3'] = osv_3

    del df['osv']

    return df


def deal_lan(df, **params):
    """
    处理语言
    :param df:
    :param params:
    :return:
    """
    df['lan'] = df['lan'].fillna(0)

    df['lan'] = df['lan'].apply(lambda x: str(x).lower())
    # 优先处理cn: 1
    df['lan'] = df['lan'].apply(lambda x: '1' if 'cn' in str(x) else x)
    # 处理tw： 2
    df['lan'] = df['lan'].apply(lambda x: '2' if 'tw' in str(x) else x)
    # 处理hk: 2
    df['lan'] = df['lan'].apply(lambda x: '2' if 'hk' in str(x) else x)
    # 处理en: 3
    df['lan'] = df['lan'].apply(lambda x: '3' if 'en' in str(x) else x)
    # 处理zh: 1
    df['lan'] = df['lan'].apply(lambda x: '1' if 'zh' in str(x) else x)
    # 其他： 4
    df['lan'] = df['lan'].apply(lambda x: '4' if not x in ['0', '1', '2', '3'] else x)

    return df


def deal_h_w_ppi(df, fillna_type, **params):
    """
    处理宽高/密度
    :param df:
    :param params:
    :return:
    """
    import numpy as np

    if fillna_type is 'mean':
        # 利用均值填充
        df['h'].replace(0, df['h'].mean(), inplace=True)
        df['w'].replace(0, df['w'].mean(), inplace=True)
        df['ppi'].replace(0, df['ppi'].mean(), inplace=True)

    elif fillna_type is 'median':
        # 利用均值填充
        df['h'].replace(0, df['h'].median(), inplace=True)
        df['w'].replace(0, df['w'].median(), inplace=True)
        df['ppi'].replace(0, df['ppi'].median(), inplace=True)

    elif fillna_type is 'mode':
        # 利用众数填充
        df['h'].replace(0, df['h'].mode().max(), inplace=True)
        df['w'].replace(0, df['w'].mode().max(), inplace=True)
        df['ppi'].replace(0, df['ppi'].mode().max(), inplace=True)

    # 归一化函数
    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    # # 添加面积列
    df['h_w'] = df['h'] * df['w']
    df['h_2'] = df['h'] * df['h']
    df['w_2'] = df['w'] * df['w']

    # # 添加宽高列
    # df['aspect_ratio'] = df['h'] / df['w']

    # df['h'] = df[['h']].apply(max_min_scaler)
    # df['w'] = df[['w']].apply(max_min_scaler)
    # df['ppi'] = df[['ppi']].apply(max_min_scaler)
    # df['area'] = df[['area']].apply(max_min_scaler)
    # df['aspect_ratio'] = df[['aspect_ratio']].apply(max_min_scaler)

    return df


def one_hot_col(df, **params):
    """
    标签编码
    :param df:
    :return:
    """
    object_cols = list(df.dtypes[df.dtypes == np.object].index)

    print('one_hot 处理的特征列： %s' % ' '.join(object_cols))
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
    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    oof_xgb = np.zeros(new_train.shape[0])
    prediction_xgb = np.zeros(new_test.shape[0])
    cv_model = []
    for i, (tr, va) in enumerate(skf.split(new_train, y)):
        print('fold:', i + 1, 'training')
        dtrain = xgb.DMatrix(new_train[tr], y[tr])
        dvalid = xgb.DMatrix(new_train[va], y[va])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
        bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=1000,
                        verbose_eval=50, params=xgb_params)

        cv_model.append(bst)

        oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
        prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)

    print('the roc_auc_score for train:', roc_auc_score(y, oof_xgb))
    prediction_xgb /= n_splits
    return oof_xgb, prediction_xgb, cv_model


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
    lgb_params = {
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
    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    oof_lgb = np.zeros(new_train.shape[0])  # 用于存放训练集概率，由每折验证集所得
    prediction_lgb = np.zeros(new_test.shape[0])  # 用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame()  # 存放特征重要性
    for i, (tr, va) in enumerate(skf.split(new_train, y)):
        print('fold:', i + 1, 'training')
        dtrain = lgb.Dataset(new_train[tr], y[tr])
        dvalid = lgb.Dataset(new_train[va], y[va], reference=dtrain)
        # 训练：
        bst = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=30000, valid_sets=dvalid, verbose_eval=400,
                        early_stopping_rounds=1000)
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

    prediction_lgb /= n_splits
    return oof_lgb, prediction_lgb, feature_importance_df


def cgb_model(new_train, y, new_test, columns, **params):
    """
    cgb 模型
    :param new_train:
    :param y:
    :param new_test:
    :param columns:
    :param params:
    :return:
    """
    from catboost import CatBoostClassifier

    cgb_params = {
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 5.0,
        'loss_function': 'Logloss',
        'iterations': 946,
        'custom_metric': 'F1',
        'eval_metric': 'F1',
        'random_seed': 2019,
        'logging_level': 'Silent',
        'thread_count': 10
    }

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    oof_cgb = np.zeros(new_train.shape[0])  # 用于存放训练集概率，由每折验证集所得
    prediction_cgb = np.zeros(new_test.shape[0])  # 用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame()  # 存放特征重要性
    for i, (tr, va) in enumerate(skf.split(new_train, y)):
        # 数据
        y_train, y_valid = y.iloc[tr], y.iloc[va]
        X_train, X_valid = new_train.iloc[tr, :], new_train.iloc[va, :]

        # classifier参数设置
        model = CatBoostClassifier(**cgb_params)

        print('fold:', i + 1, 'training')
        # 训练：
        bst = model.fit(X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        use_best_model=True
                        )
        print("N trees = ", model.tree_count_)
        # 预测验证集：
        oof_cgb[va] += bst.predict(new_train.iloc[va, :])
        # 预测测试集：
        prediction_cgb += bst.predict(new_test)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = columns
        fold_importance_df["importance"] = bst.feature_importances_
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('the roc_auc_score for train:', roc_auc_score(y, oof_cgb))  # 线下auc评分

    prediction_cgb /= n_splits
    return oof_cgb, prediction_cgb, feature_importance_df


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

    elif model is 'cgb':
        sub.to_csv(DefaultConfig.submit_cgb_path, index=None)


def model_predict(traindata, label, testdata, **params):
    """
    模型预测与结果保存
    :param traindata:
    :param testdata:
    :param label:
    :param params:
    :return:
    """
    train = np.array(traindata.drop(DefaultConfig.delete_columns, axis=1).astype(int))
    test = np.array(testdata.drop(DefaultConfig.delete_columns, axis=1).astype(int))
    columns = [i for i in traindata.columns if i not in DefaultConfig.delete_columns]

    for model in list(DefaultConfig.select_model):
        if model is 'lgb':
            print('model is :', model)
            # 模型训练预测：
            oof_lgb, prediction_lgb, feature_importance_df = lgb_model(train, label, test, columns)

            # 保存feature_importance_df
            feature_importance_df.to_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key='lgb')

            # 保存结果
            save_result(model, testdata, prediction_lgb)

        elif model is 'xgb':
            print('model is :', model)
            # 模型训练预测：
            oof_lgb, prediction_xgb, cv_model = xgb_model(train, label, test, columns)

            fi = []
            for i in cv_model:
                tmp = {
                    'name': columns,
                    'score': i.booster().get_score(importance_type='weight')
                }
                fi.append(pd.DataFrame(tmp))

            fi = pd.concat(fi)
            # 保存feature_importance_df
            fi.to_hdf(path_or_buf=DefaultConfig.xgb_feature_cache_path, key='xgb')

            # 保存结果
            save_result(model, testdata, prediction_xgb)

        elif model is 'cgb':
            print('model is :', model)
            # 模型训练预测：
            oof_cgb, prediction_cgb, feature_importance_df = cgb_model(
                traindata.drop(DefaultConfig.delete_columns, axis=1), pd.Series(label),
                testdata.drop(DefaultConfig.delete_columns, axis=1), columns)

            # 保存feature_importance_df
            feature_importance_df.to_hdf(path_or_buf=DefaultConfig.cgb_feature_cache_path, key='cgb')

            # 保存结果
            save_result(model, testdata, prediction_cgb)


def draw_feature(models, **params):
    """
    绘制特征重要度
    :param model:
    :param params:
    :return:
    """
    for model in models:
        if os.path.exists(DefaultConfig.lgb_feature_cache_path) and model is 'lgb':
            # 读取feature_importance_df
            feature_importance_df = reduce_mem_usage(
                pd.read_hdf(path_or_buf=DefaultConfig.lgb_feature_cache_path, key=model, mode='r'))

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

        if os.path.exists(DefaultConfig.xgb_feature_cache_path) and model is 'xgb':
            # 读取feature_importance_df
            feature_importance_df = reduce_mem_usage(
                pd.read_hdf(path_or_buf=DefaultConfig.xgb_feature_cache_path, key=model, mode='r'))

            plt.figure(figsize=(8, 8))
            feature_importance_df.groupby(['name'])['score'].agg('mean').sort_values(ascending=False).head(
                40).plot.barh()
            plt.show()

        if os.path.exists(DefaultConfig.cgb_feature_cache_path) and model is 'cgb':
            # 读取feature_importance_df
            feature_importance_df = reduce_mem_usage(
                pd.read_hdf(path_or_buf=DefaultConfig.cgb_feature_cache_path, key=model, mode='r'))

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


def merge(**params):
    project_path = DefaultConfig.project_path

    lgb_path = project_path + '/data/submit/submit_lgb.csv'
    xgb_path = project_path + '/data/submit/submit_xgboost.csv'
    cgb_path = project_path + '/data/submit/judge_by_catboost.csv'

    lgb = pd.read_csv(lgb_path)
    xgb = pd.read_csv(xgb_path)
    cgb = pd.read_csv(cgb_path)

    label = []
    for i in range(cgb.shape[0]):
        one = 0
        zero = 0

        if int(lgb.ix[i, 'label']) is 1:
            one += 1
        else:
            zero += 1

        if int(xgb.ix[i, 'label']) is 1:
            one += 1
        else:
            zero += 1

        if int(cgb.ix[i, 'label']) is 1:
            one += 1
        else:
            zero += 1

        if one > zero:
            label.append(1)
        else:
            label.append(0)

    cgb['label'] = label

    cgb.to_csv(project_path + '/data/submit/merge.csv', index=None)


if __name__ == '__main__':
    merge()




