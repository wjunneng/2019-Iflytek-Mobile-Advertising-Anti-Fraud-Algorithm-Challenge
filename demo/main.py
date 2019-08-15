import time
import os
from util import *
from config import DefaultConfig


def main():
    start = time.clock()

    not_replace = False

    if os.path.exists(DefaultConfig.traindata_cache_path) and os.path.exists(DefaultConfig.testdata_cache_path) and not_replace:
        # 训练集
        train = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.traindata_cache_path, mode='r', key='train'))
        # 标签
        label = pd.read_hdf(path_or_buf=DefaultConfig.label_cache_path, mode='r', key='label')
        # 测试集
        test = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.testdata_cache_path, mode='r', key='test'))
    else:
        # 获取训练集
        traindata = get_traindata()
        # 获取测试集
        testdata_feature = get_testdata_feature()

        # 测试集label 置-1
        testdata_feature = add_testdata_label(testdata_feature)
        print('测试集label 置-1 耗时： %s \n' % str(time.clock() - start))

        # 请求会话时间
        testdata_feature = add_begintime(testdata_feature)
        # 请求会话时间 与 请求到达服务时间的差
        testdata_feature = add_nginxtime_begintime(testdata_feature)
        print('测试集 请求会话时间 与 请求到达服务时间的差 耗时： %s \n' % str(time.clock() - start))

        # 请求会话时间
        traindata = add_begintime(traindata)
        # 请求会话时间 与 请求到达服务时间的差
        traindata = add_nginxtime_begintime(traindata)
        print('训练集 请求会话时间 与 请求到达服务时间的差 耗时： %s \n' % str(time.clock() - start))

        # 合并训练、测试集
        data = merge_train_test_data(traindata, testdata_feature)
        print('合并训练、测试集 耗时： %s \n' % str(time.clock() - start))

        # 媒体信息：处理app所属类别    【up：0.01】
        # data = deal_apptype(data)
        # print('处理app 耗时： %s \n' % str(time.clock() - start))

        # IP信息：处理城市和省份  【up:0.01】
        data = deal_city_province(data)
        print('处理城市和省份 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：处理横竖屏   记住此处有删除数据的部分,后面划分训练集和测试集要小心.  【up:0.009】
        data = deal_orientation(data)
        print('训练集 处理横竖屏 耗时： %s \n' % str(time.clock() - start))

        # # 时间：处理时间 效果不好    【down:0.0011】
        # data = conversion_time(data, ['nginxtime', 'begintime'])
        # print('数据集 处理时间 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：处理idfamd5    【up:0.01】
        data = deal_idfamd5(data)
        print('数据集 设备信息 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：处理设备类型   【up:0.001】
        data = deal_dvctype(data)
        print('数据集 设备类型 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：处理网络类型   【up:0.001】
        data = deal_ntt(data)
        print('数据集 处理网络类型 耗时： %s \n' % str(time.clock() - start))

        # 设备信息:处理运营商   【up：0.014】
        data = deal_carrier(data)
        print('数据集 处理运营商 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：处理操作系统   【up:0.05】
        data = deal_os(data)
        print('数据集 处理操作系统 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：语言   【up:0.027】
        data = deal_lan(data)
        print('数据集 处理语言 耗时： %s \n' % str(time.clock() - start))

        # 设备信息：长/宽/密度   mean【down0.01】/median【比mean好0.01】/mode【比median好0.01】   area【】/aspect_ratio【】
        data = deal_h_w_ppi(data, 'mode')
        print('数据集 长/宽/密度 耗时： %s \n' % str(time.clock() - start))

        # 对除了'sid'外的columns进行one_hot编码
        data = one_hot_col(data)
        print('\n对除了“sid”外的columns进行one_hot编码 耗时： %s \n' % str(time.clock() - start))

        # 划分数据：
        train = data[data['label'] != -1]
        label = train['label']

        test = data[data['label'] == -1].reset_index(drop=True)
        print('划分数据 耗时： %s \n' % str(time.clock() - start))

        print('训练集和测试集 shape: %s \n' % str(train.shape) + str(test.shape))

        if DefaultConfig.save:
            train.to_hdf(path_or_buf=DefaultConfig.traindata_cache_path, key='train')
            label.to_hdf(path_or_buf=DefaultConfig.label_cache_path, key='label')
            test.to_hdf(path_or_buf=DefaultConfig.testdata_cache_path, key='test')

    print('\n加载数据 耗时： %s \n' % str(time.clock() - start))
    # 模型预测
    model_predict(train, label.values, test)
    print('模型预测 耗时： %s \n' % str(time.clock() - start))

    # 绘制特征重要图
    draw_feature(DefaultConfig.select_model)


if __name__ == '__main__':
    main()
