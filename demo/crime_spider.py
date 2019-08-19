# coding: utf-8
import os
import urllib.request
import urllib.parse

import pandas as pd
from lxml import etree
from demo.util import get_traindata, get_testdata_feature
from demo.config import DefaultConfig
from demo.util import reduce_mem_usage
from bs4 import BeautifulSoup
import random
import requests


class CrimeSpider:
    """
    http://www.ip138.com/ 采集医疗数据
    """

    def __init__(self, df, column):
        self.df = df
        self.column = column

    def get_ip_list(self, url, headers):
        """
        获取ip列表
        :param headers:
        :return:
        """
        web_data = requests.get(url, headers=headers)
        soup = BeautifulSoup(web_data.text, 'lxml')
        ips = soup.find_all('tr')
        ip_list = []
        for i in range(1, len(ips)):
            ip_info = ips[i]
            tds = ip_info.find_all('td')
            ip_list.append(tds[1].text + ':' + tds[2].text)
        return ip_list

    def get_random_ip(self, ip_list):
        """
        随机ip
        :param ip_list:
        :return:
        """
        proxy_list = []
        for ip in ip_list:
            proxy_list.append('http://' + ip)
        proxy_ip = random.choice(proxy_list)
        proxies = {'http': proxy_ip}
        return proxies

    def get_html(self, url):
        """
        根据url, 请求html
        :param url:
        :return:
        """
        url = 'http://www.xicidaili.com/nn/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.60 Safari/537.17'
        }
        ip_list = self.get_ip_list(url, headers=headers)
        proxies = self.get_random_ip(ip_list)

        print(proxies)
        req = requests.get(url=url, headers=headers, proxies=proxies)
        res = urllib.request.urlopen(req)
        html = res.read().decode('gbk')
        return html

    def food_spider(self, html):
        """
        解析html
        :param url:
        :param html:
        :return:
        """
        selector = etree.HTML(html)
        tables = selector.xpath('//table[@align="center"]/tr[3]/td[@align="center"]/ul[@class="ul1"]')
        try:
            result = {}
            # key
            key = tables[0].xpath('./li/text()')[0].split('：')[0]
            # value
            value = tables[0].xpath('./li/text()')[0].split('：')[1]
            # 信息
            result[key] = value

            return result
        except:
            return {}

    def inspect_crawl(self):
        """
        检查项抓取模块
        :return:
        """
        import time

        start = time.clock()

        area_list = []
        ip_list = []
        index = 0
        for ip in dict.fromkeys(self.df[self.column], True):
            print(ip)
            index += 1
            if index % 1000 == 0:
                print([index, self.df.shape[0]])
                print('耗时： %s' % str(time.clock() - start))
                print('\n')

            try:
                url = 'http://www.ip138.com/ips1388.asp?ip=%s&action=2' % ip
                html = self.get_html(url)

                # ip
                ip_list.append(ip)

                # 地区
                area_list.append(self.food_spider(html)['本站数据'])

            except Exception as e:
                print(e)
                return []

        return ip_list, area_list

    def main(self, path, save=True):
        """
        主函数
        :param path:
        :param save:
        :return:
        """
        # 文件是否存在
        if os.path.exists(path):
            result = reduce_mem_usage(pd.read_hdf(path_or_buf=path, key=self.column, mode='r'))
        else:
            ip_list, area_list = self.inspect_crawl()

            result = pd.DataFrame(data={'id': self.df['sid'], self.column: ip_list, 'area': area_list})

            if save is True:
                result.to_hdf(path_or_buf=path, key=self.column, mode='r')

        return result


if __name__ == '__main__':
    # 获取训练集
    traindata = get_traindata()
    # 获取测试集
    testdata_feature = get_testdata_feature()

    # 训练集
    # print('train:\n')
    # CrimeSpider(traindata[['sid', 'ip']], 'ip').main(DefaultConfig.traindata_ip_path)
    # 测试集
    # print('test:\n')
    # CrimeSpider(testdata_feature[['sid', 'ip']], 'ip').main(DefaultConfig.testdata_feature_ip_path)
