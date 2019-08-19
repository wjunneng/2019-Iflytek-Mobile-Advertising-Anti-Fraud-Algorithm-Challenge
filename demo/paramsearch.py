from sortedcontainers import SortedList
import copy
import collections
import numpy as np
from itertools import product


class paramsearch:
    """
    a class for doing grid search on a set of parameters provided in a dict. 'pdict' should be a dictionary like the following:
    pdict = {'depth':[1,2], 'iterations':[250,100,500], 'thread_count':4}

    when grid_search is called it will return an iterator that provides samples from the dictionary e.g.
    {'depth':1, 'iterations':250, 'thread_count':4}
    {'depth':2, 'iterations':250, 'thread_count':4}
    {'depth':1, 'iterations':100, 'thread_count':4}
    etc.
    after calling an iteration of grid_search, you need to test the classifier and run 'register_result'
    This will update the internal list of results, so that the next call to grid_search will use the best
    parameters for all the parameters not currently being updated.

    grid_search can be provided a list e.g. grid_search(['depth']) this will use the current best parameters for all
    the other arguments and only search over 'depth'. You can then call e.g. grid_search(['iterations']) and it will use
    the best depth found previously and cycle through all the 'iterations'. Searching incrementally can be much faster
    than doing a full grid search, but may miss the global optimum.
    """

    def __init__(self, pdict):
        self.pdict = {}
        # 如果某些内容未作为序列传入，则将其设为包含1个元素的序列, 不要将字符串视为序列
        for a, b in pdict.items():
            if isinstance(b, collections.Sequence) and not isinstance(b, str):
                self.pdict[a] = b
            else:
                self.pdict[a] = [b]
        # 我们的结果是一个排序列表，因此最佳分数始终是最终元素
        self.results = SortedList()

    def grid_search(self, keys=None):
        # 仅对列出的键进行网格搜索。 如果没有提供，请全部执行
        if keys == None:
            keylist = self.pdict.keys()
        else:
            keylist = keys

        # 这将是键，值对列表的列表
        listoflists = []
        for key in keylist: listoflists.append([(key, i) for i in self.pdict[key]])
        for p in product(*listoflists):
            # 对当前最佳参数集进行任何更改
            if len(self.results) > 0:
                template = self.results[-1][1]
            else:
                template = {a: b[0] for a, b in self.pdict.items()}
            # 如果我们的更新与当前最佳相同，请不要打扰
            if self.equaldict(dict(p), template):
                continue
            # 采取当前最好的，只更新要改变的
            yield self.overwritedict(dict(p), template)

    def equaldict(self, a, b):
        for key in a.keys():
            if a[key] != b[key]:
                return False
        return True

    def overwritedict(self, new, old):
        old = copy.deepcopy(old)
        for key in new.keys():
            old[key] = new[key]
        return old

    # 将（得分，参数）对保存到结果中。 由于'结果'是一个排序列表，最好的分数永远是最后的元素。 添加少量噪音
    # 因为当两个分数完全相同时，排序列表不喜欢它
    def register_result(self, result, params):
        self.results.add((result + np.random.randn() * 1e-10, params))

    def bestscore(self):
        return self.results[-1][0]

    def bestparam(self):
        return self.results[-1][1]
