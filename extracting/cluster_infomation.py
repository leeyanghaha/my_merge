from extracting.geo_and_time.extract_time_info import get_sutime
from extracting.geo_and_time.extract_geo_location import extract_geo_freq_list
from extracting.geo_and_time.extract_time_info import get_text_time, get_earlist_latest_post_time
from extracting.hot_and_level.event_hot import HotDegree, AttackHot
from extracting.hot_and_level.event_level import AttackLevel
from extracting.keyword_info.n_gram_keyword import get_quality_n_gram

import classifying.classifier_class_info as cci
import utils.tweet_keys as tk
import utils.spacy_utils as su
import utils.array_utils as au
import utils.function_utils as fu
import utils.file_iterator as fi

from collections import OrderedDict as Od
import numpy as np
import sys
import os


classes_table = {
    cci.event_t: [AttackHot, AttackLevel, cci.ClassifierTerror],
    cci.event_nd: [AttackHot, AttackLevel, cci.ClassifierNaturalDisaster],
    cci.event_k: [AttackHot, AttackLevel, cci.ClassifierK],
}


class ClusterInfoGetter:
    TAR_FULL = 1
    
    def __init__(self, event_type):
        using_class = classes_table[event_type]
        hot_class, lvl_class, clf_class = using_class
        self.hot = hot_class()
        self.lvl = lvl_class()
        self.clf = clf_class()
        self.sutime_obj = get_sutime()
        # with open(os.devnull, "w") as devnull:
        #     old_stdout, old_stderr = sys.stdout, sys.stderr
        #     sys.stdout, sys.stderr = devnull, devnull
        #     sys.stdout, sys.stderr = old_stdout, old_stderr
    
    @staticmethod
    def get_readable_geo_list(geo_freq_list):
        """
        根据一组描述地点以及频率的tuple，提取各个地点的可读知识信息
        :param geo_freq_list: list，每个元素为tuple，长度为0说明没有提取到有价值的地点实体；
            tuple的第一个元素是GoogleResult类型的对象，第二个元素是其在某一聚类中出现的频率
        :return: list，每个元素是一个tuple，其中含义依次为：
            行政级别，地点名称（城市级/街道级），地点所在国家全称，地点出现频次
        """
        readable_info = list()
        for geo, freq in geo_freq_list:
            if geo.quality == 'locality':
                info = (geo.quality, geo.city, geo.country_long, geo.bbox, freq)
            else:
                info = (geo.quality, geo.address, geo.country_long, geo.bbox, freq)
            readable_info.append(info)
        return readable_info
    
    @staticmethod
    def importance_sort(twarr, docarr, tokensarr, keywords):
        """
        对输入的推特列表，依据包括文本长度/分词个数、用户的影响力、含地点实体的个数、
        含本组推特列表的关键词的个数，进行降序排序
        :param twarr: list，推特列表
        :param docarr: list，每个元素为 spaCy.tokens.Doc，详见spaCy文档
        :param tokensarr: list，每个元素为list，与twarr中的推特一一对应，即每条推特的分词表
        :param keywords: list，每个元素为str，见 extracting.keyword_info.n_gram_keyword.get_quality_n_gram
        :return: list，排序后的推特列表
        """
        assert len(twarr) == len(docarr) == len(tokensarr)
        exception_substitute = np.zeros((len(twarr),))
        with np.errstate(divide='raise'):
            try:
                word_num = np.array([len(tokens) for tokens in tokensarr])
                text_len = np.array([len(tw[tk.key_text]) for tw in twarr])
                len_score = (np.log(word_num) + np.log(text_len)) * 0.2
            except:
                print('len_score error')
                len_score = exception_substitute
            try:
                hd = HotDegree()
                influ_score = np.array([hd.tw_propagation(tw) + hd.user_influence(tw[tk.key_user]) for tw in twarr])
                influ_score = np.log(influ_score + 1) * 0.4
            except:
                print('influ_score error')
                influ_score = exception_substitute
        
        ents_list = [[e for e in doc.ents if len(e.text) > 1] for doc in docarr]
        geo_num_list = [sum([(e.label_ in su.LABEL_LOCATION) for e in ents]) for ents in ents_list]
        gpe_score = np.array(geo_num_list) * 0.5
        
        keywords_set = set(keywords)
        key_common_num = [len(set(tokens).intersection(keywords_set)) for tokens in tokensarr]
        keyword_score = np.array(key_common_num)
        
        score_arr = gpe_score + len_score + influ_score + keyword_score
        sorted_idx = np.argsort(score_arr)[::-1]
        # # del
        # for idx in sorted_idx[:10]:
        #     text = twarr[idx][tk.key_text]
        #     g_score = round(gpe_score[idx], 4)
        #     l_score = round(len_score[idx], 4)
        #     i_score = round(influence_score[idx], 4)
        #     sum_score = round(g_score + l_score + i_score, 4)
        #     print('{}, {}, {}, {}\n{}\n'.format(g_score, l_score, i_score, sum_score, text))
        # # del
        return [twarr[idx] for idx in sorted_idx]
    
    @staticmethod
    def group_similar_tweets(twarr, process_num=0):
        """
        按照文本相似程度，调整推特列表中各推特的顺序，使得相近的文本被安排到相近的位置上；
        对一组文本进行两两比较操作复杂度O(n^2)，文本超过1000条就已十分耗时，分进程并行操作
        :param twarr: list，推特列表
        :param process_num: 使用的子进程的数量
        :return: list，排序后的推特列表
        """
        txtarr = [tw[tk.key_text] for tw in twarr]
        idx_g, txt_g = au.group_similar_items(txtarr, score_thres=0.3, process_num=process_num)
        tw_groups = [[twarr[idx] for idx in g] for g in idx_g]
        return au.merge_array(tw_groups)
    
    @staticmethod
    def clear_twarr_fields(twarr):
        """
        清理推特列表中各推特不合规的字段
        :param twarr: list，推特列表
        :return: list，整理后的推特列表
        """
        do_not_copy = {tk.key_spacy, tk.key_event_cluid, tk.key_event_label}
        post_twarr = list()
        for tw in twarr:
            new_tw = dict([(k, v) for k, v in tw.items() if k not in do_not_copy])
            post_twarr.append(new_tw)
        return post_twarr
    
    def cluid_twarr2cic(self, cluid, twarr, target=None):
        """
        对输入的（聚类编号、聚类推特列表），从聚类推特列表提取聚类信息、包装为一个结构化的对象
        :param cluid: int，聚类编号
        :param twarr: list，聚类推特列表
        :param target: int/None，若为 ClusterInfoGetter.TAR_FULL，则完整提取所有信息；
            否则仅提取地点信息（仅为了进行基于地点的聚类合并时加速）
        :return: ClusterInfoCarrier 对象
        """
        txtarr = [tw[tk.key_text] for tw in twarr]
        docarr = su.textarr_nlp(txtarr)
        prob = hot = level = time_list = keywords = None
        """ geo info """
        geo_freq_list = extract_geo_freq_list(twarr, docarr)
        geo_list = self.get_readable_geo_list(geo_freq_list)
        if target == ClusterInfoGetter.TAR_FULL:
            # print('full info')
            """ summary info """
            prob = self.clf.predict_mean_proba(twarr)
            hot = self.hot.hot_degree(twarr)
            level = self.lvl.get_level(twarr)
            """ time info """
            top_geo, top_freq = geo_freq_list[0] if geo_freq_list else (None, None)
            text_times, most_time = get_text_time(twarr, top_geo, self.sutime_obj)
            earliest_time, latest_time = get_earlist_latest_post_time(twarr)
            time_list = list(map(lambda t: t.isoformat(), [most_time, earliest_time, latest_time]))
            # most_time_str = most_time.isoformat()
            # earliest_time_str = earliest_time.isoformat()
            # latest_time_str = latest_time.isoformat()
            # time_list = [most_time_str, earliest_time_str, latest_time_str]
            """ keyword info """
            tokensarr, gram_keywords = get_quality_n_gram(txtarr, n_range=range(1, 5), len_thres=4)
            # auto_keywords = get_quality_autophrase(pidx, txtarr, conf_thres=0.5, len_thres=2)
            keywords = gram_keywords[:100]
            idxes, keywords = au.group_and_reduce(keywords, dist_thres=0.3, process_num=0)
            max_keyword_num = int(50 * np.tanh((len(twarr) + 20) / 40))
            keywords = keywords[:max_keyword_num]
            """ sort twarr """
            twarr = self.importance_sort(twarr, docarr, tokensarr, keywords)
            twarr = self.clear_twarr_fields(twarr)
        return ClusterInfoCarrier(cluid, twarr, prob, hot, level, geo_list, time_list, keywords)


"""
每个聚类使用一个json字符串表示，对应的对象含如下字段

= 'summary': dict, 概括聚类基本信息，含如下字段
    - 'cluster_id': int, 区分聚类，一个id只分配给一个聚类，即使聚类消失，id也不会给其他聚类
    - 'prob': float, 聚类置信度，标志聚类的预警程度
    - 'level': str, 表明事件等级，目前定义了4个级别(一般事件、较大事件、重大事件、特别重大事件)
    - 'hot': int, 表明事件热度，取值范围(0, 100)
    - 'keywords': list, 每个元素类型为str，代表一个关键词(组)，元素按重要程度从高到低排序

= 'inferred_geo': list, 聚类的地点提取结果，元素按确定程度从高到低排序
    每个元素类型为dict，含如下字段
        - 'quality': str, 表明该地点行政级别，
        - 'address': str, 表明该地点所处位置，行政级别为'city'时该字段
        - 'country': str, 表明该地点所属的国家全称
        - 'bbox': dict, 表明该地点的坐标范围(矩形)，含如下字段
            + 'northeast': list, 该地点的东北坐标，[纬度，经度]
            + 'southwest': list, 该地点的西南坐标，[纬度，经度]
        # - 'weight': float, 表明该地点在聚类中被识别的相对权重

= 'inferred_time': dict, 聚类的时间提取结果，含如下字段
    - 'most_possible_time': str, ISO-8601格式，表明从聚类推断的最可能指向的时间(段)
    - 'earliest_time': str, ISO-8601格式，表明聚类中发布最早的推文的时间
    - 'latest_time': str, ISO-8601格式，表明聚类中发布最晚的推文的时间

= 'tweet_list': list, 聚类的推文列表，元素按重要程度从高到低排序
    每个元素类型为str，即推特的json格式字符串
"""


class ClusterInfoCarrier:
    large_geo = {'country', 'continent'}
    
    def __init__(self, cluid, twarr, prob, hot, level, geo_list, time_list, keywords):
        self.cluid = cluid
        self.twarr = twarr
        self.prob = prob
        self.hot = hot
        self.level = level
        self.geo_list = geo_list
        self.time_list = time_list
        self.keywords = keywords
        
        self.geo_table = [Od(zip(['quality', 'address', 'country', 'bbox', 'freq'], g)) for g in self.geo_list]
        self.s_geo_table = [row for row in self.geo_table if row['quality'] not in self.large_geo]
        self.time_table = Od(zip(['most_possible_time', 'earliest_time', 'latest_time'], self.time_list)) \
            if self.time_list is not None else None
    
    def construct_od(self):
        """
        从当前聚类信息构造一个字段按顺序保存的字典，该字典维护一个用于描述聚类信息的树形结构，
        各字段含义与数值类型见本文件 line 182-209 的说明
        :return: OrderedDict，保存了聚类的结构化信息
        """
        summary = Od(zip(
            ['cluster_id', 'prob', 'level', 'hot', 'keywords'],
            [self.cluid, self.prob, self.level, self.hot, self.keywords],
        ))
        od = Od(zip(
            ['summary', 'geo_infer', 'time_infer', 'tweet_list'],
            [summary, self.geo_table, self.time_table, self.twarr],
        ))
        return od


def merge_cic_list2cluid_twarr_list(cic_list):
    """
    分析输入的 cic_list 中每个对象所指示的频次最高的地点信息，按照这一信息进行聚类合并
    :param cic_list: list，每个元素为 ClusterInfoCarrier
    :return: list，每个元素为tuple，
        见 clustering.gsdpmm.gsdpmm_stream_ifd_dynamic.GSDPMMStreamIFDDynamic#get_cluid_twarr_list
    """
    geo2group_id, cluid2group = dict(), dict()
    for cic in cic_list:
        cluid, clu_geo_table = cic.cluid, cic.geo_table
        s_geo_table = cic.s_geo_table
        if len(s_geo_table) == 0:
            cluid2group[cluid] = [cic]
        else:
            clu_top_geo = s_geo_table[0]['address']
            if clu_top_geo not in geo2group_id:
                group_id = cluid
                geo2group_id[clu_top_geo] = group_id
                cluid2group[group_id] = [cic]
            else:
                group_id = geo2group_id[clu_top_geo]
                cluid2group[group_id].append(cic)
    new_cluid_twarr_list = list()
    for group_id, group_cic_list in cluid2group.items():
        new_cluid = group_cic_list[0].cluid
        new_twarr = au.merge_array([cic.twarr for cic in group_cic_list])
        new_cluid_twarr_list.append((new_cluid, new_twarr))
    return new_cluid_twarr_list


def write_cic_list(path, cic_list):
    """
    调用 cic_list 中的各元素的 construct_od返回一个OrderedDIct，将每个OrderedDIct持久化到指定路径下的文件中
    :param path: str，输出路径
    :param cic_list: list，每个元素为 ClusterInfoCarrier
    :return:
    """
    fi.mkdir(path, remove_previous=True)
    cic_list = sorted(cic_list, key=lambda item: len(item.twarr), reverse=True)
    print('    bext: output cic list, len={}'.format(len(cic_list)))
    for idx, cic in enumerate(cic_list):
        cluid = cic.cluid
        cic.twarr = ClusterInfoGetter.group_similar_tweets(cic.twarr, process_num=10)
        od = cic.construct_od()
        json_str = fu.dumps(od)
        cluster_file = fi.join(path, '{}_cluid:{}.json'.format(idx, cluid))
        fu.write_lines(cluster_file, [json_str])
        
        # textarr_file = fi.join(path, '{}_text.json'.format(idx))
        # textarr = [tw[tk.key_text] for tw in cic.twarr]
        # fu.write_lines(textarr_file, textarr)
    print('    bext: output into files over')

def cic_format(cic_list):
    cic_list = sorted(cic_list, key=lambda item: len(item.twarr), reverse=True)
    res = []
    print('    bext: output cic list, len={}'.format(len(cic_list)))
    for idx, cic in enumerate(cic_list):
        cic.twarr = ClusterInfoGetter.group_similar_tweets(cic.twarr, process_num=10)
        od = cic.construct_od()
        res.append(od)
    return res
