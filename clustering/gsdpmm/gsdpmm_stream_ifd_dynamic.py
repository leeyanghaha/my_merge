from collections import Counter
import numpy as np

import utils.pattern_utils as pu
import utils.tweet_keys as tk
from utils.id_freq_dict import IdFreqDict


class GSDPMMStreamIFDDynamic:
    """ Dynamic cluster number, stream, dynamic dictionary, with ifd """
    ACT_FULL = 'full'
    ACT_STORE = 'store'
    ACT_SAMPLE = 'sample'
    FULL_ITER_NUM = 30
    SAMPLE_ITER_NUM = 3
    
    @staticmethod
    def pre_process_twarr(tw_batch):
        """
        预处理推特列表，将其转化为可以被本模块处理的对象
        :param tw_batch: list，推特列表
        :return: list，每个元素为 TweetHolder
        """
        return [TweetHolder(tw) for tw in tw_batch]
    
    def __init__(self):
        print(self.__class__.__name__)
        self.alpha = self.beta = self.beta0 = None
        self.cludict = self.max_cluid = None
        self.twharr = list()
    
    def set_hyperparams(self, alpha, beta):
        """
        设置聚类器使用两个概率模型的先验超参
        :param alpha: float，聚类分离程度；越大越容易产生更多聚类
        :param beta: float，聚类主题集中程度，越小则相同单词少的推特越不容易进入同一聚类
        :return:
        """
        self.alpha, self.beta = alpha, beta
    
    def input_batch(self, tw_batch, action, iter_num=None):
        """
        给定一组新的推特列表，指定以什么方式将这些推特加入当前聚类器；
        ACT_STORE：仅将新列表存储到 self.twharr 中待用；
        ACT_SAMPLE：根据当前 self.cludict 中已经产生的聚类，为每条新推特采样并分配聚类；
        :param tw_batch: list，推特列表
        :param action: str，仅用于表示执行动作
        :param iter_num: int，迭代次数
        :return:
        """
        if not tw_batch:
            return
        a_store, a_full, a_sample = self.ACT_STORE, self.ACT_FULL, self.ACT_SAMPLE
        twh_batch = self.pre_process_twarr(tw_batch)
        if action == a_store:
            self.twharr.extend(twh_batch)
        elif action == a_sample:
            self.sample_newest(twh_batch, iter_num)
        # elif action == a_full:
        #     self.twharr.extend(twh_batch)
        #     self.full_cluster(iter_num)
        else:
            print('wrong action: {}'.format(action))
        self.check_empty_cluster()
    
    def full_cluster(self, iter_num):
        """
        将当前聚类清空（若已有），并以 self.twharr 中所有的推特包装对象为 corpus 进行完整的聚类采样过程
        :param iter_num: int，总体聚类迭代次数
        :return:
        """
        for twh in self.twharr:
            twh.cluster = None
        self.max_cluid = 0
        self.cludict = {self.max_cluid: ClusterHolder(self.max_cluid)}
        iter_num = self.FULL_ITER_NUM if not iter_num else iter_num
        self.GSDPMM_twarr(list(), self.twharr, iter_num)
    
    def sample_newest(self, new_twharr, iter_num):
        """
        根据当前 self.cludict 中已经产生的聚类，为 new_twharr 中每条推特采样并分配聚类；过程中可能产生新聚类
        :param new_twharr: list，推特列表
        :param iter_num: int，每条推特采样聚类的迭代次数
        :return:
        """
        iter_num = self.SAMPLE_ITER_NUM if not iter_num else iter_num
        self.GSDPMM_twarr(self.twharr, new_twharr, iter_num)
        self.twharr.extend(new_twharr)
    
    def pop_oldest(self, oldest_k):
        """
        从 self.twharr 中弹出最早的 oldest_k 条推特包装对象，并更新这些推特所在聚类的统计值
        :param oldest_k: int，要弹出的推特的条数
        :return:
        """
        if len(self.twharr) == 0:
            return
        for _ in range(oldest_k):
            oldest_twh = self.twharr.pop(0)
            oldest_twh.update_cluster(None)
        self.check_empty_cluster()
    
    """ information """
    def get_current_tw_num(self):
        return len(self.twharr)
    
    def get_current_twarr(self):
        return [twh.tw for twh in self.twharr]
    
    def get_twid_cluid_list(self):
        return [(tw[tk.key_id], tw[tk.key_event_cluid]) for tw in self.get_current_twarr()]
    
    def get_cluid_twarr_list(self, twnum_thres):
        """
        根据当前已计算的聚类结果，返回描述聚类信息的简单列表
        :param twnum_thres: 过滤推特列表长度小于该阈值的聚类
        :return: list，每个元素是一个tuple；tuple的第一个元素是聚类的编号，第二个元素是聚类的推特列表
        """
        cludict = self.cludict
        if not cludict:
            return None
        cluid_twarr_list = list()
        for cluid in sorted(cludict.keys()):
            clutwarr = cludict[cluid].get_twarr()
            if len(clutwarr) >= twnum_thres:
                cluid_twarr_list.append((cluid, clutwarr))
        return cluid_twarr_list
    
    def filter_dup_id(self, twarr):
        res_idset, res_twarr = set(), list()
        self_idset = set([tw[tk.key_id] for tw in self.get_current_twarr()])
        for tw in twarr:
            twid = tw[tk.key_id]
            if twid in self_idset or twid in res_idset:
                continue
            res_idset.add(twid)
            res_twarr.append(tw)
        return res_twarr
    
    def check_empty_cluster(self):
        """
        检查 self.cludict 中有没有空的聚类，即持有推特数量为 0 的聚类
        :return:
        """
        if not self.cludict:
            return
        for cluid in list(self.cludict.keys()):
            cluster = self.cludict[cluid]
            if cluster.twnum == 0:
                assert len(cluster.twhdict) == 0
                self.cludict.pop(cluid)
    
    def sample(self, twh, D, using_max=False, no_new_clu=False, cluid_range=None):
        """
        采样函数，输入一个推特包装对象，为其采样一个 self.cludict 中已有的聚类对象，
        :param twh: TweetHolder，推特的包装对象
        :param D: 采样所使用的文本corpus大小
        :param using_max: 若是，则采样依据概率分布随机采样聚类；否则使用概率值最大的聚类
        :param no_new_clu: 若是，则不计算新聚类的概率；否则一起计算产生新聚类的概率大小
        :param cluid_range: 指定采样已有聚类时所用的聚类ID编号范围
        :return:
        """
        alpha = self.alpha
        beta = self.beta
        beta0 = self.beta0
        cludict = self.cludict
        cluids = list()
        probs = list()
        tw_word_freq = twh.valid_tokens.word_freq_enumerate(newest=False)
        # tw_word_freq = twh.tokens.word_freq_enumerate(newest=False)
        if cluid_range is None:
            cluid_range = cludict.keys()
        for cluid in cluid_range:
            cluster = cludict[cluid]
            clu_tokens = cluster.tokens
            n_zk = clu_tokens.get_freq_sum() + beta0
            old_clu_prob = cluster.twnum / (D - 1 + alpha)
            prob_delta = 1.0
            ii = 0
            for word, freq in tw_word_freq:
                n_zwk = clu_tokens.freq_of_word(word) if clu_tokens.has_word(word) else 0
                if freq == 1:
                    prob_delta *= (n_zwk + beta) / (n_zk + ii)
                    ii += 1
                else:
                    for jj in range(freq):
                        prob_delta *= (n_zwk + beta + jj) / (n_zk + ii)
                        ii += 1
            cluids.append(cluid)
            probs.append(old_clu_prob * prob_delta)
        if not no_new_clu:
            ii = 0
            new_clu_prob = alpha / (D - 1 + alpha)
            prob_delta = 1.0
            for word, freq in tw_word_freq:
                if freq == 1:
                    prob_delta *= beta / (beta0 + ii)
                    ii += 1
                else:
                    for jj in range(freq):
                        prob_delta *= (beta + jj) / (beta0 + ii)
                        ii += 1
            cluids.append(self.max_cluid + 1)
            probs.append(new_clu_prob * prob_delta)
        if using_max:
            return cluids[np.argmax(probs)]
        else:
            return int(np.random.choice(a=cluids, p=np.array(probs) / np.sum(probs)))
    
    def GSDPMM_twarr(self, old_twharr, new_twharr, iter_num):
        """
        实际执行聚类以及采样，若old_twharr为空，则认为new_twharr是corpus，
        并在其上进行完整的聚类过程，耗时较多；
        若old_twharr不为空，则认为old_twharr已经持有了之前聚类的结果信息，
        并对new_twharr中的每条推特包装对象采样已有的聚类对象
        :param old_twharr: list，元素类型为 TweetHolder
        :param new_twharr: list，元素类型为 TweetHolder
        :param iter_num: 聚类循环所用的迭代次数
        :return:
        """
        cludict = self.cludict
        """ recalculate the valid dictionary """
        valid_dict = IdFreqDict()
        D = len(old_twharr) + len(new_twharr)
        for twh in old_twharr + new_twharr:
            valid_dict.merge_freq_from(twh.tokens, newest=False)
        valid_dict.drop_words_by_condition(3)
        """ reallocate & parameter """
        for cluster in cludict.values():
            cluster.clear()
        for old_twh in old_twharr:
            # if old_twh.get_cluid() not in cludict:
            #     continue
            old_twh.validate(valid_dict)
            old_cluster = old_twh.cluster
            old_twh.cluster = None
            old_twh.update_cluster(old_cluster)
        for new_twh in new_twharr:
            new_twh.validate(valid_dict)
            if old_twharr:
                new_cluid = self.sample(new_twh, D, using_max=True, no_new_clu=True)
            else:
                new_cluid = self.max_cluid
            cluster = cludict[new_cluid]
            new_twh.update_cluster(cluster)
        self.beta0 = self.beta * valid_dict.vocabulary_size()
        """ start iteration """
        for i in range(iter_num):
            print('  {} th clustering, clu num: {}'.format(i + 1, len(cludict)))
            for twh in new_twharr:
                cluster = twh.cluster
                twh.update_cluster(None)
                if cluster.twnum == 0:
                    cludict.pop(cluster.cluid)
                cluid = self.sample(twh, D, using_max=(i == iter_num - 1))
                if cluid not in cludict:
                    self.max_cluid = cluid
                    cludict[cluid] = ClusterHolder(cluid)
                twh.update_cluster(cludict[cluid])
        for twh in new_twharr:
            twh.update_cluid_into_tw()
    
    """ extra functions """
    def get_hyperparams_info(self):
        return '{}, alpha={:<5}, beta={:<5}'.format(self.__class__.__name__, self.alpha, self.beta)
    
    # def cluster_mutual_similarity(self):
    #     cluid2vector = self.cluid2info_by_key('vector')
    #     cluid_arr = sorted(cluid2vector.keys())
    #     clu_num = len(cluid_arr)
    #     assert clu_num >= 2
    #     vecarr = [cluid2vector[cluid] for cluid in cluid_arr]
    #     sim_matrix = au.cosine_similarity(vecarr)
    #     pair_sim_arr = list()
    #     for i in range(0, clu_num - 1):
    #         for j in range(i + 1, clu_num):
    #             pair_sim_arr.append([cluid_arr[i], cluid_arr[j], float(sim_matrix[i][j])])
    #     return pair_sim_arr


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.twhdict = dict()
        self.tokens = IdFreqDict()
        self.twnum = 0
    
    """ basic functions """
    def get_twharr(self):
        """
        返回聚类当前持有的推特包装对象的列表，不考虑排列顺序
        :return:  list，每个元素类型为TweetHolder
        """
        return list(self.twhdict.values())
    
    def get_twarr(self):
        """
        返回聚类当前持有的推特对象的列表，不考虑排列顺序
        :return: list，推特列表
        """
        return [twh.tw for twh in self.twhdict.values()]
    
    def get_lbarr(self):
        """
        返回聚类当前的推特对象所持有的标记（若存在该信息）的列表，不考虑排列顺序
        :return: list，元素为int，表示推特原本的标记值（原本属于哪个聚类）
        """
        return [twh[tk.key_event_label] for twh in self.twhdict.values()]
    
    def clear(self):
        """
        清空当前聚类的统计信息，包括分词表、推特列表、推特计数
        :return:
        """
        self.twhdict.clear()
        self.tokens.clear()
        self.twnum = 0
    
    def update_by_twh(self, twh, factor):
        """
        将输入的推特包装对象加入/移出当前聚类，并根据其 valid_tokens 更新当前聚类的分词表等统计信息
        :param twh: TweetHolder，要加入的推特包装对象
        :param factor: int，1表示加入，0表示移出
        :return:
        """
        twh_tokens = twh.valid_tokens
        twh_id = twh.id
        if factor > 0:
            self.tokens.merge_freq_from(twh_tokens, newest=False)
            self.twhdict[twh_id] = twh
            self.twnum += 1
        else:
            self.tokens.drop_freq_from(twh_tokens, newest=False)
            if twh_id in self.twhdict:
                self.twhdict.pop(twh_id)
            self.twnum -= 1
    
    """ extra functions """
    def get_rep_label(self, rep_thres):
        """
        计算当前聚类中是否存在标记数占推特列表总数的比例大于阈值 rep_thres 的标记
        :param rep_thres: float，判定阈值
        :return: int，若存在足够占比的标记则返回该标记，否则返回-1
        """
        lb_count = Counter(self.get_lbarr())
        max_label, max_lbnum = lb_count.most_common(1)[0]
        rep_label = -1 if max_lbnum < self.twnum * rep_thres else max_label
        return rep_label
    
    # def extract_keywords(self):
    #     pos_keys = {su.pos_prop, su.pos_comm, su.pos_verb}
    #     self.ifds = self.keyifd, self.entifd = IdFreqDict(), IdFreqDict()
    #     for twh in self.twhdict.values():
    #         doc = twh.get(tk.key_spacy)
    #         for token in doc:
    #             word, pos = token.text.strip().lower(), token.pos_
    #             if pos in pos_keys and word not in pu.stop_words:
    #                 self.keyifd.count_word(word)
    #         for ent in doc.ents:
    #             if ent.label_ in su.LABEL_LOCATION:
    #                 self.entifd.count_word(ent.text.strip().lower())
    #
    # def similarity_vec_with(self, cluster):
    #     def ifd_sim_vec(ifd1, ifd2):
    #         v_size1, v_size2 = ifd1.vocabulary_size(), ifd2.vocabulary_size()
    #         freq_sum1, freq_sum2 = ifd1.get_freq_sum(), ifd2.get_freq_sum()
    #         common_words = set(ifd1.vocabulary()).intersection(set(ifd2.vocabulary()))
    #         comm_word_num = len(common_words)
    #         comm_freq_sum1 = sum([ifd1.freq_of_word(w) for w in common_words])
    #         comm_freq_sum2 = sum([ifd2.freq_of_word(w) for w in common_words])
    #         portion1 = freq_sum1 / comm_freq_sum1 if comm_freq_sum1 > 0 else 0
    #         portion2 = freq_sum2 / comm_freq_sum2 if comm_freq_sum2 > 0 else 0
    #         comm_portion1 = comm_word_num / v_size1 if v_size1 > 0 else 0
    #         comm_portion2 = comm_word_num / v_size2 if v_size2 > 0 else 0
    #         return [portion1, portion2, comm_portion1, comm_portion2]
    #     this_keyifd, this_entifd = self.ifds
    #     that_keyifd, that_entifd = cluster.ifds
    #     sim_vec_key = ifd_sim_vec(this_keyifd, that_keyifd)
    #     sim_vec_ent = ifd_sim_vec(this_entifd, that_entifd)
    #     sim_vec = np.concatenate([sim_vec_key, sim_vec_ent], axis=0)
    #     return sim_vec


class TweetHolder:
    # using_ifd = token_dict()
    
    def __init__(self, tw):
        self.tw = tw
        self.id = tw.get(tk.key_id)
        self.cluster = None
        self.tokens = IdFreqDict()
        self.valid_tokens = IdFreqDict()
        self.tokenize()
    
    def __contains__(self, key): return key in self.tw
    
    def __getitem__(self, key): return self.get(key)
    
    def __setitem__(self, key, value): self.setdefault(key, value)
    
    def get(self, key): return self.tw.get(key, None)
    
    def setdefault(self, key, value): self.tw.setdefault(key, value)
    
    def get_cluid(self):
        """
        返回当前已被分配的聚类的ID
        :return: int，聚类的ID编号
        """
        return self.cluster.cluid
    
    def update_cluid_into_tw(self):
        """
        更新推特对象（self.tw，dict类型）的 tk.key_event_cluid 字段为当前已被分配的聚类的ID，
        若尚未被分配聚类则置为None
        :return:
        """
        self.tw[tk.key_event_cluid] = self.cluster.cluid if self.cluster is not None else None
    
    def tokenize(self):
        """
        将推特对象的text进行分词并保存分词结果，使用 self.tokens 进行分词计数
        :return:
        """
        # tokens = (t.text.lower() for t in self.tw[tk.key_spacy])
        tokens = pu.valid_tokenize(self.tw[tk.key_text].lower())
        for token in tokens:
            self.tokens.count_word(token)
    
    def validate(self, using_ifd):
        """
        更新分词表，将 self.tokens 中存在于 using_ifd 的单词重新计数到 self.valid_tokens 中
        :param using_ifd: utils.id_freq_dict.IdFreqDict，包含当前迭代中的合法分词
        :return:
        """
        self.valid_tokens.clear()
        for word, freq in self.tokens.word_freq_enumerate(newest=False):
            if using_ifd.has_word(word):
                self.valid_tokens.count_word(word, freq)
    
    def update_cluster(self, cluster):
        """
        若原本有聚类，则将当前推特从原本的 self.cluster 中分离；
        并将当前推特合并至 cluster 中（若不为None），更新 self.cluster
        :param cluster: 目标聚类对象
        :return:
        """
        if self.cluster is not None:
            self.cluster.update_by_twh(self, factor=-1)
        self.cluster = cluster
        if cluster is not None:
            cluster.update_by_twh(self, factor=1)


if __name__ == '__main__':
    import utils.function_utils as fu
    import utils.timer_utils as tmu
    
    tmu.check_time()
    twarr = fu.load_array("/home/nfs/cdong/tw/src/calling/filtered_twarr.json")
    print('load over')
    g = GSDPMMStreamIFDDynamic()
    g.set_hyperparams(30, 0.01)
    g.input_batch(twarr, g.ACT_FULL, 25)
    tmu.check_time()
