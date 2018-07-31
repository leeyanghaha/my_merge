from collections import Counter

import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
import utils.tweet_utils as tu
import clustering.cluster_service as cs


class GSDPMMStream:
    """ Dynamic cluster number, stream, dynamic dictionary, without ifd """
    
    def __init__(self, hold_batch_num):
        print(self.__class__.__name__)
        self.alpha, self.beta = None, None
        self.init_batch_ready = False
        self.hold_batch_num = hold_batch_num
        self.twarr, self.label, self.history_length, self.z = list(), list(), list(), list()
        self.max_cluid = 0
    
    def set_hyperparams(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
    
    def input_batch_with_label(self, tw_batch, lb_batch=None):
        tu.twarr_nlp(tw_batch)
        self.history_length.append(len(tw_batch))
        if len(self.history_length) < self.hold_batch_num:
            # insufficient tweets
            self.twarr.extend(tw_batch)
            self.label.extend(lb_batch) if lb_batch is not None else None
            return (None, None) if lb_batch is not None else None
        elif not self.init_batch_ready:
            # the first time when len(self.batch_twnum_list) == self.hold_batch_num, may get merged
            self.init_batch_ready = True
            self.twarr += tw_batch
            self.label.extend(lb_batch) if lb_batch is not None else None
            self.z = self.GSDPMM_twarr(list(), list(), self.twarr, iter_num=30)
            z_ = [int(i) for i in self.z]
            return (z_, self.label[:]) if lb_batch is not None else z_
        else:
            # normal process of new twarr
            new_z = self.GSDPMM_twarr(self.twarr, self.z, tw_batch, iter_num=5)
            self.twarr += tw_batch
            self.label.extend(lb_batch) if lb_batch is not None else None
            self.z += new_z
            
            oldest_len = self.history_length.pop(0)
            self.z = self.z[oldest_len:]
            self.twarr = self.twarr[oldest_len:]
            if lb_batch is not None:
                self.label = self.label[oldest_len:]
            z_ = [int(i) for i in self.z]
            return (z_, self.label[:]) if lb_batch is not None else z_
    
    """ first GSDPMM_twarr version """
    # def GSDPMM_twarr(self, old_twarr, old_z, new_twarr, iter_num):
    #     pos_token = tk.key_ark
    #     twarr = old_twarr + new_twarr
    #     words = dict()
    #     """pre-process the tweet text, including dropping non-common terms"""
    #     for tw in twarr:
    #         tokens = tw[pos_token]
    #         for i in range(len(tokens) - 1, -1, -1):
    #             tokens[i][0] = tokens[i][0].lower().strip('#').strip()
    #             if not ClusterService.is_valid_keyword(tokens[i][0]): del tokens[i]
    #         for wordlabel in tokens:
    #             word = wordlabel[0]
    #             if word in words: words[word]['freq'] += 1
    #             else: words[word] = {'freq': 1, 'id': len(words.keys())}
    #     min_df = 3
    #     for w in list(words.keys()):
    #         if words[w]['freq'] < min_df: del words[w]
    #     for idx, w in enumerate(sorted(words.keys())):
    #         words[w]['id'] = idx
    #     for tw in twarr:
    #         tw['dup'] = dict(Counter([wlb[0] for wlb in tw[pos_token] if wlb[0] in words]))
    #     """definitions of parameters"""
    #     D = len(twarr)
    #     V = len(words.keys())
    #     alpha, beta = self.alpha, self.beta
    #     beta0 = V * beta
    #     new_z = [0] * len(new_twarr)
    #     K = 1 if not old_z else max(old_z) + 1
    #     m_z = [0] * K
    #     n_z = [0] * K
    #     n_zw = [[0] * V for _ in range(K)]
    #     """initialize the counting arrays"""
    #     for old_d in range(len(old_twarr)):
    #         cluster = old_z[old_d]
    #         m_z[cluster] += 1
    #         for word, freq in old_twarr[old_d]['dup'].items():
    #             n_z[cluster] += freq
    #             n_zw[cluster][words[word]['id']] += freq
    #     for new_d in range(len(new_twarr)):
    #         cluster = int(K * np.random.random())
    #         new_z[new_d] = cluster
    #         m_z[cluster] += 1
    #         for word, freq in new_twarr[new_d]['dup'].items():
    #             n_z[cluster] += freq
    #             n_zw[cluster][words[word]['id']] += freq
    #     """make sampling using current counting information"""
    #     def sample_cluster(tw, cur_iter=None):
    #         prob = [0] * K
    #         tw_freq_dict = tw['dup']
    #         for k in range(K):
    #             prob[k] = m_z[k] / (D - 1 + alpha)
    #             i_ = 0
    #             for word, freq in tw_freq_dict.items():
    #                 for j_ in range(freq):
    #                     prob[k] *= (n_zw[k][words[word]['id']] + beta + j_) / (n_z[k] + beta0 + i_)
    #                     i_ += 1
    #         new_cluster_prob = alpha / (D - 1 + alpha)
    #         i_ = 0
    #         for word, freq in tw_freq_dict.items():
    #             for j_ in range(freq):
    #                 new_cluster_prob *= (beta + j_) / (beta0 + i_)
    #                 i_ += 1
    #         if cur_iter is not None and cur_iter >= iter_num - 1:
    #             return np.argmax(prob + [new_cluster_prob])
    #         else:
    #             return au.sample_index_by_array_value(np.array(prob + [new_cluster_prob]))
    #
    #     """start iteration"""
    #     for i in range(iter_num):
    #         for new_d in range(len(new_twarr)):
    #             freq_dict = new_twarr[new_d]['dup']
    #             cluster = new_z[new_d]
    #             m_z[cluster] -= 1
    #             for word, freq in freq_dict.items():
    #                 n_zw[cluster][words[word]['id']] -= freq
    #                 n_z[cluster] -= freq
    #
    #             min_tw_num_idx = int(np.argmin(m_z))
    #             if m_z[min_tw_num_idx] == 0:
    #                 m_z.pop(min_tw_num_idx)
    #                 n_z.pop(min_tw_num_idx)
    #                 n_zw.pop(min_tw_num_idx)
    #                 K -= 1
    #                 for new_d_ in range(len(new_twarr)):
    #                     if new_z[new_d_] > min_tw_num_idx:
    #                         new_z[new_d_] -= 1
    #
    #             cluster = sample_cluster(new_twarr[new_d], i)
    #
    #             if cluster >= K:
    #                 m_z.append(0)
    #                 n_z.append(0)
    #                 n_zw.append([0] * V)
    #                 K += 1
    #
    #             new_z[new_d] = cluster
    #             m_z[cluster] += 1
    #             for word, freq in freq_dict.items():
    #                 n_z[cluster] += freq
    #                 n_zw[cluster][words[word]['id']] += freq
    #
    #     return new_z
    
    """ modified GSDPMM_twarr """
    
    def GSDPMM_twarr(self, old_twarr, old_z, new_twarr, iter_num):
        pos_token = tk.key_ark
        twarr = old_twarr + new_twarr
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            tokens = tw[pos_token]
            for i in range(len(tokens) - 1, -1, -1):
                tokens[i][0] = tokens[i][0].lower().strip('#').strip()
                if not cs.is_valid_keyword(tokens[i][0]): del tokens[i]
            for wordlabel in tokens:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
        min_df = 3
        for w in list(words.keys()):
            if words[w]['freq'] < min_df: del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[pos_token] if wlb[0] in words]))
        """definitions of parameters"""
        D = len(twarr)
        V = len(words.keys())
        alpha, beta = self.alpha, self.beta
        beta0 = V * beta
        new_z = [0] * len(new_twarr)
        K = {self.max_cluid} if not old_z else set(old_z)
        m_z = dict([(k, 0) for k in K])
        n_z = dict([(k, 0) for k in K])
        n_zw = dict([(k, [0] * V) for k in K])
        """initialize the counting arrays"""
        for old_d in range(len(old_twarr)):
            old_cluid = old_z[old_d]
            m_z[old_cluid] += 1
            for word, freq in old_twarr[old_d]['dup'].items():
                n_z[old_cluid] += freq
                n_zw[old_cluid][words[word]['id']] += freq
        for new_d in range(len(new_twarr)):
            new_cluid = np.random.choice(list(K))
            new_z[new_d] = new_cluid
            m_z[new_cluid] += 1
            for word, freq in new_twarr[new_d]['dup'].items():
                n_z[new_cluid] += freq
                n_zw[new_cluid][words[word]['id']] += freq
        """make sampling using current counting information"""
        
        def sample_cluster(_tw, cur_iter=None):
            prob = {}
            tw_freq_dict = _tw['dup']
            for k in K:
                prob[k] = m_z[k] / (D - 1 + alpha)
                _i = 0
                for _word, _freq in tw_freq_dict.items():
                    for _j in range(_freq):
                        prob[k] *= (n_zw[k][words[_word]['id']] + beta + _j) / (n_z[k] + beta0 + _i)
                        _i += 1
            new_clu_prob = alpha / (D - 1 + alpha)
            _i = 0
            for _word, _freq in tw_freq_dict.items():
                for _j in range(_freq):
                    new_clu_prob *= (beta + _j) / (beta0 + _i)
                    _i += 1
            
            prob[self.max_cluid + 1] = new_clu_prob
            cluid_arr = sorted(prob.keys())
            prob_arr = [prob[__cluid] for __cluid in cluid_arr]
            if cur_iter is not None and cur_iter >= iter_num - 1:
                return cluid_arr[np.argmax(prob_arr)]
            else:
                return cluid_arr[au.sample_index(np.array(prob_arr))]
        
        """start iteration"""
        for i in range(iter_num):
            for new_d in range(len(new_twarr)):
                freq_dict = new_twarr[new_d]['dup']
                old_cluid = new_z[new_d]
                new_z[new_d] = -1
                m_z[old_cluid] -= 1
                for word, freq in freq_dict.items():
                    n_z[old_cluid] -= freq
                    n_zw[old_cluid][words[word]['id']] -= freq
                
                for _cluid in list(m_z.keys()):
                    if m_z[_cluid] == 0:
                        m_z.pop(_cluid), n_z.pop(_cluid), n_zw.pop(_cluid), K.remove(_cluid)
                
                new_cluid = sample_cluster(new_twarr[new_d], i)
                
                if new_cluid > self.max_cluid:
                    new_cluid = self.max_cluid = self.max_cluid + 1
                    m_z[self.max_cluid] = 0
                    n_z[self.max_cluid] = 0
                    n_zw[self.max_cluid] = [0] * V
                    K.add(self.max_cluid)
                
                new_z[new_d] = new_cluid
                m_z[new_cluid] += 1
                for word, freq in freq_dict.items():
                    n_z[new_cluid] += freq
                    n_zw[new_cluid][words[word]['id']] += freq
        
        return new_z
    
    """ with re-tweet chain """
    
    # def GSDPMM_twarr(self, old_twarr, old_z, new_twarr, iter_num):
    #     pos_token = tk.key_ark
    #     twarr = old_twarr + new_twarr
    #     words = dict()
    #     """pre-process the tweet text, including dropping non-common terms"""
    #     for tw in twarr:
    #         tokens = tw[pos_token]
    #         for i in range(len(tokens) - 1, -1, -1):
    #             tokens[i][0] = tokens[i][0].lower().strip('#').strip()
    #             if not ClusterService.is_valid_keyword(tokens[i][0]): del tokens[i]
    #         for wordlabel in tokens:
    #             word = wordlabel[0]
    #             if word in words:
    #                 words[word]['freq'] += 1
    #             else:
    #                 words[word] = {'freq': 1, 'id': len(words.keys())}
    #     min_df = 3
    #     for w in list(words.keys()):
    #         if words[w]['freq'] < min_df:
    #             del words[w]
    #     for idx, w in enumerate(sorted(words.keys())):
    #         words[w]['id'] = idx
    #     for tw in twarr:
    #         tw['dup'] = dict(Counter([wlb[0] for wlb in tw[pos_token] if wlb[0] in words]))
    #     """definitions of parameters"""
    #     D = len(twarr)
    #     V = len(words.keys())
    #     alpha, beta = self.alpha, self.beta
    #     beta0 = V * beta
    #     # print('beta {} beta0 {}'.format(beta, beta0))
    #     new_z = [0] * len(new_twarr)
    #     K = {self.max_cluid} if not old_z else set(old_z)
    #     m_z = dict([(k, 0) for k in K])
    #     n_z = dict([(k, 0) for k in K])
    #     n_zw = dict([(k, [0] * V) for k in K])
    #
    #     old_retw_dict = dict([(k, set()) for k in K])
    #     new_retw_dict = dict([(k, set()) for k in K])
    #     """initialize the counting arrays"""
    #     for old_d, old_tw in enumerate(old_twarr):
    #         old_cluid = old_z[old_d]
    #         m_z[old_cluid] += 1
    #         for word, freq in old_tw.get('dup').items():
    #             n_z[old_cluid] += freq
    #             n_zw[old_cluid][words[word]['id']] += freq
    #
    #     for new_d, new_tw in enumerate(new_twarr):
    #         new_cluid = None
    #         # retwid = tu.in_reply_to(new_tw)
    #         # if retwid is not None:
    #         #     print('in reply to {}'.format(retwid))
    #         if new_cluid is None:
    #             new_cluid = np.random.choice(list(K))
    #
    #         new_z[new_d] = new_cluid
    #         m_z[new_cluid] += 1
    #         for word, freq in new_tw.get('dup').items():
    #             n_z[new_cluid] += freq
    #             n_zw[new_cluid][words[word]['id']] += freq
    #     """make sampling using current counting information"""
    #     def sample_cluster(_tw, cur_iter=None):
    #         prob = {}
    #         tw_freq_dict = _tw['dup']
    #         for k in K:
    #             p = m_z[k] / (D - 1 + alpha)
    #             _i = 0
    #             prob_delta = 1.0
    #             for _word, _freq in tw_freq_dict.items():
    #                 for _j in range(_freq):
    #                     # print(k, n_zw[k][words[_word]['id']], beta, n_z[k], beta0)
    #                     prob_delta *= (n_zw[k][words[_word]['id']] + beta + _j) / (n_z[k] + beta0 + _i)
    #                     _i += 1
    #             prob[k] = p * prob_delta
    #             print('old:{} init old:{} delta:{}'.format(prob[k], p, prob_delta))
    #         _i = 0
    #         new_clu_prob = alpha / (D - 1 + alpha)
    #         prob_delta = 1.0
    #         for _word, _freq in tw_freq_dict.items():
    #             for _j in range(_freq):
    #                 prob_delta *= (beta + _j) / (beta0 + _i)
    #                 _i += 1
    #         prob[self.max_cluid + 1] = new_clu_prob * prob_delta
    #         print('new:{} init new:{} delta:{}'.format(prob[self.max_cluid + 1], new_clu_prob, prob_delta))
    #         print('beta:{} beta0:{} V:{}'.format(beta, beta0, V))
    #         print()
    #
    #         cluid_arr = sorted(prob.keys())
    #         prob_arr = [prob[__cluid] for __cluid in cluid_arr]
    #         # print(prob_arr)
    #         if cur_iter is not None and cur_iter >= iter_num - 1:
    #             return cluid_arr[np.argmax(prob_arr)]
    #         else:
    #             return cluid_arr[au.sample_index(np.array(prob_arr))]
    #
    #     """start iteration"""
    #     for i in range(iter_num):
    #         for new_d in range(len(new_twarr)):
    #             freq_dict = new_twarr[new_d]['dup']
    #             old_cluid = new_z[new_d]
    #             new_z[new_d] = -1
    #             m_z[old_cluid] -= 1
    #             for word, freq in freq_dict.items():
    #                 n_z[old_cluid] -= freq
    #                 n_zw[old_cluid][words[word]['id']] -= freq
    #
    #             for _cluid in list(m_z.keys()):
    #                 if m_z[_cluid] == 0:
    #                     m_z.pop(_cluid), n_z.pop(_cluid), n_zw.pop(_cluid), K.remove(_cluid)
    #
    #             new_cluid = sample_cluster(new_twarr[new_d], i)
    #
    #             if new_cluid > self.max_cluid:
    #                 new_cluid = self.max_cluid = self.max_cluid + 1
    #                 m_z[self.max_cluid] = 0
    #                 n_z[self.max_cluid] = 0
    #                 n_zw[self.max_cluid] = [0] * V
    #                 K.add(self.max_cluid)
    #
    #             new_z[new_d] = new_cluid
    #             m_z[new_cluid] += 1
    #             for word, freq in freq_dict.items():
    #                 n_z[new_cluid] += freq
    #                 n_zw[new_cluid][words[word]['id']] += freq
    #
    #     return new_z
    
    def get_hyperparams_info(self):
        return 'GSDPMM,stream, alpha={:<5}, beta={:<5}'.format(self.alpha, self.beta)
    
    # def clusters_similarity(self):
    #     return ClusterService.cluster_inner_similarity(self.twarr[:], self.z[:])
