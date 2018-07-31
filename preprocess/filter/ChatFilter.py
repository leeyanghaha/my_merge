import timeit
from collections import Counter

import numpy as np
from nltk.corpus import stopwords

import utils.pattern_utils as pu
import utils.tweet_keys as tk


class ChatFilter:
    def __init__(self):
        self.alpha = self.beta = self.iter_num = None
        self.twarr = self.words = None
        self.small_double = 1e-150
        self.large_double = 1e150
        self.stopwords = set(stopwords.words('english'))

    def set_hyperparams(self, alpha, beta, iter_num):
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num

    def set_twarr(self, twarr):
        self.twarr = twarr[:]
        self.preprocess()

    def clean_text(self, text):
        tokens = text.split()
        for i in range(len(tokens) - 1, -1, -1):
            tokens[i] = tokens[i].lower().strip()
            if not pu.is_valid_keyword(tokens[i]):
                del tokens[i]
        return tokens

    def preprocess(self):
        twarr = self.twarr
        words = self.words = dict()
        key_tokens = 'tokens'
        for tw in twarr:
            tokens = self.clean_text(tw[tk.key_text])
            for word in tokens:
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1}
            tw[key_tokens] = tokens
        min_df = 5
        for w in list(words.keys()):
            if words[w]['freq'] < min_df:
                del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([word for word in tw[key_tokens] if word in words]))

    def sample_cluster(self, tw, cur_iter=None):
        words = self.words
        alpha = self.alpha
        beta = self.beta
        iter_num = self.iter_num
        K = self.K
        D = self.D
        V = self.V
        beta0 = V * beta
        m_z = self.m_z
        n_z = self.n_z
        n_zw = self.n_zw
        prob = [0] * K

        if 'dup' in tw:
            freq_dict = tw['dup']
        else:
            freq_dict = dict(Counter([word for word in self.clean_text(tw[tk.key_text]) if word in words]))

        # if freq_dict.__len__() == 0:
        #    # print(tw['text'])
        #    pass

        for k in range(K):
            prob[k] = m_z[k] / (D - 1 + alpha)
            rule_value = 1.0
            i_ = 0
            for word, freq in freq_dict.items():
                for j_ in range(freq):
                    rule_value *= (n_zw[k][words[word]['id']] + beta + j_) / (n_z[k] + beta0 + i_)
                    i_ += 1
            prob[k] *= rule_value
        new_cluster_prob = alpha / (D - 1 + alpha)
        i_ = 0
        for word, freq in freq_dict.items():
            for j_ in range(freq):
                new_cluster_prob *= (beta + j_) / (beta0 + i_)
                i_ += 1
        array = prob + [new_cluster_prob]
        if cur_iter is not None and cur_iter > iter_num - 5:
            return np.argmax(array)
        else:
            try:
                return np.random.choice(a=[i for i in range(len(array))], p=np.array(array) / np.sum(array))
            except:
                if 'dup' in tw:
                    print(tw['dup'])

    def recluster_using_GSDMM(self):
        start = timeit.default_timer()
        twarr = self.twarr
        words = self.words
        """definitions of parameters"""
        K = self.K = 1
        D = self.D = len(twarr)
        V = self.V = len(words.keys())
        iter_num = self.iter_num
        z = self.z = [0] * D
        m_z = self.m_z = [0] * K
        n_z = self.n_z = [0] * K
        n_zw = self.n_zw = [[0] * V for _ in range(K)]
        """initialize the counting arrays"""
        for d in range(D):
            if twarr[d]['dup'].__len__() == 0:
                z[d] = -1
                continue
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            for word, freq in twarr[d]['dup'].items():
                n_z[cluster] += freq
                n_zw[cluster][words[word]['id']] += freq
        """start iteration"""
        for i in range(iter_num):
            # yue
            print("iter", i, "cluster num:", len(m_z), m_z)
            for d in range(D):
                if z[d] == -1:
                    continue
                freq_dict = twarr[d]['dup']
                cluster = z[d]
                m_z[cluster] -= 1
                for word, freq in freq_dict.items():
                    n_zw[cluster][words[word]['id']] -= freq
                    n_z[cluster] -= freq

                min_tw_num_idx = int(np.argmin(m_z))
                if m_z[min_tw_num_idx] == 0:
                    m_z.pop(min_tw_num_idx)
                    n_z.pop(min_tw_num_idx)
                    n_zw.pop(min_tw_num_idx)
                    self.K -= 1
                    for d_ in range(D):
                        if z[d_] > min_tw_num_idx:
                            z[d_] -= 1

                cluster = self.sample_cluster(twarr[d], i)

                if cluster >= self.K:
                    m_z.append(0)
                    n_z.append(0)
                    n_zw.append([0] * V)
                    self.K += 1

                z[d] = cluster
                m_z[cluster] += 1
                for word, freq in freq_dict.items():
                    n_z[cluster] += freq
                    n_zw[cluster][words[word]['id']] += freq
        stop = timeit.default_timer()
        print('trainning time', stop - start)
        return n_zw, z
