from collections import Counter

import numpy as np

import utils.array_utils as au
import utils.pattern_utils as pu
import utils.spacy_utils as su
import utils.tweet_keys as tk
from utils.id_freq_dict import IdFreqDict
from config.dict_loader import token_dict


class GSDMMStreamIFDDynamic:
    """ Dynamic cluster number, stream, dynamic dictionary, with ifd """
    ACT_STORE = 'store'
    ACT_FULL = 'full'
    ACT_SAMPLE = 'sample'
    
    default_full_iter = 30
    default_sample_iter = 4
    
    @staticmethod
    def pre_process_twarr(tw_batch):
        return [TweetHolder(tw) for tw in tw_batch]
    
    def __init__(self):
        print(self.__class__.__name__)
        self.alpha = self.alpha0 = self.beta = self.beta0 = self.K = None
        self.cludict = None
        self.twh_batches = list()
    
    def set_hyperparams(self, alpha, beta, k=100):
        alpha = 0.1
        beta = 0.1
        k = 150
        self.alpha, self.beta, self.K = alpha, beta, k
        self.alpha0 = alpha * k
        self.cludict = dict([(i, ClusterHolder(i)) for i in range(k)])
    
    """ sampling """
    def input_batch(self, tw_batch, action, iter_num=None):
        ACT_STORE, ACT_FULL, ACT_SAMPLE = self.ACT_STORE, self.ACT_FULL, self.ACT_SAMPLE
        assert action in {ACT_STORE, ACT_FULL, ACT_SAMPLE}
        twh_batch = self.pre_process_twarr(tw_batch)
        if action == ACT_STORE:
            self.twh_batches.append(twh_batch)
        elif action == ACT_FULL:
            self.twh_batches.append(twh_batch)
            self.full_cluster(iter_num)
        elif action == ACT_SAMPLE:
            self.sample_newest_pop_oldest(twh_batch, iter_num)
        else:
            raise ValueError('wrong action: {}'.format(action))
    
    def full_cluster(self, iter_num):
        print('full_cluster')
        iter_num = self.default_full_iter if not iter_num else iter_num
        self.GSDPMM_twarr(list(), self.get_current_twharr(), iter_num)
    
    def sample_newest_pop_oldest(self, twh_batch, iter_num):
        iter_num = self.default_sample_iter if not iter_num else iter_num
        self.GSDPMM_twarr(self.get_current_twharr(), twh_batch, iter_num)
        self.twh_batches.append(twh_batch)
        oldest_twarr = self.twh_batches.pop(0)
        for twh in oldest_twarr:
            twh.update_cluster(None)
    
    """ information """
    def get_current_twharr(self):
        return au.merge_array(self.twh_batches)
    
    def get_current_twarr(self):
        return [twh.tw for twh in self.get_current_twharr()]
    
    def get_twid_cluid_list(self):
        return [(tw[tk.key_id], tw[tk.key_event_cluid]) for tw in self.get_current_twharr()]
    
    def get_cluid_twarr_list(self, twnum_thres):
        cluid_twarr_list = list()
        for cluid in sorted(self.cludict.keys()):
            clutwarr = self.cludict[cluid].get_twarr()
            if len(clutwarr) >= twnum_thres:
                cluid_twarr_list.append((cluid, clutwarr))
        return cluid_twarr_list
    
    def filter_dup_id(self, twarr):
        res_idset, res_twarr = set(), list()
        self_idset = set([tw[tk.key_id] for tw in self.get_current_twharr()])
        for tw in twarr:
            twid = tw[tk.key_id]
            if twid in self_idset or twid in res_idset:
                continue
            res_idset.add(twid)
            res_twarr.append(tw)
        return res_twarr
    
    def sample(self, twh, D, using_max=False):
        alpha = self.alpha
        beta = self.beta
        alpha0 = self.alpha0
        beta0 = self.beta0
        cludict = self.cludict
        cluids = list()
        probs = list()
        tw_word_freq = twh.valid_tokens.word_freq_enumerate(newest=False)
        # tw_word_freq = twh.tokens.word_freq_enumerate(newest=False)
        for cluid, cluster in cludict.items():
            clu_tokens = cluster.tokens
            n_zk = clu_tokens.get_freq_sum() + beta0
            clu_prob_prefix = (cluster.twnum + alpha) / (D - 1 + alpha0)
            prob_delta = 1.0
            ii = 0
            for word, freq in tw_word_freq:
                n_zwk = clu_tokens.freq_of_word(word) if clu_tokens.has_word(word) else 0
                if freq == 1:
                    prob_delta *= (n_zwk + beta) / (n_zk + ii)
                    ii += 1
                elif freq > 1:
                    for jj in range(freq):
                        prob_delta *= (n_zwk + beta + jj) / (n_zk + ii)
                        ii += 1
            cluids.append(cluid)
            probs.append(clu_prob_prefix * prob_delta)
        
        if using_max:
            return cluids[np.argmax(probs)]
        else:
            return int(np.random.choice(a=cluids, p=np.array(probs) / np.sum(probs)))
    
    def GSDPMM_twarr(self, old_twharr, new_twharr, iter_num):
        cludict = self.cludict
        cluidarr = list(cludict.keys())
        valid_dict = IdFreqDict()
        if len(old_twharr) > 0:
            for cluster in cludict.values():
                cluster.clear()
        D = len(old_twharr) + len(new_twharr)
        """ recalculate the valid dictionary """
        for twh in old_twharr + new_twharr:
            valid_dict.merge_freq_from(twh.tokens, newest=False)
        valid_dict.drop_words_by_condition(3)
        """ reallocate & parameter """
        for old_twh in old_twharr:
            if old_twh.get_cluid() not in cludict:
                continue
            old_twh.validate(valid_dict)
            old_cluster = old_twh.cluster
            old_twh.cluster = None
            old_twh.update_cluster(old_cluster)
        for new_twh in new_twharr:
            new_twh.validate(valid_dict)
            if len(old_twharr) > 0:
                new_cluid = self.sample(new_twh, D, using_max=True)
            else:
                new_cluid = au.choice(cluidarr)
            new_twh.update_cluster(cludict[new_cluid])
        self.beta0 = self.beta * valid_dict.vocabulary_size()
        """ start iteration """
        for i in range(iter_num):
            print('  {} th clustering'.format(i))
            for twh in new_twharr:
                cluster = twh.cluster
                twh.update_cluster(None)
                if cluster.twnum == 0:
                    cludict.pop(cluster.cluid)
                cluid = self.sample(twh, D, using_max=(i == iter_num - 1))
                twh.update_cluster(cludict[cluid])
        for twh in new_twharr:
            twh.update_cluid_into_tw()
    
    """ extra functions """
    def get_hyperparams_info(self):
        return '{}, alpha={:<5}, beta={:<5}'.format(self.__class__.__name__, self.alpha, self.beta)


class ClusterHolder:
    def __init__(self, cluid):
        self.cluid = cluid
        self.twhdict = dict()
        self.tokens = IdFreqDict()
        self.twnum = 0
    
    """ basic functions """
    def get_twharr(self): return list(self.twhdict.values())
    
    def get_twarr(self): return [twh.tw for twh in self.twhdict.values()]
    
    def get_lbarr(self): return [twh[tk.key_event_label] for twh in self.twhdict.values()]
    
    def clear(self):
        self.twhdict.clear()
        self.tokens.clear()
        self.twnum = 0
    
    def update_by_twh(self, twh, factor):
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
        return self.cluster.cluid
    
    def update_cluid_into_tw(self):
        self.tw[tk.key_event_cluid] = self.cluster.cluid if self.cluster is not None else None
    
    def tokenize(self):
        # tokens = (t.text.lower() for t in self.tw[tk.key_spacy])
        tokens = pu.valid_tokenize(self.tw[tk.key_text].lower())
        for token in tokens:
            self.tokens.count_word(token)
    
    def validate(self, using_ifd):
        self.valid_tokens.clear()
        for word, freq in self.tokens.word_freq_enumerate(newest=False):
            if using_ifd.has_word(word):
                self.valid_tokens.count_word(word, freq)
    
    def update_cluster(self, cluster):
        if self.cluster is not None:
            self.cluster.update_by_twh(self, factor=-1)
        self.cluster = cluster
        if cluster is not None:
            cluster.update_by_twh(self, factor=1)
