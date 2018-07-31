import numpy as np

import utils.array_utils as au
import utils.tweet_utils as tu
from clustering.gsdmm.gsdmm_semantic import SemanticClusterer


class GSDMMSemanticStream(SemanticClusterer):
    def __init__(self, hold_batch_num=10):
        SemanticClusterer.__init__(self)
        self.init_batch_ready = False
        self.twarr = list()
        self.label = list()
        self.z = list()
        self.hold_batch_num = hold_batch_num
        self.batch_twnum_list = list()
    
    def set_hyperparams(self, alpha, etap, etac, etav, etah, K):
        self.hyperparams = (self.alpha, self.etap, self.etac, self.etav, self.etah, self.K) = \
            (alpha, etap, etac, etav, etah, K)
    
    def input_batch_with_label(self, tw_batch, lb_batch):
        self.batch_twnum_list.append(len(tw_batch))
        self.label += lb_batch
        self.twarr += tw_batch
        if len(self.batch_twnum_list) < self.hold_batch_num:
            return None, None
        if not self.init_batch_ready:
            self.preprocess_twarr(self.twarr)
            # self.z = self.GSDMM_twarr(*self.hyperparams, iter_num=self.iternum)
            self.z = self.GSDMM_new_twarr(list(), list(), self.twarr, *self.hyperparams, iter_num=70)
            self.init_batch_ready = True
            return self.z[:], self.label[:]
        self.preprocess_twarr(self.twarr)
        old_twarr_len = len(self.twarr) - len(tw_batch)
        old_twarr = self.twarr[:old_twarr_len]
        new_z = self.GSDMM_new_twarr(old_twarr, self.z, tw_batch, *self.hyperparams, iter_num=7)
        self.z += new_z
        for d in range(self.batch_twnum_list.pop(0)):
            self.z.pop(0), self.twarr.pop(0), self.label.pop(0)
        return self.z[:], self.label[:]
    
    def GSDMM_new_twarr(self, old_twarr, old_z, new_twarr, alpha, etap, etac, etav, etah, K, iter_num):
        new_twarr = tu.twarr_nlp(new_twarr)
        prop_n_dict, comm_n_dict, verb_dict, ht_dict = \
            self.prop_n_dict, self.comm_n_dict, self.verb_dict, self.ht_dict
        D_old, D_new = len(old_twarr), len(new_twarr)
        D = D_old + D_new
        VP = prop_n_dict.vocabulary_size()
        VC = comm_n_dict.vocabulary_size()
        VV = verb_dict.vocabulary_size()
        VH = ht_dict.vocabulary_size()
        alpha0 = K * alpha
        etap0 = VP * etap
        etac0 = VC * etac
        etav0 = VV * etav
        etah0 = VH * etah
        
        new_z = [-1] * D_new
        m_z = [0] * K
        n_z_p = [0] * K
        n_z_c = [0] * K
        n_z_v = [0] * K
        n_z_h = [0] * K
        n_zw_p = [[0] * VP for _ in range(K)]
        n_zw_c = [[0] * VC for _ in range(K)]
        n_zw_v = [[0] * VV for _ in range(K)]
        n_zw_h = [[0] * VH for _ in range(K)]
        """initialize the counting arrays"""
        def update_clu_dicts_by_tw(tw, clu_id, factor=1):
            count_tw_into_tables(tw[self.key_prop_n], prop_n_dict, n_z_p, n_zw_p, clu_id, factor)
            count_tw_into_tables(tw[self.key_comm_n], comm_n_dict, n_z_c, n_zw_c, clu_id, factor)
            count_tw_into_tables(tw[self.key_verb], verb_dict, n_z_v, n_zw_v, clu_id, factor)
            count_tw_into_tables(tw[self.key_ht], ht_dict, n_z_h, n_zw_h, clu_id, factor)
        
        def count_tw_into_tables(tw_freq_dict_, ifd_, n_z_, n_zw_, clu_id, factor):
            for word, freq in tw_freq_dict_.word_freq_enumerate():
                if factor > 0:
                    n_z_[clu_id] += freq
                    n_zw_[clu_id][ifd_.word2id(word)] += freq
                else:
                    n_z_[clu_id] -= freq
                    n_zw_[clu_id][ifd_.word2id(word)] -= freq
        
        for d in range(D_old):
            k = old_z[d]
            m_z[k] += 1
            update_clu_dicts_by_tw(old_twarr[d], k, factor=1)
        for d in range(D_new):
            k = int(K * np.random.random())
            new_z[d] = k
            m_z[k] += 1
            update_clu_dicts_by_tw(new_twarr[d], k, factor=1)
        """make sampling using current counting information"""
        def rule_value_of(tw_freq_dict_, word_id_dict_, n_z_, n_zw_, p, p0, clu_id):
            i_ = value = 1.0
            for word, freq in tw_freq_dict_.word_freq_enumerate():
                for ii in range(0, freq):
                    value *= (n_zw_[clu_id][word_id_dict_.word2id(word)] + ii + p) / (n_z_[clu_id] + i_ + p0)
                    i_ += 1
            return value
        
        def sample_cluster(tw, cur_iter):
            prob = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                prob[k] *= rule_value_of(tw[self.key_prop_n], prop_n_dict, n_z_p, n_zw_p, etap, etap0, k)
                prob[k] *= rule_value_of(tw[self.key_comm_n], comm_n_dict, n_z_c, n_zw_c, etac, etac0, k)
                prob[k] *= rule_value_of(tw[self.key_verb], verb_dict, n_z_v, n_zw_v, etav, etav0, k)
                prob[k] *= rule_value_of(tw[self.key_ht], ht_dict, n_z_h, n_zw_h, etah, etah0, k)
            if cur_iter >= iter_num - 1:
                return np.argmax(prob)
            else:
                return au.sample_index(np.array(prob))
        """start iteration"""
        for i in range(iter_num):
            for d in range(D_new):
                k = new_z[d]
                m_z[k] -= 1
                update_clu_dicts_by_tw(new_twarr[d], k, factor=-1)
                k = sample_cluster(new_twarr[d], i)
                new_z[d] = k
                m_z[k] += 1
                update_clu_dicts_by_tw(new_twarr[d], k, factor=1)
        return new_z
    
    def get_hyperparams_info(self):
        return 'GSDMM,semantic,stream, alpha={},etap={},etac={},etav={},etah={},K={}'.\
            format(self.alpha, self.etap, self.etac, self.etav, self.etah, self.K)
    
    # def clusters_similarity(self):
    #     return ClusterService.cluster_inner_similarity(self.twarr, self.z)
