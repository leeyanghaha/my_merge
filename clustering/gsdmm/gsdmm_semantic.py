import numpy as np
import pandas as pd

from config.configure import getcfg
import utils.tweet_keys as tk
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
from utils.id_freq_dict import IdFreqDict
import clustering.cluster_service as cs


class SemanticClusterer:
    """ Use GSDMM as basic, no streaming """
    prop_n_tags = {'NNP', 'NNPS', }
    comm_n_tags = {'NN', }
    verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', }
    ht_rags = {'HT', }
    key_prop_n = 'prop'
    key_comm_n = 'comm'
    key_verb = 'verb'
    key_ht = 'ht'
    
    def __init__(self):
        self.twarr = None
        self.word_dicts = (self.prop_n_dict, self.comm_n_dict, self.verb_dict, self.ht_dict) = \
            (IdFreqDict(), IdFreqDict(), IdFreqDict(), IdFreqDict())
    
    def preprocess_twarr(self, twarr):
        """pre-process the tweet text, including dropping non-common terms"""
        key_tokens = tk.key_wordlabels
        self.twarr = twarr
        for word_dict in self.word_dicts:
            word_dict.clear()
        # self.prop_n_dict, self.comm_n_dict, self.verb_dict, self.ht_dict = \
        #     IdFreqDict(), IdFreqDict(), IdFreqDict(), IdFreqDict()
        pos_tag2dict_map = dict([(tag, self.prop_n_dict) for tag in self.prop_n_tags] +
                                [(tag, self.comm_n_dict) for tag in self.comm_n_tags] +
                                [(tag, self.verb_dict) for tag in self.verb_tags] +
                                [(tag, self.ht_dict) for tag in self.ht_rags])
        for tw in twarr:
            tokens = tw[key_tokens]
            for i in range(len(tokens) - 1, -1, -1):
                tokens[i][0] = tokens[i][0].lower().strip()
                word, _, pos_tag = tokens[i]
                if not cs.is_valid_keyword(word):
                    del tokens[i]
                if word.startswith('#') and not pos_tag.lower() == 'ht':
                    pos_tag = tokens[i][2] = 'HT'
                if pos_tag in pos_tag2dict_map:
                    pos_tag2dict_map[pos_tag].count_word(word)
        self.prop_n_dict.drop_words_by_condition(3)
        self.comm_n_dict.drop_words_by_condition(4)
        self.verb_dict.drop_words_by_condition(4)
        self.ht_dict.drop_words_by_condition(3)
        for tw in twarr:
            tw[self.key_prop_n], tw[self.key_comm_n], tw[self.key_verb], tw[self.key_ht] = \
                IdFreqDict(), IdFreqDict(), IdFreqDict(), IdFreqDict()
            tw_pos_tag2dict_map = dict([(tag, tw[self.key_prop_n]) for tag in self.prop_n_tags] +
                                       [(tag, tw[self.key_comm_n]) for tag in self.comm_n_tags] +
                                       [(tag, tw[self.key_verb]) for tag in self.verb_tags] +
                                       [(tag, tw[self.key_ht]) for tag in self.ht_rags])
            for token in tw[key_tokens]:
                word, _, pos_tag = token
                if pos_tag in tw_pos_tag2dict_map and pos_tag2dict_map[pos_tag].has_word(word):
                        tw_pos_tag2dict_map[pos_tag].count_word(word)
    
    def input_twarr_with_label(self, twarr, label):
        # def GSDMM_twarr(self, alpha, etap, etac, etav, etah, K, iter_num, ref_labels=None)
        # self.GSDMM_twarr(0.01, 0.01, 0.01, 0.01, 0.01, 5, 30)
        base_path = getcfg().dc_test + 'SEMANTIC/'
        a_range = etap_range = etac_range = etav_range = etah_range = [0.01, 0.05, 0.1]
        K_range = [30, 40]
        iter_num = 50
        """cluster using different hyperparams in multiprocess way"""
        process_num = 19
        hyperparams = [(a, ep, ec, ev, eh, k) for a in a_range for ep in etap_range for ec in etac_range
                       for ev in etav_range for eh in etah_range for k in K_range]
        param_num = len(hyperparams)
        res_list = cs.clustering_multi(SemanticClusterer.GSDMM_twarr,
                                       [(self, *param, iter_num, label) for param in hyperparams], process_num)
        column_name = ['alpha', 'etap', 'etac', 'etav', 'etah', 'K']
        # """start plotting figures"""
        # frame = pd.DataFrame(index=np.arange(0, param_num), columns=column_name, data=hyperparams)
        # for (alpha, K), indices in frame.groupby(['alpha', 'K']).groups.items():
        #     fig = plt.figure()
        #     fig.set_figheight(8)
        #     fig.set_figwidth(8)
        #     for i in indices:
        #         clu_word_distrb, tw_cluster_pred, iter_x, nmi_y = res_list[i]
        #         legend_params = ('etap', 'etac', 'etav', 'etah')
        #         plt_label = ','.join([p_name + str(frame.loc[i][p_name]) for p_name in legend_params])
        #         plt.plot(iter_x, nmi_y, '-', lw=1.5, label=plt_label)
        #     title = 'alpha=' + str(alpha) + ',K=' + str(K)
        #     plt.title(title)
        #     plt.ylabel('NMI')
        #     plt.ylim(0.25, 0.75)
        #     plt.legend(loc='lower left')
        #     plt.text(iter_num * 0.6, 0.70,
        #              'final nmi: ' + str(round(max([res_list[i][3][-1] for i in indices]), 4)), fontsize=15)
        #     plt.grid(True, '-', color='#333333', lw=0.8)
        #     plt.savefig(base_path + 'SEMANTIC' + title + '.png')
        """start dumping cluster information"""
        # def concat_param_name_values(param_names, param_values):
        #     if not len(param_names) == len(param_values):
        #         raise ValueError('inconsistent param number')
        #     return '_'.join(['{}_{:<3}'.format(param_names[i], param_values[i]) for i in range(len(param_names))])
        #
        # top_rank = 30
        # true_cluster = [i for i in range(12)]
        # tbl_recall_list = [ClusterService.event_table_recall(label, res_list[i][1], true_cluster) for i in range(param_num)]
        # top_recall_idx = pd.DataFrame(data=[(i, tbl_recall_list[i][1], res_list[i][3][-1]) for i in range(param_num)])\
        #     .sort_values(by=[1, 2], ascending=False).loc[:, 0][:top_rank]
        # top_nmi_idx = np.argsort([res_list[i][3][-1] for i in range(param_num)])[-1:-top_rank-1:-1]
        #
        # def dump_cluster_info(top_idx_list_, base_path_):
        #     for rank, idx in enumerate(top_idx_list_):
        #         res_dir = '{}{}_recall_{:0<6}_nmi_{:0<6}_{}/'.\
        #             format(base_path_, rank, round(tbl_recall_list[idx][1], 4), round(res_list[idx][3][-1], 4),
        #                    concat_param_name_values(column_name, hyperparams[idx]))
        #         fi.makedirs(res_dir)
        #         tw_topic_arr = ClusterService.create_clusters_with_labels(twarr, res_list[idx][1])
        #         for i, _twarr in enumerate(tw_topic_arr):
        #             if not len(_twarr) == 0:
        #                 fu.dump_array(res_dir + str(i) + '.txt', [tw[tk.key_text] for tw in _twarr])
        #         cluster_table = tbl_recall_list[idx][0]
        #         cluster_table.to_csv(res_dir + 'table.csv')
        #
        # top_recall_path = base_path + 'max_recalls/'
        # fi.rmtree(top_recall_path)
        # dump_cluster_info(top_recall_idx, top_recall_path)
        # top_nmi_path = base_path + 'max_nmis/'
        # fi.rmtree(top_nmi_path)
        # dump_cluster_info(top_nmi_idx, top_nmi_path)
        return 0, 0
    
    def GSDMM_twarr(self, alpha, etap, etac, etav, etah, K, iter_num):
        twarr = self.twarr
        prop_n_dict = self.prop_n_dict
        comm_n_dict = self.comm_n_dict
        verb_dict = self.verb_dict
        ht_dict = self.ht_dict
        D = len(twarr)
        VP = prop_n_dict.vocabulary_size()
        VC = comm_n_dict.vocabulary_size()
        VV = verb_dict.vocabulary_size()
        VH = ht_dict.vocabulary_size()
        alpha0 = alpha * K
        etap0 = VP * etap
        etac0 = VC * etac
        etav0 = VV * etav
        etah0 = VH * etah
        
        z = [0] * D
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
        def count_tw_into_tables(tw_freq_dict_, word_id_dict_, n_z_, n_zw_, clu_id, factor=1):
            for word, freq in tw_freq_dict_.word_freq_enumerate():
                if factor > 0:
                    n_z_[clu_id] += freq
                    n_zw_[clu_id][word_id_dict_.word2id(word)] += freq
                else:
                    n_z_[clu_id] -= freq
                    n_zw_[clu_id][word_id_dict_.word2id(word)] -= freq
        
        def update_clu_dicts_by_tw(tw, clu_id, factor=1):
            count_tw_into_tables(tw[self.key_prop_n], prop_n_dict, n_z_p, n_zw_p, clu_id, factor)
            count_tw_into_tables(tw[self.key_comm_n], comm_n_dict, n_z_c, n_zw_c, clu_id, factor)
            count_tw_into_tables(tw[self.key_verb], verb_dict, n_z_v, n_zw_v, clu_id, factor)
            count_tw_into_tables(tw[self.key_ht], ht_dict, n_z_h, n_zw_h, clu_id, factor)
        
        for d in range(D):
            k = int(K * np.random.random())
            z[d] = k
            m_z[k] += 1
            update_clu_dicts_by_tw(twarr[d], k, factor=1)
        """make sampling using current counting information"""
        def rule_value_of(tw_freq_dict_, word_id_dict_, n_z_, n_zw_, p, p0, clu_id):
            i_ = value = 1.0
            for word, freq in tw_freq_dict_.word_freq_enumerate():
                for ii in range(0, freq):
                    value *= (n_zw_[clu_id][word_id_dict_.word2id(word)] + ii + p) / (n_z_[clu_id] + i_ + p0)
                    i_ += 1
            return value
        
        def sample_cluster(tw, cur_iter=None):
            prob = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                prob[k] *= rule_value_of(tw[self.key_prop_n], prop_n_dict, n_z_p, n_zw_p, etap, etap0, k)
                prob[k] *= rule_value_of(tw[self.key_comm_n], comm_n_dict, n_z_c, n_zw_c, etac, etac0, k)
                prob[k] *= rule_value_of(tw[self.key_verb], verb_dict, n_z_v, n_zw_v, etav, etav0, k)
                prob[k] *= rule_value_of(tw[self.key_ht], ht_dict, n_z_h, n_zw_h, etah, etah0, k)
            if cur_iter is not None and cur_iter > iter_num - 5:
                return np.argmax(prob)
            else:
                return au.sample_index(np.array(prob))
        
        """start iteration"""
        z_iter = list()
        for i in range(iter_num):
            z_iter.append(z[:])
            
            for d in range(D):
                k = z[d]
                m_z[k] -= 1
                update_clu_dicts_by_tw(twarr[d], k, factor=-1)
                
                k = sample_cluster(twarr[d], i)
                
                z[d] = k
                m_z[k] += 1
                update_clu_dicts_by_tw(twarr[d], k, factor=1)
        
        z_iter.append(z[:])
        return z_iter
