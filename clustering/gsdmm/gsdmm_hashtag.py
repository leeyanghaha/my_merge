import math
from collections import Counter

import utils.multiprocess_utils
import utils.tweet_keys as tk
import utils.array_utils as au
import utils.function_utils as fu
import utils.file_iterator as fi
import clustering.cluster_service as cs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GSDMMHashtag:
    @staticmethod
    def input_twarr_with_label(twarr, label):
        alpha_range = beta_range = gamma_range = [i/100 for i in range(1, 10, 3)] + [i/10 for i in range(1, 10, 3)]
        K_range = [20, 30, 40, 50]
        """cluster using different hyperparams in multiprocess way"""
        iter_num = 100
        process_num = 20
        hyperparams = [(a, b, g, k) for a in alpha_range for b in beta_range
                       for g in gamma_range for k in K_range]
        param_num = len(hyperparams)
        res_list = list()
        for i in range(int(math.ceil(param_num / process_num))):
            param_list = [(None, twarr, *param, iter_num, label) for param in
                          hyperparams[i * process_num: (i + 1) * process_num]]
            res_list += utils.multiprocess_utils.multi_process(GSDMMHashtag.GSDMM_twarr_hashtag, param_list)
            print('{:<3} /'.format(min((i + 1) * process_num, param_num)), param_num, 'params processed')
        frame = pd.DataFrame(index=np.arange(0, param_num), columns=['alpha', 'beta', 'gamma', 'K'])
        for i in range(param_num):
            frame.loc[i] = hyperparams[i]
        """start plotting figures"""
        for (alpha, K), indices in frame.groupby(['alpha', 'K']).groups.items():
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(8)
            for i in indices:
                beta = frame.loc[i]['beta']
                gamma = frame.loc[i]['gamma']
                key_distrb, ht_distrb, tw_cluster_pred, iter_x, nmi_y = res_list[i]
                plt.plot(iter_x, nmi_y, '-', lw=1.5, label='beta=' + str(beta) + ',gamma=' + str(gamma))
            title = 'alpha=' + str(alpha) + ',K=' + str(K)
            plt.title(title)
            plt.ylabel('NMI')
            plt.ylim(0.25, 0.75)
            plt.legend(loc='lower left')
            plt.text(iter_num * 0.6, 0.70,
                     'final nmi: ' + str(round(max([res_list[i][4][-1] for i in indices]), 4)), fontsize=15)
            plt.legend(loc='lower left')
            plt.grid(True, '-', color='#333333', lw=0.8)
            plt.savefig('./GSDMM_ht/GSDMM_ht_' + title + '.png')
        
        # top_ramk = 20
        # alpha_idx = 0
        # beta_idx = 1
        # gamma_idx = 2
        # K_idx = 3
        # tw_cluster_pred_idx = 6
        # nmi_idx = 8
        # table_idx = 9
        # recall_idx = 10
        #
        # event_cluster_label = [i for i in range(12)]
        # summary_list = [hyperparams[i] + res_list[i] +
        #                 ClusterService.event_table_recall(label, res_list[i][2], event_cluster_label)
        #                 for i in range(param_num)]
        # top_recall_summary_list = [summary_list[i] for i in
        #                            np.argsort([summary[recall_idx] for summary in summary_list])[::-1][
        #                            :top_ramk]]
        # top_nmi_summary_list = [summary_list[i] for i in
        #                         np.argsort([summary[nmi_idx][-1] for summary in summary_list])[::-1][
        #                         :top_ramk]]
        #
        # def dump_cluster_info(summary_list_, base_path):
        #     for rank, summary in enumerate(summary_list_):
        #         res_dir = base_path + '{}_recall_{:0<6}_nmi_{:0<6}_alpha_{:0<6}_beta_{:0<6}_gamma_{:0<6}_K_{}/'. \
        #             format(rank, round(summary[recall_idx], 6), round(summary[nmi_idx][-1], 6),
        #                    summary[alpha_idx], summary[beta_idx], summary[gamma_idx], summary[K_idx])
        #         fi.makedirs(res_dir)
        #         tw_topic_arr = ClusterService.create_clusters_with_labels(twarr, summary[tw_cluster_pred_idx])
        #         for i, _twarr in enumerate(tw_topic_arr):
        #             if not len(_twarr) == 0:
        #                 fu.dump_array(res_dir + str(i) + '.txt', [tw[tk.key_text] for tw in _twarr])
        #         table = summary[table_idx]
        #         table.to_csv(res_dir + 'table.csv')
        #
        # top_recall_path = '/home/nfs/cdong/tw/testdata/cdong/GSDMM_ht/max_recalls/'
        # fi.rmtree(top_recall_path)
        # dump_cluster_info(top_recall_summary_list, top_recall_path)
        # top_nmi_path = '/home/nfs/cdong/tw/testdata/cdong/GSDMM_ht/max_nmis/'
        # fi.rmtree(top_nmi_path)
        # dump_cluster_info(top_nmi_summary_list, top_nmi_path)
        return None, None
    
    @staticmethod
    def GSDMM_twarr_hashtag(twarr, alpha, beta, gamma, K, iter_num):
        ner_pos_token = tk.key_wordlabels
        twarr = twarr[:]
        key_dict = dict()
        ht_dict = dict()
        
        def word_count_id(word_dict, w):
            if w in word_dict:
                word_dict[w]['freq'] += 1
            else:
                word_dict[w] = {'freq': 1, 'id': word_dict.__len__()}
        
        def rearrange_id(word_dict):
            for idx, w in enumerate(sorted(word_dict.keys())):
                word_dict[w]['id'] = idx
        
        def drop_words_freq_less_than(word_dict, min_freq):
            for w in list(word_dict.keys()):
                if word_dict[w]['freq'] < min_freq:
                    del word_dict[w]
            rearrange_id(word_dict)
        
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                key = wordlabels[i][0] = wordlabels[i][0].lower().strip()  # hashtags are reserved here
                if not cs.is_valid_keyword(key):
                    del wordlabels[i]
                else:
                    if key.startswith('#'):
                        word_count_id(ht_dict, key)
                    else:
                        word_count_id(key_dict, key)
        drop_words_freq_less_than(ht_dict, 3)
        drop_words_freq_less_than(key_dict, 5)
        for tw in twarr:
            tw['key'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[0] in key_dict]))
            tw['ht'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[0] in ht_dict]))
        # pos_tw_num = len([1 for label in ref_labels if label <= 11])
        # neg_tw_num = len(twarr) - pos_tw_num
        # print('hashtag in pos:', len([1 for tw in twarr[:pos_tw_num] if tw['ht'].__len__() > 0]) / pos_tw_num)
        # print('hashtag in pos = 1:', len([1 for tw in twarr[:pos_tw_num] if tw['ht'].__len__() == 1]) / pos_tw_num)
        # print('hashtag in pos = 2:', len([1 for tw in twarr[:pos_tw_num] if tw['ht'].__len__() == 2]) / pos_tw_num)
        # print('hashtag in pos >= 3:', len([1 for tw in twarr[:pos_tw_num] if tw['ht'].__len__() >= 3]) / pos_tw_num)
        # print('hashtag in neg:', len([1 for tw in twarr[pos_tw_num:] if tw['ht'].__len__() > 0]) / neg_tw_num)
        # print('hashtag in neg = 1:', len([1 for tw in twarr[pos_tw_num:] if tw['ht'].__len__() == 1]) / neg_tw_num)
        # print('hashtag in neg = 2:', len([1 for tw in twarr[pos_tw_num:] if tw['ht'].__len__() == 2]) / neg_tw_num)
        # print('hashtag in neg >= 3:', len([1 for tw in twarr[pos_tw_num:] if tw['ht'].__len__() >= 3]) / neg_tw_num)
        # print('tw num:', len(twarr), 'pos_tw_num', pos_tw_num, 'neg_tw_num', neg_tw_num)
        # print('----')
        """definitions of parameters"""
        D = len(twarr)
        V = len(key_dict)
        H = len(ht_dict)
        alpha0 = K * alpha
        beta0 = V * beta  # hyperparam for keyword_info
        gamma0 = H * gamma  # hyperparam for hashtag
        
        z = [0] * D
        m_z = [0] * K
        n_z_key = [0] * K
        n_z_ht = [0] * K
        n_zw_key = [[0] * V for _ in range(K)]
        n_zw_ht = [[0] * H for _ in range(K)]
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            key_freq_dict = twarr[d]['key']
            ht_freq_dict = twarr[d]['ht']
            for key, freq in key_freq_dict.items():
                n_z_key[cluster] += freq
                n_zw_key[cluster][key_dict[key]['id']] += freq
            for ht, freq in ht_freq_dict.items():
                n_z_ht[cluster] += freq
                n_zw_ht[cluster][ht_dict[ht]['id']] += freq
        """make sampling using current counting information"""
        
        def rule_value_of(tw_freq_dict_, word_id_dict_, n_z_, n_zw_, p, p0, cluster):
            i_ = value = 1
            for w_, w_freq in tw_freq_dict_.items():
                for i in range(0, w_freq):
                    value *= (n_zw_[cluster][word_id_dict_[w_]['id']] + i + p) / (n_z_[cluster] + i_ + p0)
                    i_ += 1
            return value
        
        def sample_cluster(tw, iter=None):
            prob = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                key_freq_dict = tw['key']
                ht_freq_dict = tw['ht']
                prob[k] *= rule_value_of(key_freq_dict, key_dict, n_z_key, n_zw_key, beta, beta0, k)
                prob[k] *= rule_value_of(ht_freq_dict, ht_dict, n_z_ht, n_zw_ht, gamma, gamma0, k)
            if iter is not None and iter > iter_num - 5:
                return np.argmax(prob)
            else:
                return au.sample_index(np.array(prob))
        
        """start iteration"""
        
        def update_using_freq_dict(tw_freq_dict_, word_id_dict_, n_z_, n_zw_, factor):
            for w, w_freq in tw_freq_dict_.items():
                w_freq *= factor
                n_z_[cluster] += w_freq
                n_zw_[cluster][word_id_dict_[w]['id']] += w_freq
        
        """ start iteration """
        z_iter = list()
        for i in range(iter_num):
            z_iter.append(z[:])

            for d in range(D):
                cluster = z[d]
                m_z[cluster] -= 1
                key_freq_dict = twarr[d]['key']
                ht_freq_dict = twarr[d]['ht']
                
                update_using_freq_dict(key_freq_dict, key_dict, n_z_key, n_zw_key, -1)
                update_using_freq_dict(ht_freq_dict, ht_dict, n_z_ht, n_zw_ht, -1)
                
                cluster = sample_cluster(twarr[d], i)
                
                z[d] = cluster
                m_z[cluster] += 1
                update_using_freq_dict(key_freq_dict, key_dict, n_z_key, n_zw_key, 1)
                update_using_freq_dict(ht_freq_dict, ht_dict, n_z_ht, n_zw_ht, 1)
        
        z_iter.append(z[:])
        return z_iter
