from collections import Counter

from config.configure import getcfg
import utils.file_iterator as fi
import utils.tweet_keys as tk
import utils.array_utils as au
import utils.function_utils as fu
from clustering.cluster_service import ClusterService

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GSDPMM:
    @staticmethod
    def input_twarr_with_label(twarr, label):
        alpha_range = beta_range = [i / 100 for i in range(1, 10, 2)] + [i / 10 for i in range(1, 10, 2)]
        """cluster using different hyperparams in multiprocess way"""
        iter_num = 100
        process_num = 20
        hyperparams = [(a, b) for a in alpha_range for b in beta_range]
        params = [(None, twarr, *param, iter_num, label) for param in hyperparams]
        res_list = ClusterService.clustering_multi(GSDPMM.GSDPMM_twarr, params, process_num)
        param_num = len(hyperparams)
        """group the data by alpha"""
        frame = pd.DataFrame(index=np.arange(0, param_num), columns=['alpha', 'beta'])
        for i in range(param_num):
            frame.loc[i] = hyperparams[i]
        """start plotting figures"""
        for alpha, indices in frame.groupby('alpha').groups.items():
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(8)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            for i in indices:
                beta = frame.loc[i]['beta']
                topic_word_dstrb, tw_cluster_pred, iter_x, nmi_y, k_y = res_list[i]
                ax1.plot(iter_x, nmi_y, '-', lw=1.5, label='beta=' + str(round(beta, 2)))
                ax2.plot(iter_x, k_y, '^', lw=1.5, label='beta=' + str(round(beta, 2)))
            title = 'alpha=' + str(round(alpha, 2))
            ax1.set_title(title)
            ax1.set_ylabel('NMI')
            ax1.set_ylim(0.25, 0.75)
            ax1.legend(loc='lower left')
            ax1.text(iter_num * 0.6, 0.70,
                     'final nmi: ' + str(round(max([res_list[i][3][-1] for i in indices]), 4)), fontsize=15)
            ax2.set_xlabel('iteration')
            ax2.set_ylabel('K num')
            ax2.legend(loc='lower left')
            plt.grid(True, '-', color='#333333', lw=0.8)
            plt.savefig(getcfg().dc_test + 'GSDPMM/GSDPMM_alpha=' + title + '.png')
        
        # top_K = 20
        # alpha_idx = 0
        # beta_idx = 1
        # tw_cluster_pred_idx = 3
        # nmi_idx = 5
        # table_idx = 7
        # recall_idx = 8
        #
        # event_cluster_label = [i for i in range(12)]
        # summary_list = [hyperparams[i] + res_list[i] +
        #                 ClusterService.event_table_recall(label, res_list[i][1], event_cluster_label)
        #                 for i in range(param_num)]
        # top_recall_summary_list = [summary_list[i] for i in
        #                            np.argsort([summary[recall_idx] for summary in summary_list])[::-1][
        #                            :top_K]]
        # top_nmi_summary_list = [summary_list[i] for i in
        #                         np.argsort([summary[nmi_idx][-1] for summary in summary_list])[::-1][:top_K]]
        #
        # top_nmi_path = getcfg().dc_test + 'GSDPMM/max_nmis/'
        # top_recall_path = getcfg().dc_test + 'GSDPMM/max_recalls/'
        # fi.rmtree(top_nmi_path)
        # fi.rmtree(top_recall_path)
        #
        # def dump_cluster_info(summary_list, base_path):
        #     for rank, summary in enumerate(summary_list):
        #         res_dir = base_path + '{}_recall_{}_nmi_{}_alpha_{}_beta_{}/'. \
        #             format(rank, round(summary[recall_idx], 6), round(summary[nmi_idx][-1], 6),
        #                    summary[alpha_idx], summary[beta_idx])
        #         fi.makedirs(res_dir)
        #         tw_topic_arr = ClusterService.create_clusters_with_labels(twarr, summary[tw_cluster_pred_idx])
        #         for i, _twarr in enumerate(tw_topic_arr):
        #             if not len(_twarr) == 0:
        #                 fu.dump_array(res_dir + str(i) + '.txt', [tw[tk.key_text] for tw in _twarr])
        #         table = summary[table_idx]
        #         table.to_csv(res_dir + 'table.csv')
        #
        # dump_cluster_info(top_recall_summary_list, top_recall_path)
        # dump_cluster_info(top_nmi_summary_list, top_nmi_path)
        return None, None
    
    @staticmethod
    def GSDPMM_twarr(twarr, alpha, beta, iter_num):
        ner_pos_token = tk.key_wordlabels
        twarr = twarr[:]
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                if not ClusterService.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            for wordlabel in wordlabels:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
        min_df = 4
        for w in list(words.keys()):
            if words[w]['freq'] < min_df:
                del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[0] in words]))
        """definitions of parameters"""
        K = 1  # default 1 set by the algorithm
        D = len(twarr)
        V = len(words.keys())
        beta0 = V * beta
        z = [0] * D
        m_z = [0] * K
        n_z = [0] * K
        n_zw = [[0] * V for _ in range(K)]
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            for word, freq in twarr[d]['dup'].items():
                n_z[cluster] += freq
                n_zw[cluster][words[word]['id']] += freq
        """make sampling using current counting information"""
        
        def sample_cluster(tw, iter=None):
            prob = [0] * K
            freq_dict = tw['dup']
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
            if iter is not None and iter > iter_num - 5:
                return np.argmax(prob + [new_cluster_prob])
            else:
                return au.sample_index(np.array(prob + [new_cluster_prob]))
        
        """start iteration"""
        z_iter = list()
        for i in range(iter_num):
            z_iter.append(z[:])
            
            for d in range(D):
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
                    K -= 1
                    for d_ in range(D):
                        if z[d_] > min_tw_num_idx:
                            z[d_] -= 1
                
                cluster = sample_cluster(twarr[d], i)
                
                if cluster >= K:
                    m_z.append(0)
                    n_z.append(0)
                    n_zw.append([0] * V)
                    K += 1
                
                z[d] = cluster
                m_z[cluster] += 1
                for word, freq in freq_dict.items():
                    n_z[cluster] += freq
                    n_zw[cluster][words[word]['id']] += freq
        
        z_iter.append(z[:])
        return z_iter
