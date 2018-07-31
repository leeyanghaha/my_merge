import math
from collections import Counter

import utils.multiprocess_utils
from config.configure import getcfg
import utils.tweet_keys as tk
import utils.array_utils as au
import utils.function_utils as fu
import clustering.cluster_service as cs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GSDMM:
    @staticmethod
    @fu.sync_real_time_counter('GSDMM input_twarr_with_label')
    def input_twarr_with_label(twarr, label):
        # alpha_range = beta_range = [i/100 for i in range(1, 10, 3)] + [i/10 for i in range(1, 10, 3)] + \
        #                            [i for i in range(1, 10, 3)]
        # K_range = [30, 40, 50]
        alpha_range = beta_range = [i/100 for i in range(1, 10, 4)] + [i/10 for i in range(1, 10, 4)]
        K_range = [30, 40, 50]
        """cluster using different hyperparams in multiprocess way"""
        iter_num = 100
        process_num = 20
        hyperparams = [(a, b, K) for a in alpha_range for b in beta_range for K in K_range]
        res_list = list()
        for i in range(int(math.ceil(len(hyperparams) / process_num))):
            param_list = [(twarr, *param, iter_num) for param in
                          hyperparams[i * process_num: (i + 1) * process_num]]
            res_list += utils.multiprocess_utils.multi_process(GSDMM.GSDMM_twarr, param_list)
            print('{:<4} /'.format((i + 1) * process_num), len(hyperparams), 'params processed')
        """group the data by K"""
        frame = pd.DataFrame(index=np.arange(0, len(hyperparams)), columns=['alpha', 'beta', 'K'])
        for i in range(len(hyperparams)):
            frame.loc[i] = hyperparams[i]
        print('\n', frame, '\n')
        """start plotting figures"""
        for (alpha, K), indices in frame.groupby(['alpha', 'K']).groups.items():
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(8)
            all_nmi = list()
            for i in indices:
                beta = frame.loc[i]['beta']
                tw_cluster_pred_iter = res_list[i]
                iter_x = range(len(tw_cluster_pred_iter))
                nmi_y = [au.score(label, pred, 'nmi') for pred in tw_cluster_pred_iter]
                all_nmi.append(nmi_y)
                plt.plot(iter_x, nmi_y, '-', lw=1.5, label='beta={}'.format(round(beta, 2)))
            plt.xlabel('iteration')
            plt.ylabel('NMI')
            plt.ylim(0.0, 0.75)
            plt.title('K=' + str(K))
            plt.legend(loc='lower right')
            plt.grid(True, '-', color='#333333', lw=0.8)
            plt.text(iter_num - 40, 0.70, 'final nmi: ' + str(round(max([nmi[-1] for nmi in all_nmi]), 6)),
                     fontsize=14, verticalalignment='bottom', horizontalalignment='left')
            plt.savefig(getcfg().dc_test + 'GSDMM/' + 'alpha={},K={}.png'.format(round(alpha, 2), K))
        # fu.dump_array('GSDMM_result.txt', res_list)
        
    @staticmethod
    def GSDMM_twarr(twarr, alpha, beta, K, iter_num):
        ner_pos_token = tk.key_wordlabels
        twarr = twarr[:]
        words = dict()
        """pre-process the tweet text, including dropping non-common terms"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                if not cs.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            for wordlabel in wordlabels:
                word = wordlabel[0]
                if word in words:
                    words[word]['freq'] += 1
                else:
                    words[word] = {'freq': 1, 'id': len(words.keys())}
        min_df = 3
        for w in list(words.keys()):
            if words[w]['freq'] < min_df:
                del words[w]
        for idx, w in enumerate(sorted(words.keys())):
            words[w]['id'] = idx
        for tw in twarr:
            tw['dup'] = dict(Counter([wlb[0] for wlb in tw[ner_pos_token] if wlb[0] in words]))
        """definitions of parameters"""
        V = len(words.keys())
        D = len(twarr)
        alpha0 = K * alpha
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
            freq_dict = twarr[d]['dup']
            for word in freq_dict.keys():
                n_z[cluster] += freq_dict[word]
                n_zw[cluster][words[word]['id']] += freq_dict[word]
        """make sampling using current counting information"""
        small_double = 1e-150
        large_double = 1e150
        def recompute(prob, underflowcount):
            max_count = max(underflowcount)
            return [prob[k] * (large_double ** (underflowcount[k] - max_count)) for k in range(len(prob))]
        def sample_cluster(tw, iter=None):
            prob = [0] * K
            underflowcount = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D - 1 + alpha0)
                rule_value = 1.0
                i = 0
                freq_dict = tw['dup']
                for w, freq in freq_dict.items():
                    for j in range(freq):
                        wid = words[w]['id']
                        rule_value *= (n_zw[k][wid] + beta + j) / (n_z[k] + beta0 + i)
                        if rule_value < small_double:
                            underflowcount[k] -= 1
                            rule_value *= large_double
                        i += 1
                prob[k] *= rule_value
            
            prob = recompute(prob, underflowcount)
            if iter is not None and iter > 95:
                return np.argmax(prob)
            else:
                return au.sample_index(np.array(prob))
        
        """start iteration"""
        z_iter = list()
        for i in range(iter_num):
            z_iter.append(z[:])
            
            for d in range(D):
                cluster = z[d]
                m_z[cluster] -= 1
                freq_dict = twarr[d]['dup']
                for word in freq_dict.keys():
                    wordid = words[word]['id']
                    wordfreq = freq_dict[word]
                    n_zw[cluster][wordid] -= wordfreq
                    n_z[cluster] -= wordfreq
                
                cluster = sample_cluster(twarr[d], i)
                
                z[d] = cluster
                m_z[cluster] += 1
                for word in freq_dict.keys():
                    wordid = words[word]['id']
                    wordfreq = freq_dict[word]
                    n_zw[cluster][wordid] += wordfreq
                    n_z[cluster] += wordfreq
        
        z_iter.append(z[:])
        return z_iter
