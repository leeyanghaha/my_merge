import numpy as np

import utils.array_utils as au
import utils.tweet_keys as tk
from clustering.cluster_service import ClusterService


class LECMClusterer:
    @staticmethod
    def LECM_twarr_with_label(twarr, tw_cluster_label):
        # Currently best hyperparam 1, 0.1, 0.1, 1
        # tw_topic_arr, tw_cluster_pred = LECMClusterer.LECM_twarr(twarr, 1, 0.1, 0.1, 1, 20, 1)
        # print('one epoch:alpha {:<5}, eta {:<5}, beta {:<5}, lambd {:<5}, NMI {:<8}\n'.
        #       format(0.1, 0.1, 0.1, 0.1, au.score(tw_cluster_pred, tw_cluster_label, 'nmi')))
        tw_topic_arr = tw_cluster_pred = nmi = 0
        for alpha in [1]:
            for eta in [0.1]:
                for beta in [0.1]:
                    for lambd in [1]:
                        tw_topic_arr_, tw_cluster_pred_ = LECMClusterer.LECM_twarr(twarr, alpha, eta, beta, lambd, 20, 70)
                        nmi_ = au.score(tw_cluster_pred_, tw_cluster_label, 'nmi')
                        print('alpha {:<5}, eta {:<5}, beta {:<5}, lambd {:<5}, NMI{:<8}'.
                              format(alpha, eta, beta, lambd, nmi_))
                        if nmi < nmi_:
                            tw_topic_arr, tw_cluster_pred = tw_topic_arr_, tw_cluster_pred_
                            nmi = nmi_
        return tw_topic_arr, tw_cluster_pred
    
    @staticmethod
    def LECM_twarr(twarr, alpha, eta, beta, lambd, K, iter_num):
        ner_pos_token = tk.key_wordlabels
        twarr = twarr[:]
        geo_word_id_dict = dict()
        ent_word_id_dict = dict()
        key_word_id_dict = dict()
        
        def word_count_id(word_dict, w):
            if w in word_dict:
                word_dict[w]['freq'] += 1
            else:
                word_dict[w] = {'freq': 1, 'id': word_dict.__len__()}
    
        def word_count_freq(word_dict, w):
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
        
        """for every tweet, count all its elements into the corresponding dictionary"""
        for tw in twarr:
            wordlabels = tw[ner_pos_token]
            for i in range(len(wordlabels) - 1, -1, -1):
                wordlabels[i][0] = wordlabels[i][0].lower().strip('#').strip()
                if not ClusterService.is_valid_keyword(wordlabels[i][0]):
                    del wordlabels[i]
            tw['geo_and_time'] = dict()
            tw['ent'] = dict()
            tw['key'] = dict()
            for wordlabel in wordlabels:
                word = wordlabel[0]
                ner = wordlabel[1]
                if 'geo_and_time' in ner:
                    word_count_id(geo_word_id_dict, word)
                    word_count_freq(tw['geo_and_time'], word)
                elif not ner.startswith('O'):
                    word_count_id(ent_word_id_dict, word)
                    word_count_freq(tw['ent'], word)
                else:
                    word_count_id(key_word_id_dict, word)
                    word_count_freq(tw['key'], word)
    
        # iterNum = 70
        # K = 40
        # alpha = 0.1
        # eta = 0.1
        # beta = 0.5
        # lambd = 0.3
        """cluster hot_and_level"""
        D = twarr.__len__()
        alpha0 = alpha * K
        """geo_and_time hot_and_level"""
        L = geo_word_id_dict.__len__()
        eta0 = eta * L
        """non geo_and_time hot_and_level"""
        Y = ent_word_id_dict.__len__()
        beta0 = beta * Y
        """keyword_info hot_and_level"""
        V = key_word_id_dict.__len__()
        lambd0 = lambd * V
    
        print('D', D, 'L', L, 'Y', Y, 'V', V)
        # print('alpha', alpha, 'eta', eta, 'beta', beta, 'lambd', lambd)
    
        z = [0] * D
        m_z = [0] * K
        n_z_geo = [0] * K
        n_z_ent = [0] * K
        n_z_key = [0] * K
        n_zw_geo = [[0] * L for _ in range(K)]
        n_zw_ent = [[0] * Y for _ in range(K)]
        n_zw_key = [[0] * V for _ in range(K)]
    
        """initialize the counting arrays"""
        for d in range(D):
            cluster = int(K * np.random.random())
            z[d] = cluster
            m_z[cluster] += 1
            tw_geo_freq_dict = twarr[d]['geo_and_time']
            tw_ent_freq_dict = twarr[d]['ent']
            tw_key_freq_dict = twarr[d]['key']
            for word in tw_geo_freq_dict.keys():
                n_z_geo[cluster] += tw_geo_freq_dict[word]
                n_zw_geo[cluster][geo_word_id_dict[word]['id']] += tw_geo_freq_dict[word]
            for word in tw_ent_freq_dict.keys():
                n_z_ent[cluster] += tw_ent_freq_dict[word]
                n_zw_ent[cluster][ent_word_id_dict[word]['id']] += tw_ent_freq_dict[word]
            for word in tw_key_freq_dict.keys():
                n_z_key[cluster] += tw_key_freq_dict[word]
                n_zw_key[cluster][key_word_id_dict[word]['id']] += tw_key_freq_dict[word]
    
        """make sampling using current counting"""
        def rule_value_of(tw_freq_dict_, word_id_dict, n_z_, n_zw_, p, p0, cluster):
            b = 1
            value = 1.0
            for w_, w_count in tw_freq_dict_.items():
                for idx in range(1, w_count + 1):
                    wid = word_id_dict[w_]['id']
                    value *= (n_zw_[cluster][wid] + idx + p) / (n_z_[cluster] + b + p0)
                    b += 1
            return value
    
        def sample_cluster(tw):
            geo_freq_dict = tw['geo_and_time']
            ent_freq_dict = tw['ent']
            key_freq_dict = tw['key']
            prob = [0] * K
            for k in range(K):
                prob[k] = (m_z[k] + alpha) / (D + alpha0)
                prob[k] *= rule_value_of(geo_freq_dict, geo_word_id_dict, n_z_geo, n_zw_geo, eta, eta0, k)
                prob[k] *= rule_value_of(ent_freq_dict, ent_word_id_dict, n_z_ent, n_zw_ent, beta, beta0, k)
                prob[k] *= rule_value_of(key_freq_dict, key_word_id_dict, n_z_key, n_zw_key, lambd, lambd0, k)
                # bb=1.0
                # b = 1
                # rule_value = 1.0
                # for geo_w, w_count in geo_freq_dict.items():
                #     for idx in range(1, w_count + 1):
                #         wid = geo_word_id_dict[geo_w]['id']
                #         rule_value *= (n_zw_geo[k][wid] + idx + eta)/(n_z_geo[k] + b + eta0)
                #         b += 1
                # bb*=rule_value
                # b = 1
                # rule_value = 1.0
                # for ent_w, w_count in ent_freq_dict.items():
                #     for idx in range(1, w_count + 1):
                #         wid = ent_word_id_dict[ent_w]['id']
                #         rule_value *= (n_zw_ent[k][wid] + idx + beta)/(n_z_ent[k] + b + beta0)
                #         b += 1
                # bb *= rule_value
                # b = 1
                # rule_value = 1.0
                # for key_w, w_count in key_freq_dict.items():
                #     for idx in range(1, w_count + 1):
                #         wid = key_word_id_dict[key_w]['id']
                #         rule_value *= (n_zw_key[k][wid] + idx + lambd)/(n_z_key[k] + b + lambd0)
                #         b += 1
                # bb *= rule_value
                # print(aa-bb)
    
                # if rule_value < smallDouble:
                #     underflowcount[k] -= 1
                #     rule_value *= largeDouble
                # prob = recompute(prob, underflowcount)
            return au.sample_index(np.array(prob))
    
        def update_using_freq_dict(tw_freq_dict_, word_id_dict, n_z_, n_zw_, factor):
            for w, w_freq in tw_freq_dict_.items():
                w_id = word_id_dict[w]['id']
                w_freq *= factor
                n_z_[cluster] += w_freq
                n_zw_[cluster][w_id] += w_freq
    
        for i in range(iter_num):
            if iter_num > 10:
                print(str(i) + '\t' + str(m_z) + '\n' if i % int((iter_num + 1) / 10) == 0 else '', end='')
            for d in range(D):
                tw = twarr[d]
                tw_geo_freq_dict = tw['geo_and_time']
                tw_ent_freq_dict = tw['ent']
                tw_key_freq_dict = tw['key']
    
                cluster = z[d]
                m_z[cluster] -= 1
                # for geo_w, geo_w_freq in tw_geo_freq_dict.items():
                #     geo_w_id = geo_words[geo_w]['id']
                #     n_z_geo[cluster] -= geo_w_freq
                #     n_zw_geo[cluster][geo_w_id] -= geo_w_freq
                # for ent_w, ent_w_freq in tw_ent_freq_dict.items():
                #     ent_w_id = ent_words[ent_w]['id']
                #     n_z_ent[cluster] -= ent_w_freq
                #     n_zw_ent[cluster][ent_w_id] -= ent_w_freq
                # for key_w, key_w_freq in tw_key_freq_dict.items():
                #     key_w_id = key_words[key_w]['id']
                #     n_z_key[cluster] -= key_w_freq
                #     n_zw_key[cluster][key_w_id] -= key_w_freq
                update_using_freq_dict(tw_geo_freq_dict, geo_word_id_dict, n_z_geo, n_zw_geo, -1)
                update_using_freq_dict(tw_ent_freq_dict, ent_word_id_dict, n_z_ent, n_zw_ent, -1)
                update_using_freq_dict(tw_key_freq_dict, key_word_id_dict, n_z_key, n_zw_key, -1)
    
                cluster = sample_cluster(tw)
    
                z[d] = cluster
                m_z[cluster] += 1
                update_using_freq_dict(tw_geo_freq_dict, geo_word_id_dict, n_z_geo, n_zw_geo, 1)
                update_using_freq_dict(tw_ent_freq_dict, ent_word_id_dict, n_z_ent, n_zw_ent, 1)
                update_using_freq_dict(tw_key_freq_dict, key_word_id_dict, n_z_key, n_zw_key, 1)
    
        tw_topic_arr = [[] for _ in range(K)]
        for d in range(D):
            tw_topic_arr[z[d]].append(twarr[d])
    
        def print_top_freq_word_in_dict(word_id_dict, n_zw_, cluster):
            c = 1
            freq_list = [[word, n_zw_[cluster][word_id_dict[word]['id']]]
                         for word in word_id_dict.keys() if
                         n_zw_[cluster][word_id_dict[word]['id']] >= 10]
            for pair in sorted(freq_list, key=lambda x: x[1], reverse=True)[:20]:
                print('{:<15}{:<5}'.format(pair[0], pair[1]), end='\n' if c % 5 == 0 else '\t')
                c += 1
            print('\n' if (c - 1) % 5 != 0 else '', end='')
    
        for i, twarr in enumerate(tw_topic_arr):
            if len(twarr) == 0:
                continue
            print('\ncluster', i, '\t\ttweet number', len(twarr))
            print('-geo_and_time')
            print_top_freq_word_in_dict(geo_word_id_dict, n_zw_geo, i)
            print('-ent')
            print_top_freq_word_in_dict(ent_word_id_dict, n_zw_ent, i)
            print('-key')
            print_top_freq_word_in_dict(key_word_id_dict, n_zw_key, i)
    
        return tw_topic_arr, z
