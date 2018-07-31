from clustering.cluster_info_set import *
from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import *

import utils.array_utils as au
import utils.tweet_keys as tk
import utils.timer_utils as tmu

from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


# def parse_cluinfo_readable(cluid2info):
#     cluidarr = sorted(cluid2info.keys())
#     for cluid in cluidarr:
#         info = cluid2info[cluid]
#         print(cluid)
#         loc_freq_list, top_geo_info, hot, level, sorted_twarr = info
#
#         print("\nloc freq list")
#         print(loc_freq_list)
#         print("\ntop geo information")
#         for geo_info in top_geo_info:
#             print(geo_info)
#         print("\nhot: {}, level:{}".format(hot, level))
#         print("\nsorted twarr")
#         for idx, tw in enumerate(sorted_twarr):
#             print(idx, tw[tk.key_text])
#         print("\n\n\n")


def cluster_inter_sim_with_label(cludict):
    for cluster in cludict.values():
        cluster.extract_keywords()
    vec_label_arr = list()
    cluidarr = cludict.keys()
    for cluid1 in cluidarr:
        for cluid2 in cluidarr:
            if cluid1 == cluid2:
                continue
            cluster1, cluster2 = cludict[cluid1], cludict[cluid2]
            sim_vec = cluster1.similarity_vec_with(cluster2)
            rep_lb1, rep_lb2 = cluster1.get_rep_label(0.7), cluster2.get_rep_label(0.7)
            label = (rep_lb1 == rep_lb2 and rep_lb1 != -1)
            vec_label_arr.append(np.concatenate([sim_vec, [label]], axis=0))
    return vec_label_arr


def cluster_inter_sim(cludict):
    clf = joblib.load('./data/judge_merge_model')
    for cluster in cludict.values():
        cluster.extract_keywords()
    print("valid cluster num ", len(cludict))
    cluidarr = sorted(cludict.keys())
    for cluid1 in cluidarr:
        sim_list = list()
        for cluid2 in cluidarr:
            if cluid1 <= cluid2:
                continue
            cluster1, cluster2 = cludict[cluid1], cludict[cluid2]
            sim_vec = cluster1.similarity_vec_with(cluster2)
            sim_list.append(((cluster1, cluster2), sim_vec))
        def print_freq(word_freq):
            for idx, (word, freq) in enumerate(word_freq):
                if idx > 0 and idx % 5 == 0:
                    print()
                print('{:16}'.format('{:12} {}'.format(word, freq)), end='')
        if len(sim_list) < 1:
            continue
        probarr = clf.predict_proba(np.array([pair[1] for pair in sim_list]))[:, 1]
        max_idx = np.argmax(probarr)
        (c1, c2), vec = sim_list[max_idx]
        prob = probarr[max_idx]
        print("prob", prob, "cid1:{}, cid2:{}".format(c1.cluid, c2.cluid))
        print("-key-")
        print_freq(c1.keyifd.most_common(20))
        print("\n--")
        print_freq(c2.keyifd.most_common(20))
        print("\n-ent-")
        print_freq(c1.entifd.most_common(20))
        print_freq(c2.entifd.most_common(20))
        print("\n-----\n")

    # # n = len(cluidarr) - 1
    # # print(len(pair_list), (n + 1) * n / 2)
    #
    # from pprint import pprint
    #
    # for idx in range(len(probarr)):
    #     (cluster1, cluster2), sim_vec = pair_list[idx]
    #     prob = probarr[idx]
    #     print("prob", prob)
    #     print("\n--")
    #     print_freq(cluster1.keyifd.most_common(20))
    #     print_freq(cluster2.keyifd.most_common(20))
    #     print("\n--")
    #     print_freq(cluster1.entifd.most_common(20))
    #     print_freq(cluster2.entifd.most_common(20))
    #     print("\n-----\n")


def is_valid_cluster(cluster):
    return cluster.twnum > 5


class Analyzer:
    def __init__(self, hold_batch_num):
        self.hold_batch_num = hold_batch_num
        self.tw_batches, self.cluid_batches = list(), list()
        self.cludict = dict()
    
    def set_batches(self, tw_batches):
        self.tw_batches.clear()
        for batch_idx in range(len(tw_batches)):
            self.tw_batches.append(GSDPMMStreamIFDDynamic.pre_process_twarr(tw_batches[batch_idx]))
        cluid_set = set([tw[tk.key_event_cluid] for tw in au.merge_array(tw_batches)])
        self.cludict = dict([(cluid, ClusterHolder(cluid)) for cluid in cluid_set])
    
    def start_iter(self):
        cludict = self.cludict
        hold_batch_num = self.hold_batch_num
        sep_pattern = '\n--\n\n'
        siminfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_siminfo.txt'
        textinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_textinfo.txt'
        scoreinfo_pattern = '/home/nfs/cdong/tw/src/clustering/iteration/20180302/{:0>3}_scoreinfo.txt'
        batch_window = list()

        clf = ClassifierTerror()
        similarity_data = list()
        for batch_idx, twharr in enumerate(self.tw_batches):
            print('\r{}\rbatch {}/{}'.format(' ' * 20, batch_idx, len(self.tw_batches)), end='', flush=True)
            batch_window.append(twharr)
            for twh in twharr:
                twh.update_cluster(cludict[twh[tk.key_event_cluid]])
            if batch_idx < hold_batch_num:     # if tw batch number is not enough
                continue
            valid_cludict = dict([(cluid, clu) for cluid, clu in cludict.items() if is_valid_cluster(clu)])
            
            from extracting.keyword_info.n_gram_keyword import get_textarr_n_grams_list
            def print_cluster_info():
                for cluid, cluster in valid_cludict.items():
                    cluster.extract_keywords()
                    
                    # print("rep label", cluster.get_rep_label(0.7))
                    # print('--')
                    textarr = [tw[tk.key_text] for tw in cluster.get_twarr()]
                    
                    # print("is event score", clf.predict_average_proba(textarr))
                    # print('--')
                    # print("keywords", get_keywords(textarr))
                    print('--')
                    print("keywords", cluster.keyifd.most_common(20))
                    print('--')
                    print("ents", cluster.entifd.most_common(20))
                    print('--')
                    print("tw num", len(cluster.get_twarr()))
                    print('--')
                    for tw in cluster.get_twarr():
                        print(tw[tk.key_text])
                    print('\n\n----\n\n')
            
            def get_similarity_vecarr():
                vec_label_arr = cluster_inter_sim_with_label(valid_cludict)
                similarity_data.extend(vec_label_arr)
            
            """ """
            # get_similarity_vecarr()
            # print_cluster_info()
            cluster_inter_sim(valid_cludict)
            return
            
            oldest_twharr = batch_window.pop(0)
            for twh in oldest_twharr:
                twh.update_cluster(None)
        
        similarity_data = np.array(similarity_data, dtype=np.float32)
        print(similarity_data.shape)
        np.save('./data/data_sim', similarity_data)


def train_keyword_similarity(exec_train=True):
    np.random.seed(134590)
    data = np.load('./data/data_sim.npy')
    data_len = len(data)
    sep_idx = int(data_len * 0.8)
    # train, test = data[:sep_idx], data[sep_idx:]
    rand_idx = au.shuffle([i for i in range(data_len)])
    train, test = data[rand_idx[:sep_idx]], data[rand_idx[sep_idx:]]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    print(data.shape, train.shape, test.shape)
    
    if exec_train:
        clf = LogisticRegression()
        clf.fit(train_x, train_y)
        joblib.dump(clf, './data/judge_merge_model')
    else:
        clf = joblib.load('./data/judge_merge_model')
    
    # print("coef:", c.coef_.tolist())
    pred = clf.predict_proba(test_x)[:, 1]
    au.precision_recall_threshold(test_y, pred, thres_range=[i / 10 for i in range(1, 10)])


if __name__ == '__main__':
    info_set = filtered_info_set
    # """ train classifier for merging clusters """
    # train_keyword_similarity(exec_train=False)
    # exit()
    """ filtered twarr """
    tmu.check_time()
    _tw_batches = info_set.load_tw_batches(load_cluid_arr=True)
    tmu.check_time()
    
    """ handpicked twarr """
    # _tw_batches = get_labelled_tw_batches()
    # tmu.check_time()
    # tu.twarr_nlp(au.merge_array(_tw_batches))
    # tmu.check_time(print_func=lambda dt: print('{} tweets spacy over in {} s'.format(len(au.merge_array(_tw_batches)), dt)))
    # _cluid_batches = fu.load_array(cluid_evo_file)
    
    a = Analyzer(50)
    a.set_batches(_tw_batches)
    tmu.check_time()
    a.start_iter()
    tmu.check_time()
