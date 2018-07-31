from clustering.cluster_info_set import *
from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic


def stream_cluster_with_label(tw_batches, hold_batch_num):
    print('total tw={}, hold batch={}'.format(len(au.merge_array(tw_batches)), hold_batch_num))
    print('batch size={}, batch num={}'.format(len(tw_batches[0]), len(tw_batches)))
    g = GSDPMMStreamIFDDynamic()
    g.set_hyperparams(hold_batch_num, 30, 0.01)
    
    twid_cluid_batches = list()
    for batch_idx, tw_batch in enumerate(tw_batches):
        print('\r{}\r{}/{} batch, {} tws done'.format(
            ' ' * 30, batch_idx + 1, len(tw_batches), sum([len(t) for t in tw_batches[:batch_idx + 1]])),
            end='', flush=True)
        twid_cluid_iter = g.input_batch(tw_batch)
        if not twid_cluid_iter:
            continue
        twid_cluid_batches.extend(twid_cluid_iter)
    tmu.check_time(print_func=lambda dt: print('\ncluster over in {} s'.format(dt)))
    # assert len(cluid_batches) == len(tw_batches) and \
    #        len(au.merge_array(cluid_batches)) == len(au.merge_array(tw_batches))
    return twid_cluid_batches


# def analyze_stream_k(z_evo=None, l_evo=None, s_evo=None, rep_score=0.7):
#     """ evaluate the clustering results """
#     if not z_evo or not l_evo or not s_evo:
#         z_evo, l_evo, s_evo = fu.load_array('evolution.txt')[0]
#         for s in s_evo:
#             for k in list(s.keys()):
#                 if type(k) is str:
#                     s[int(k)] = s[k]
#                     s.pop(k)
#     all_z = set(au.merge_array(z_evo))
#     # print('\nhistory_z {}, recorded cluster number {}'.format(sorted(all_z), len(all_z)))
#     print('\nrecorded cluster number {}'.format(len(all_z)))
#
#     all_lb = set(au.merge_array(l_evo))
#     ne_lb = int(max(all_lb))
#     evo = pd.DataFrame(dtype=int)
#     for idx in range(len(z_evo)):
#         z_batch, lb_batch, s_batch = z_evo[idx], l_evo[idx], s_evo[idx]
#         df = cs.cluid_label_table(z_batch, lb_batch)
#
#         print('\r{}\r{}'.format(' '*5, idx), end='')
#         # print(sorted(set(df.columns)))
#         # print(sorted(set(s_batch.keys())))
#         diff1 = set(df.index).difference(set(s_batch.keys()))
#         diff2 = set(s_batch.keys()).difference(set(df.index))
#         if len(diff1) > 0 or len(diff2) > 0:
#             print('yashila', diff1, diff2)
#
#         for cluid, row in df.iterrows():
#             row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
#             rep_cluid = int(row.index[row_max_idx])  # which cluster is the representative
#             rep_twnum = int(row[rep_cluid])
#             is_k_score = round(s_batch[cluid], 3)
#             if row.loc[rep_cluid] == 0 or row.loc[rep_cluid] < row_sum * rep_score or rep_cluid == ne_lb:
#                 fill_cluid = ne_lb
#             else:
#                 fill_cluid = rep_cluid
#             evo.loc[idx, cluid] = '{: <4},{:0<5},{}'.format(fill_cluid, is_k_score, rep_twnum)
#             # evo.loc[idx, cluid] = (fill_cluid, is_k_score, rep_twnum)
#
#     evo.fillna('', inplace=True)
#     lb_distrb = Counter(au.merge_array(l_evo))
#     effective_cluids = list()
#     for info in au.merge_array(evo.values.tolist()):
#         if info == '':
#             continue
#         fill_cluid, is_k_score, rep_twnum = info.split(',')
#         fill_cluid = int(fill_cluid)
#         if info != '' and fill_cluid != ne_lb:
#             effective_cluids.append(fill_cluid)
#     detected_e_number = Counter(effective_cluids)
#     detected_e_id = sorted(detected_e_number.keys())
#     num_detected = len(detected_e_id)
#     event_id_corpus = sorted(all_lb.difference({ne_lb}))
#     num_corpus = len(event_id_corpus)
#
#     first_col = evo.columns[0]
#     evo.loc['eventdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, detected_e_number[eventid])
#                                                    for eventid in detected_e_id])
#     evo.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, lb_distrb[eventid])
#                                                   for eventid in sorted(lb_distrb.keys())])
#     evo.loc['detectedid', first_col] = str(detected_e_id)
#     evo.loc['totalevent', first_col] = str(event_id_corpus)
#     evo.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
#     evo.to_csv('table.csv')
#     print(detected_e_id, num_detected, num_corpus, num_detected / num_corpus)


# def grid_stream_cluster_with_label_multi(tw_batches, lb_batches, c_type=None):
#     print('batch num:{}, batch size:{}'.format(len(tw_batches), len(tw_batches[0])))
#
#     # a_range = [i for i in range(1, 10, 5)] + [i*10 for i in range(1, 10, 3)] + [i*100 for i in range(1, 10, 2)]
#     # b_range = [i/1000 for i in range(1, 10, 2)] + [i/100 for i in range(1, 10, 2)]
#     # params = [(a, b) for a in a_range for b in b_range]
#     # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
#     #                [(tw_batches, lb_batches, GSDPMMStreamClusterer(4), p) for p in params])
#
#     # a_range = [100, 10, 1, 0.1]
#     # k_range = [30, 50]
#     # p_range = c_range = v_range = h_range = [0.01, 0.1, 1]
#     # params = [(a, p, c, v, h, k) for a in a_range for p in p_range
#     #           for c in c_range for v in v_range for h in h_range for k in k_range]
#     # res_list = ClusterService.clustering_multi(EventExtractor.stream_cluster_with_label_multi,
#     #                 [(tw_batches, lb_batches, SemanticStreamClusterer(4), *p) for p in params])
#
#     a_range = [1000, 500, 300, 100, 50]
#     b_range = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
#     params = [(a, b) for a in a_range for b in b_range]
#     if c_type is None:
#         # g = GSDPMMStreamClustererIFDStatic(hold_batch_num=4)
#         # g = GSDPMMStreamClustererIFDDynamic(hold_batch_num=5)
#         # g = GSDPMMStreamClusterer(hold_batch_num=5)
#
#         print('grid for GSDPMMStreamSemanticDynamic')
#         a_r = [100, 400, 700]
#         p_r = c_r = v_r = h_r = [0.01, 0.05, 0.1]
#         params = [(a, p, c, v, h) for a in a_r for p in p_r for c in c_r for v in v_r for h in h_r]
#         g = GSDPMMStreamSemanticIFDDynamic(hold_batch_num=6)
#     else:
#         g = c_type(hold_batch_num=6)
#
#     mul_func, mul_target = cs.clustering_multi, stream_cluster_with_label_multi
#     res_list = mul_func(mul_target, [(tw_batches, lb_batches, g, p) for p in params], 10)
#     for idx, res in enumerate(sorted(res_list, key=lambda x: x[1], reverse=True)):
#         print(res)
#
#
# def stream_cluster_with_label_multi(tw_batches, lb_batches, clusterer, params):
#     from copy import deepcopy
#     clusterer = deepcopy(clusterer)
#     clusterer.set_hyperparams(*params)
#
#     z_evo, l_evo = list(), list()
#     for batch_id in range(len(tw_batches)):
#         z, label = clusterer.input_batch_with_label(tw_batches[batch_id], lb_batches[batch_id])
#         if not z:
#             continue
#         if not len(label) == len(z):
#             raise ValueError('length inconsistent')
#         z_evo.append([int(i) for i in z])
#         l_evo.append([int(i) for i in label])
#
#     """ evaluation """
#     all_labels = au.merge_array(l_evo)
#     rep_score = 0.7
#     non_event_label = int(max(all_labels))
#     evolution = pd.DataFrame(dtype=int)
#     for batch_id in range(len(z_evo)):
#         z_batch, lb_batch = z_evo[batch_id], l_evo[batch_id]
#         df = cs.cluid_label_table(z_batch, lb_batch)
#         for batch_id, row in df.iterrows():
#             row_max_idx, row_sum = np.argmax(row.values), sum(row.values)
#             max_cluid = int(row.index[row_max_idx])
#             if row[max_cluid] == 0 or row[max_cluid] < row_sum * rep_score or max_cluid == non_event_label:
#                 evolution.loc[batch_id, batch_id] = non_event_label
#             else:
#                 evolution.loc[batch_id, batch_id] = max_cluid
#
#     evolution.fillna(non_event_label, inplace=True)
#     real_event_num = Counter(all_labels)
#     detected_event_ids = [int(item) for item in au.merge_array(evolution.values.tolist()) if
#                           item != non_event_label]
#     detected_event_num = Counter(detected_event_ids)
#     detected_event_id = [d for d in sorted(set(detected_event_ids))]
#     num_detected = len(detected_event_id)
#     event_id_corpus = sorted(set(all_labels).difference({non_event_label}))
#     num_corpus = len(event_id_corpus)
#
#     # first_col = evolution.columns[0]
#     # evolution.loc['eventdistrb', first_col] = '  '.join(
#     #     ['{}:{}'.format(eventid, detected_event_num[eventid])
#     #      for eventid in sorted(detected_event_num.keys())])
#     # evolution.loc['realdistrb', first_col] = '  '.join(['{}:{}'.format(eventid, real_event_num[eventid])
#     #                                                     for eventid in sorted(real_event_num.keys())])
#     # evolution.loc['detectedid', first_col] = str(detected_event_id)
#     # evolution.loc['totalevent', first_col] = str(event_id_corpus)
#     # evolution.loc['recall', first_col] = '{}/{}={}'.format(num_detected, num_corpus, num_detected / num_corpus)
#     # evolution.fillna('', inplace=True)
#
#     info = clusterer.get_hyperparams_info()
#     return info, num_detected / num_corpus, num_detected, num_corpus


info_set = FilteredInfoSet()
batch_size = info_set.batch_size
hold_batch_num = info_set.hold_batch_num


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Clustering operation")
    parser.add_argument('-a', action='store_true', default=False)
    parser.add_argument('-g', action='store_true', default=False)
    parser.add_argument('-t', nargs='?', default='None')
    args = parser.parse_args()
    
    file = "/home/nfs/cdong/tw/src/clustering/data/events2016.txt"
    twarr_batches = fu.load_array(file)
    print(len(twarr_batches), len(au.merge_array(twarr_batches)))
    exit()
    
    tmu.check_time()
    _tw_batches = info_set.load_tw_batches(load_cluid_arr=False)
    tmu.check_time()
    _twid_cluid_batches = stream_cluster_with_label(_tw_batches, hold_batch_num)
    info_set.dump_cluidarr(au.merge_array(_twid_cluid_batches))
    tmu.check_time()
