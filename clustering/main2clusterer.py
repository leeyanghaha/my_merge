# def exec_temp(parser):
#     """find tws that classifier fail to recognize on dzs_neg_data, and split them into blocks"""
#     event_extractor = EventExtractor(parser.get_dict_file_name(), parser.get_param_file_name())
#     dzs_neg_data = "/home/nfs/cdong/tw/testdata/non_pos_tweets.sum"
#     dzs_neg_twarr = fu.load_array(dzs_neg_data)
#     preds = event_extractor.make_classification(dzs_neg_twarr)
#     false_pos_twarr = [dzs_neg_twarr[idx] for idx, pred in enumerate([pred for pred in preds]) if pred > 0.5]
#     # false_pos_twarr_blocks = au.array_partition(false_pos_twarr, [1] * 10)
#     fu.dump_array("falseevents.txt", false_pos_twarr)
#
#     """query for pos events into blocks"""
#     data_path = getcfg().summary_path
#     log_path = "/home/nfs/cdong/tw/testdata/yli/queried_events_with_keyword/"
#     twarr_blocks = main2parser.query_per_query_multi(data_path, parser.seed_query_list)
#     print("query done, {} events, {} tws".format(len(twarr_blocks), sum([len(arr) for arr in twarr_blocks])))
#     event_id2info = dict()
#     tu.start_ner_service(pool_size=16, classify=True, pos=True)
#     for i in range(len(parser.seed_query_list)):
#         event_id2info[i] = dict()
#         twarr_blocks[i] = tu.twarr_ner(twarr_blocks[i])
#         query_i = parser.seed_query_list[i]
#         file_name_i = query_i.all[0].strip("\W") + "_" + "-".join(query_i.since) + ".sum"
#         event_id2info[i]["filename"] = file_name_i
#         event_id2info[i]["all"] = [w.strip("\W") for w in query_i.all]
#         event_id2info[i]["any"] = [w.strip("\W") for w in query_i.any]
#         fu.dump_array(log_path + file_name_i, twarr_blocks[i])
#     fu.dump_array(log_path + "event_id2info.txt", [event_id2info])
#     tu.end_ner_service()
#     print(event_id2info)
#     print("ner done")
#     fu.dump_array("events.txt", twarr_blocks)
#
#     """splitting dzs_neg_data into blocks"""
#     twarr_blocks = list()
#     dzs_neg_data = "/home/nfs/cdong/tw/testdata/non_pos_tweets.sum"
#     dzs_neg_parts = au.array_partition(fu.load_array(dzs_neg_data), (1, 1, 1, 1))
#     for dzs_part in dzs_neg_parts:
#         twarr_blocks.append(au.random_array_items(dzs_part, 300))
#     my_neg_data = "/home/nfs/cdong/tw/seeding/Terrorist/queried/Terrorist_counter.sum"
#     my_neg_parts = au.array_partition(fu.load_array(my_neg_data), (1, 1, 1, 1))
#     for my_part in my_neg_parts:
#         twarr_blocks.append(au.random_array_items(my_part, 300))
#
#     print("tweet distribution ", [len(twarr) for twarr in twarr_blocks], "\n\rtotal cluster", len(twarr_blocks))
#     fu.dump_array("nonevents.txt", twarr_blocks)

# def create_korea_batches_through_time(batch_size):
#     print("data source : korea")
#     false_twarr = fu.load_array("./data/falseevents.txt")
#     event_blocks = fu.load_array("./data/events.txt")
#     event_blocks.append(false_twarr)
#     non_korea_twarr = au.merge_array(event_blocks)
#     non_korea_twarr = sorted(non_korea_twarr, key=lambda item: item.get(tk.key_id))
#     twarr_blocks = fu.load_array("/home/nfs/cdong/tw/seeding/NorthKorea/korea.json")
#     twarr_blocks.append(non_korea_twarr)
#     for idx, twarr in enumerate(twarr_blocks):
#         print(idx, len(twarr), end=" -> ")
#         tflt.filter_twarr_dup_id(twarr)
#         print(len(twarr))
#     """ allocate a label for every tweet """
#     for idx, twarr in enumerate(twarr_blocks):
#         for tw in twarr:
#             tw.setdefault(tk.key_event_label, idx)
#     twarr = au.merge_array(twarr_blocks)
#     """ rearrange indexes """
#     def random_idx_for_item(item_arr, dest_item):
#         from numpy import random
#         def sample(prob):
#             return random.rand() < prob
#         non_dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] not in dest_item]
#         dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] in dest_item]
#         non_dest_cnt = dest_cnt = 0
#         res = list()
#         while len(non_dest_item_idx) > non_dest_cnt and len(dest_item_idx) > dest_cnt:
#             if sample((len(dest_item_idx) - dest_cnt) /
#                       (len(dest_item_idx) - dest_cnt + len(non_dest_item_idx) - non_dest_cnt)):
#                 res.append(dest_item_idx[dest_cnt])
#                 dest_cnt += 1
#             else:
#                 res.append(non_dest_item_idx[non_dest_cnt])
#                 non_dest_cnt += 1
#         while len(non_dest_item_idx) > non_dest_cnt:
#             res.append(non_dest_item_idx[non_dest_cnt])
#             non_dest_cnt += 1
#         while len(dest_item_idx) > dest_cnt:
#             res.append(dest_item_idx[dest_cnt])
#             dest_cnt += 1
#         return res
#     lbarr = [tw.get(tk.key_event_label) for tw in twarr]
#     idx_rearrange = random_idx_for_item(lbarr, {max(lbarr)})
#     twarr = [twarr[idx] for idx in idx_rearrange]
#     """ full split twarr & label """
#     full_idx = [i for i in range(len(twarr))]
#     batch_num = int(math.ceil(len(twarr) / batch_size))
#     tw_batches = [[twarr[j] for j in full_idx[i*batch_size: (i+1)*batch_size]] for i in range(batch_num)]
#     lb_batches = [[tw.get(tk.key_event_label) for tw in tw_batch] for tw_batch in tw_batches]
#     return tw_batches, lb_batches
