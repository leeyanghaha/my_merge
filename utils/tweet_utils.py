import utils.date_utils as du
import utils.tweet_keys as tk
import utils.spacy_utils as su

import numpy as np


def twarr_operate(
        twarr,
        pre_condition=lambda tw: tw is not None,
        operation=lambda tw: tw,
        post_condition=lambda tw: tw is not None,
):
    res_list = list()
    for tw in twarr:
        if pre_condition and not pre_condition(tw):
            continue
        res = operation(tw)
        if post_condition and not post_condition(tw):
            continue
        res_list.append(res)
    return res_list


def in_reply_to(tw):
    in_reply_id = tw.get(tk.key_in_reply_to_status_id, None)
    return in_reply_id if type(in_reply_id) is int else None


# def twarr_vector_info(twarr, info_type='similarity'):
#     if len(twarr) == 0:
#         return None
#     if len(twarr) == 1:
#         return 1.0
#     # """ we have different weight for both word and tweet; for word it is defined by its pos,
#     #     for tweet it is defined by the weight and number of words which have vectors """
#     score_sum = 0
#     arr_sum_vec = None
#     vec_info_list = [weighted_doc_vec(tw.get(tk.key_spacy)) for tw in twarr]
#     for tw_ave_vec, word_vec_num, tw_score in vec_info_list:
#         if tw_ave_vec is None or word_vec_num == 0:
#             continue
#         score_sum += tw_score
#         scored_tw_vec = tw_score * tw_ave_vec
#         arr_sum_vec = (arr_sum_vec + scored_tw_vec) if arr_sum_vec is not None else scored_tw_vec
#     if arr_sum_vec is None:
#         if info_type == 'similarity':
#             return 0
#         elif info_type == 'vector':
#             return None
#     else:
#         arr_ave_vec = arr_sum_vec / score_sum
#         if info_type == 'similarity':
#             return np.mean(au.cosine_similarity(arr_ave_vec.reshape((1, -1)), [info[0] for info in vec_info_list]))
#         elif info_type == 'vector':
#             return arr_ave_vec
#
#
# def weighted_doc_vec(doc):
#     ent_type_0 = {'NORP', 'FAC', 'GPE', 'LOC', 'ORG', }
#     ent_type_1 = {'EVENT', 'PERSON', 'ORDINAL', 'CARDINAL', }
#     ent_type_2 = {'DATE', 'TIME', 'PERCENT', }
#     ent_weight_dict = dict([(t, 8.0) for t in ent_type_0] +
#                            [(t, 2.0) for t in ent_type_1] +
#                            [(t, 1.0) for t in ent_type_2])
#     # ent_weight = dict([(t, 1.0) for t in ent_type_0] +
#     # [(t, 1.0) for t in ent_type_1] +
#     # [(t, 1.0) for t in ent_type_2])
#     tag_type_0 = {'NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBN', 'VBP', 'VBX', }
#     tag_type_1 = {'IN', 'JJ', 'JJR', 'JJS', 'MD', 'RB', 'RBR', 'RBS', 'RP', }
#     tag_weight_dict = dict([(t, 8.0) for t in tag_type_0] + [(t, 1.0) for t in tag_type_1])
#     # tag_weight = dict([(t, 1.0) for t in tag_type_0] + [(t, 1.0) for t in tag_type_1])
#     word_vec_num = weight_sum = 0
#     doc_sum_vec = None
#     for token in doc:
#         word, ent_type, tag_type = token.text.lower(), token.ent_type_, token.tag_
#         ent_weight, tag_weight = ent_weight_dict.get(ent_type, 1), tag_weight_dict.get(tag_type, 0.1)
#         # ent_weight_, tag_weight_ = ent_weight.get(ent_type, 1), tag_weight.get(tag_type, 1.0)
#         if not pu.is_valid_keyword(word) and token.has_vector:
#             continue
#         word_vec_num += 1
#         weight = ent_weight * tag_weight
#         weight_sum += weight
#         wei_word_vec = weight * token.vector
#         doc_sum_vec = wei_word_vec if doc_sum_vec is None else (doc_sum_vec + wei_word_vec)
#     doc_ave_vec = (doc_sum_vec / weight_sum) if not word_vec_num == 0 else None
#     # (weight_sum / word_vec_num): average weight of words, these words should have vector
#     return doc_ave_vec, word_vec_num, (weight_sum / word_vec_num) if not word_vec_num == 0 else 0


def twarr_nlp(twarr,
              do_nlp=lambda tw: tk.key_spacy not in tw,
              get_text=lambda tw: tw.get(tk.key_text),
              set_doc=lambda tw, doc: tw.setdefault(tk.key_spacy, doc),
              nlp=None,
):
    """
    给定推特列表，为其中的每条推特进行spaCy的文本处理工作，并调用回调函数将处理结果存回推特的dict对象中
    :param twarr: list，推特列表，每个元素为dict
    :param do_nlp: callable，判断一条推特是否需要进行处理
    :param get_text: callable，用于从推特中提取文本
    :param set_doc: callable，用于向每条推特写会处理结果
    :param nlp: spaCy.language，包含若干语言模型与处理流水线
    :return: list，推特列表，其中每条判断为需要执行处理的推特会被inplace写入处理结果
    """
    # do_nlp_idx, textarr = list(), list()
    # for twidx, tw in enumerate(twarr):
    #     if do_nlp(tw):
    #         do_nlp_idx.append(twidx)
    #         textarr.append(get_text(tw))
    # docarr = su.textarr_nlp(textarr)
    # for docidx, twidx in enumerate(do_nlp_idx):
    #     set_doc(twarr[twidx], docarr[docidx])
    do_nlp_twarr = list()
    for tw in twarr:
        if do_nlp(tw):
            do_nlp_twarr.append(tw)
    txtarr = [get_text(tw) for tw in do_nlp_twarr]
    docarr = su.textarr_nlp(txtarr, nlp)
    for idx in range(len(do_nlp_twarr)):
        tw, doc = do_nlp_twarr[idx], docarr[idx]
        set_doc(tw, doc)
    return twarr


def twarr_timestamp_array(twarr):
    return [du.get_timestamp_form_created_at(tw[tk.key_created_at]) for tw in twarr]


def rearrange_idx_by_time(twarr):
    return np.argsort([du.get_timestamp_form_created_at(tw[tk.key_created_at].strip()) for tw in twarr])


# def recover_twarr_from_ids(id_list=None):
#     base = "/home/nfs/cdong/tw/origin/"
#     file_list = fi.listchildren(base, fi.TYPE_FILE, concat=True)[:1000]
#     file_list_block = mu.split_multi_format(file_list, 20)
#     if id_list is None:
#         id_list = fu.load_array("/home/nfs/cdong/tw/src/clustering/data/filtered_id_list.json")
#     id_set = set(id_list)
#     arg_list = [(id_set, files) for files in file_list_block]
#     print("id number:", len(id_set))
#     res_list = mu.multi_process(recover_ids_from_files, args_list=arg_list)
#     recovered_twarr = au.merge_array(res_list)
#     print("recovered tw number:", len(recovered_twarr))
#     fu.dump_array("/home/nfs/cdong/tw/src/clustering/data/filtered_.json", recovered_twarr)


# def recover_ids_from_files(id_set, file_list):
#     recovered_twarr = list()
#     for file in file_list:
#         recover = [tw for tw in fu.load_array(file) if (tw[tk.key_id] in id_set)]
#         recovered_twarr.extend(recover)
#     return recovered_twarr


# """ NER service interfaces """
#
#
# def start_ner_service(pool_size=8, classify=True, pos=True):
#     get_ner_service_pool().start(pool_size, classify, pos)
#
#
# def end_ner_service():
#     get_ner_service_pool().end()
#
#
# def twarr_ner(twarr, using_field=tk.key_text):
#     """ Perform NER and POS task upon the twarr, inplace. """
#     ner_text_arr = get_ner_service_pool().execute_ner_multiple([tw[using_field] for tw in twarr])
#     if not len(ner_text_arr) == len(twarr):
#         raise ValueError("Return line number inconsistent; Error occurs during NER")
#     for idx, ner_text in enumerate(ner_text_arr):
#         wordlabels = parse_ner_text_into_wordlabels(ner_text)
#         wordlabels = remove_badword_from_wordlabels(wordlabels)
#         twarr[idx][tk.key_wordlabels] = wordlabels
#     return twarr
#
#
# def parse_ner_text_into_wordlabels(ner_text):
#     # wordlabels = [('word_0', 'entity extraction 0', 'pos 0'), ('word_1', ...), ...]
#     words = re.split('\s+', ner_text)
#     wordlabels = list()
#     for word in words:
#         if word == '':
#             continue
#         wordlabels.append(parse_ner_word_into_labels(word, slash_num=2))
#     return wordlabels
#
#
# def parse_ner_word_into_labels(ner_word, slash_num):
#     """ Split a word into array by '/' searched from the end of the word to its begin.
#     :param ner_word: With pos labels.
#     :param slash_num: Specifies the number of "/" in the pos word.
#     :return: Assume that slash_num=2, "qwe/123"->["qwe","123"], "qwe/123/zxc"->["qwe","123","zxc"],
#                               "qwe/123/zxc/456"->["qwe/123","zxc","456"],
#     """
#     res = list()
#     over = False
#     for i in range(slash_num):
#         idx = ner_word.rfind('/') + 1
#         res.insert(0, ner_word[idx:])
#         ner_word = ner_word[0:idx - 1]
#         if idx == 0:
#             over = True
#             break
#     if not over:
#         res.insert(0, ner_word)
#     return res
#
#
# def remove_badword_from_wordlabels(wordlabels):
#     for idx, wordlabel in enumerate(wordlabels):
#         if re.search('^[^a-zA-Z0-9]+$', wordlabel[0]) is not None:
#             del wordlabels[idx]
#     return wordlabels


if __name__ == '__main__':
    import utils.function_utils as fu
    import utils.timer_utils as tmu
    file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-12-09_suicide-bomb_Istanbul.json"
    twarr = fu.load_array(file)
    textarr = [tw[tk.key_text].lower() for tw in twarr]
    tmu.check_time()
    # groups = group_textarr_similar_index(textarr, 0.2)
    # tmu.check_time()
    # print(groups, len(twarr), len((au.merge_array(groups))))
    # for g in groups:
    #     for idx in g:
    #         print(twarr[idx][tk.key_text])
    #     print('\n')
