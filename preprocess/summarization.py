import re

import utils.array_utils
import utils.multiprocess_utils
from utils.id_freq_dict import IdFreqDict
import utils.tweet_keys as tk
import utils.multiprocess_utils as mu
import utils.array_utils as au
import utils.function_utils as fu
import utils.file_iterator as fi
import utils.pattern_utils as pu
import utils.ark_service_proxy as ark
import preprocess.tweet_filter as tflt
from config.configure import getcfg


def summary_files_in_path(from_path, into_path=None):
    """ Read all .json under file_path, extract tweets from them into a file under summary_path. """
    # [-13:]--hour [-13:-3]--day [-13:-5]--month,ymdh refers to the short of "year-month-date-hour"
    from_path = fi.add_sep_if_needed(from_path)
    file_ymdh_arr = pu.split_digit_arr(fi.get_parent_path(from_path)[-13:])
    if not is_target_ymdh(file_ymdh_arr):
        return
    
    into_file = '{}{}'.format(fi.add_sep_if_needed(into_path), '_'.join(file_ymdh_arr) + '.sum')
    fi.remove_file(into_file)
    subfiles = fi.listchildren(from_path, children_type=fi.TYPE_FILE)
    file_block = mu.split_multi_format([(from_path + subfile) for subfile in subfiles], process_num=20)
    twarr_blocks = mu.multi_process(sum_files, [(file_list, tflt.FILTER_LEVEL_LOW) for file_list in file_block])
    twarr = au.merge_array(twarr_blocks)
    if twarr:
        fu.dump_array(into_file, twarr, overwrite=True)


def sum_files(file_list, filter_level):
    res_twarr = list()
    for file in file_list:
        twarr = fu.load_twarr_from_bz2(file) if file.endswith('.bz2') else fu.load_array(file)
        twarr = tflt.filter_twarr(twarr, filter_level)
        res_twarr.extend(twarr)
    return res_twarr


def summary_files_in_path_into_blocks(from_path, into_path, file_name):
    from_path = fi.add_sep_if_needed(from_path)
    sub_files = fi.listchildren(from_path, children_type=fi.TYPE_FILE, pattern='.json$')
    into_file = fi.add_sep_if_needed(into_path) + file_name
    twarr_block = list()
    for idx, file in enumerate(sub_files):
        from_file = from_path + file
        twarr = fu.load_array_catch(from_file)
        if len(twarr) <= 0:
            continue
        twarr = tflt.filter_twarr(twarr, tflt.FILTER_LEVEL_HIGH)
        twarr_block.append(twarr)
    print(sorted([('id'+str(idx), len(twarr)) for idx, twarr in enumerate(twarr_block)], key=lambda x: x[1]))
    print('event number in total: {}'.format(len(twarr_block)))
    fu.dump_array(into_file, twarr_block)


def is_target_ymdh(ymdh_arr):
    # ymdh_arr resembles ['201X', '0X', '2X', '1X']
    year = int(ymdh_arr[0])
    month = int(ymdh_arr[1])
    date = int(ymdh_arr[2])
    hour = int(ymdh_arr[3])
    # return 4 == month and date <= 2
    # import datetime
    # tw_time = datetime.datetime.strptime('-'.join(ymdh_arr[0:3]), '%Y-%m-%d')
    # start_time = datetime.datetime.strptime('2016-06-08', '%Y-%m-%d')
    # return (tw_time - start_time).days >= 0
    return True


def get_tokens_multi(file_path):
    file_path = fi.add_sep_if_needed(file_path)
    # subfiles = au.random_array_items(fi.listchildren(file_path, children_type=fi.TYPE_FILE), 20)
    subfiles = fi.listchildren(file_path, children_type=fi.TYPE_FILE)
    file_list_block = mu.split_multi_format([(file_path + subfile) for subfile in subfiles], process_num=20)
    res_list = mu.multi_process(get_tokens, [(file_list,) for file_list in file_list_block])
    id_freq_dict, total_doc_num = IdFreqDict(), 0
    for ifd, doc_num in res_list:
        total_doc_num += doc_num
        id_freq_dict.merge_freq_from(ifd)
    print('total_doc_num', total_doc_num, 'total vocabulary_size', id_freq_dict.vocabulary_size())
    id_freq_dict.drop_words_by_condition(3)
    id_freq_dict.dump_dict(getcfg().post_dict_file)


def get_tokens(file_list):
    id_freq_dict, total_doc_num = IdFreqDict(), 0
    for file in file_list:
        twarr = fu.load_array(file)
        total_doc_num += len(twarr)
        for tw in twarr:
            tokens = re.findall(r'[a-zA-Z_#\-]{3,}', tw[tk.key_text].lower())
            real_tokens = list()
            for token in tokens:
                if len(token) >= 16:
                    real_tokens.extend(pu.segment(token))
                else:
                    real_tokens.append(token)
            for token in real_tokens:
                if (not pu.is_stop_word(token)) and pu.has_azAZ(token) and 3 <= len(token):
                    id_freq_dict.count_word(token)
    id_freq_dict.drop_words_by_condition(2)
    print(id_freq_dict.vocabulary_size())
    return id_freq_dict, total_doc_num


K_IFD, K_FILE = 'ifd', 'file'


def get_semantic_tokens_multi(file_path):
    pos_type_info = {
        ark.prop_label: {K_IFD: IdFreqDict(), K_FILE: getcfg().pre_prop_dict_file},
        ark.comm_label: {K_IFD: IdFreqDict(), K_FILE: getcfg().pre_comm_dict_file},
        ark.verb_label: {K_IFD: IdFreqDict(), K_FILE: getcfg().pre_verb_dict_file},
        ark.hstg_label: {K_IFD: IdFreqDict(), K_FILE: getcfg().pre_hstg_dict_file},
    }
    total_doc_num = 0
    file_path = fi.add_sep_if_needed(file_path)
    # subfiles = au.random_array_items(fi.listchildren(file_path, children_type=fi.TYPE_FILE), 40)
    subfiles = fi.listchildren(file_path, children_type=fi.TYPE_FILE)
    file_list_block = mu.split_multi_format([(file_path + subfile) for subfile in subfiles], process_num=20)
    res_list = mu.multi_process(get_semantic_tokens, [(file_list,) for file_list in file_list_block])
    for res_type_info, doc_num in res_list:
        total_doc_num += doc_num
        for label in res_type_info.keys():
            pos_type_info[label][K_IFD].merge_freq_from(res_type_info[label][K_IFD])
    print('total_doc_num', total_doc_num)
    for label in pos_type_info.keys():
        ifd, file_name = pos_type_info[label][K_IFD], pos_type_info[label][K_FILE]
        ifd.drop_words_by_condition(3)
        if label != ark.hstg_label:
            ifd.drop_words_by_condition(lambda word, _: word.startswith('#'))
        ifd.dump_dict(file_name)
        print('{}; vocab size:{}'.format(file_name, ifd.vocabulary_size()))


def get_semantic_tokens(file_list):
    pos_type_info = {
        ark.prop_label: {K_IFD: IdFreqDict()},
        ark.comm_label: {K_IFD: IdFreqDict()},
        ark.verb_label: {K_IFD: IdFreqDict()},
        ark.hstg_label: {K_IFD: IdFreqDict()},
    }
    total_doc_num = 0
    for file in file_list:
        twarr = ark.twarr_ark(fu.load_array(file))
        total_doc_num += len(twarr)
        pos_tokens = au.merge_array([tw[tk.key_ark] for tw in twarr])
        for pos_token in pos_tokens:
            word = pos_token[0].strip().lower()
            if len(word) <= 2 or not pu.is_valid_keyword(word):
                continue
            real_label = ark.pos_token2semantic_label(pos_token)
            if real_label:
                pos_type_info[real_label][K_IFD].count_word(word)
    return pos_type_info, total_doc_num


# sep = os.path.sep
# def load_twarr_from_bz2_list(bz2_file_list):
#     print(len(bz2_file_list))
#     twarr = list()
#     for bz2_file in bz2_file_list:
#         if not bz2_file.endswith(".bz2"):
#             continue
#         twarr.extend(load_twarr_from_bz2(bz2_file))
#     return twarr
# def load_twarr_from_bz2_multi(bz2_list, p_num=15):
#     list_per_p = fu.split_multi_format(bz2_list, min(p_num, len(bz2_list)))
#     return fu.multi_process(load_twarr_from_bz2_list, [(l, ) for l in list_per_p])
# def tw_file_date(file_or_dir):
#     if fi.is_dir(file_or_dir):
#         dir_name = file_or_dir[:file_or_dir.rfind('\\')] if file_or_dir.endswith('\\') else file_or_dir
#         timestr = dir_name[-15:]
#     elif fi.is_file(file_or_dir):
#         file_name = file_or_dir
#         timestr = fi.base_name(file_name)[:file_name.rfind('.')]
#     else:
#         raise ValueError('file_or_dir wrong format:', file_or_dir)
#     return timestr
# def list_raw_data(data_path, start_ymdh, end_ymdh):
#     """ used for the purpose of special file structure, which is organized as the tree
#         data_path/year_month/date/hour/minute.bz2
#         :return A list of sublist, where each sublist contains the data within one day """
#     ymd = [ym + sep + d + sep for ym in fi.listchildren(data_path, children_type='dir')
#            for d in fi.listchildren(data_path + ym + sep, children_type='dir')
#            if du.is_target_ymdh(pu.split_digit_arr(ym + sep + d), start_ymdh, end_ymdh)]
#     ymd_hM = [[data_path + ymd_ + h + sep + M for h in fi.listchildren(data_path + ymd_, children_type='dir')
#                for M in fi.listchildren(data_path + ymd_ + sep + h, children_type='file')] for ymd_ in ymd]
#     for idx in range(len(ymd_hM) - 1, -1, -1):
#         if not ymd_hM[idx]:
#             ymd_hM.pop(idx)
#     return ymd_hM
# # data_path = getconfig().data_path
# # start_ymdh, end_ymdh = ['2016', '03', '2'], ['2016', '03', '6']
# # bz2_lists = list_raw_data(data_path, start_ymdh, end_ymdh)
# # for bz2_list_per_day in bz2_lists:
# #     load_twarr_from_bz2_multi(bz2_list_per_day[:150], p_num=15)
