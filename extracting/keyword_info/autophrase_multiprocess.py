import os
from subprocess import Popen, PIPE

from config.configure import getcfg
import utils.function_utils as fu
import utils.file_iterator as fi
import utils.tweet_keys as tk
import utils.timer_utils as tmu


autophrase_base = getcfg().autophrase_path
autophrase_output_base = fi.join(autophrase_base, "OUTPUTS/")  # 保证任何autophrase的输出限制到output_base之内的某个角落
command = fi.join(autophrase_base, "auto_phrase.sh")
fi.mkdir(autophrase_output_base)


def autophrase(input_text_file, output_path, commander, process_base, min_sup):
    p = Popen(commander, shell=True, bufsize=1, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=process_base)
    p.stdin.flush()
    p.stdin.write((input_text_file + '\n').encode("utf8"))
    p.stdin.write((output_path + '\n').encode("utf8"))
    p.stdin.write((min_sup + '\n').encode("utf8"))
    p.stdin.flush()
    p.communicate()


def copy_into_process_base(process_base):
    if not fi.exists(process_base):
        fi.mkdir(process_base)
        os.system("cp %s %s" % (command, process_base))
        os.system("cp %s %s" % (fi.join(autophrase_base, 'phrasal_segmentation.sh'), process_base))
        os.system("cp -r %s %s" % (fi.join(autophrase_base, 'bin'), process_base))
        os.system("cp -r %s %s" % (fi.join(autophrase_base, 'data'), process_base))
        os.system("cp -r %s %s" % (fi.join(autophrase_base, 'tools'), process_base))


def determine_min_sup(len_textarr):
    lookup = {(0, 1): 0, (1, 30): 1, (30, 100): 2, (100, 1000): 3, (1000, 10000): 4, (10000, 100000): 5, }
    for (lower, upper), sup in lookup.items():
        if lower <= len_textarr < upper:
            return str(sup)
    default_sup = 10
    return str(default_sup)


def autophrase_wrapper(process_code, textarr):
    # process_code用于辨识进程所占用的路径，textarr是一个文本list
    process_base = fi.join(autophrase_output_base, str(process_code))
    copy_into_process_base(process_base)
    commander = fi.join(process_base, "auto_phrase.sh")
    input_text_file = fi.join(process_base, "raw_train.txt")
    output_keyword_file = fi.join(process_base, "AutoPhrase.txt")
    # 将文本列表写入文件, 执行autophrase
    fu.write_lines(input_text_file, textarr)
    min_sup = determine_min_sup(len(textarr))
    autophrase(input_text_file, process_base, commander, process_base, min_sup)
    # 读取autophrase结果
    lines = fu.read_lines(output_keyword_file)
    conf_word_list = list()
    for line in lines:
        conf, word = line.split(maxsplit=1)
        conf_word_list.append((float(conf), word))
    # fi.rmtree(os.path.join(process_base, 'tmp'))
    return conf_word_list


# def autophrase_multi(twarr_list, process_num):
#     """ Returns a keyword list that has not been filtered yet """
#     def get_textarr(twarr):
#         return [tw[tk.key_text] for tw in twarr]
#     index = 0
#     conf_word_list = list()
#     while index < len(twarr_list):
#         twarr_batch = twarr_list[index: index + process_num]
#         arg_list = [(idx, get_textarr(twarr)) for idx, twarr in enumerate(twarr_batch)]
#         conf_word_list_batch = mu.multi_process(autophrase_wrapper, arg_list)
#         conf_word_list.extend(conf_word_list_batch)
#         index += process_num
#     assert len(conf_word_list) == len(twarr_list)
#     return conf_word_list


# def autophrase_multi_top_results(twarr_list, process_num, max_word_num):
#     conf_word_list = autophrase_multi(twarr_list, process_num)
#     return fileter_keyword_list(conf_word_list, max_word_num, conf_thres, word_len_thres)


# def fileter_keyword_list(conf_word_list, *args, **kwargs):
#     return [filter_keywords(keywords, *args, **kwargs) for keywords in conf_word_list]


def filter_keywords(conf_word_list, conf_thres, len_thres):
    return [word for conf, word in conf_word_list if conf > conf_thres and len(word) > len_thres]


def get_quality_autophrase(process_code, textarr, conf_thres, len_thres):
    conf_word_list = autophrase_wrapper(process_code, textarr)
    return filter_keywords(conf_word_list, conf_thres, len_thres)


if __name__ == '__main__':
    import utils.tweet_utils as tu
    
    # text_file = "/home/nfs/cdong/tw/src/extracting/3796_r.txt"
    # textarr = fu.read_lines(text_file)
    twarr_file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-01-11_blast_Istanbul.json"
    twarr_file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-03-26_suicide-bomb_Lahore.json"
    twarr = fu.load_array(twarr_file)
    textarr = [tw[tk.key_text] for tw in twarr]
    _conf_word_list = autophrase_wrapper(0, textarr)
    # _keywords = [item[1] for item in _conf_word_list]
    print(filter_keywords(_conf_word_list, 50))
    print('\n')
    print(textarr)
    # idx_groups = tu.group_textarr_similar_index(keywords, 0.2)
    # for g in idx_groups:
    #     print([keywords[i] for i in g], '\n')
    # print(_conf_word_list[:30])
    # print()
    # print(textarr)
    exit()
    
    """ 文本数量小于30时关键词的质量已经相当低，应尽量使进入的文本数量大于一定阈值 """
    """ __main__里面的内容保持不变，是最终的接口形式 """
    _keyword_file = 'keyword_results.json'
    
    _file_name_keywords_list = fu.load_array(_keyword_file)
    
    # for filename, keyword in _file_name_keywords_list:
    #     print(filename)
    #     print(filter_keywords(keyword, 20), '\n')
    exit()
    
    _base = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/"
    _files = fi.listchildren(_base, fi.TYPE_FILE, concat=True)
    _twarr_list = [fu.load_array(file) for file in _files]
    tmu.check_time()
    _file_name_list = [fi.get_name(file) for file in _files]
    _keywords_list = autophrase_multi(_twarr_list, process_num=8)  # 主要验证这个输出结果是否正确以及是否有加速
    tmu.check_time()
    _res = list(zip(_file_name_list, _keywords_list))
    fu.dump_array(_keyword_file, _res)
    # _keywords_list = fu.load_array(_keyword_file)
    # print(len(_keywords_list))
    # print([len(_keywords) for _keywords in _keywords_list])
    # assert len(_twarr_list) == len(_keywords_list)
    # for _idx, _keywords in enumerate(_keywords_list):
    #     # print('word num', len(_keywords), ', tw num', len(_twarr_list[idx]))
    #     if len(_keywords) < 20:
    #         print(len(_twarr_list[_idx]))
