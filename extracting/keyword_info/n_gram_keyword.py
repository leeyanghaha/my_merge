from collections import Counter

import utils.function_utils as fu
import utils.timer_utils as tmu
import utils.pattern_utils as pu

import numpy as np


long_word2seg = dict()


def _segment_long_word(word):
    """
    传入一个单词，将其进行拆分；由于拆分代价较大，缓存拆分结果，且应当尽量只对过长的单词使用
    :param word: str，输入单词
    :return: list，每个元素为str；若word确实由多个单词拼接而成，则拆分为这些单词；否则list中仅有word本身
    """
    if word not in long_word2seg:
        long_word2seg[word] = pu.segment(word)
    return long_word2seg[word]


def _is_tokens_all_stopwords(tokens):
    """
    判断一组分词中，是否所有分词均为停用词或长度仅为1的单词
    :param tokens: list，分词表，每个元素为str，表示一个分词
    :return: bool，判断结果
    """
    stop_cnt = 0
    for token in tokens:
        if len(token) <= 1 or pu.is_stop_word(token, pu.my_stop_words):
            stop_cnt += 1
    return stop_cnt == len(tokens)


def valid_tokens_of_text(text):
    """
    对输入文本进行分词；其中，长度超过16的单词进行长词拆分，其结果中每个分词均视为合法分词
    :param text: str，待分词的文本
    :return: list，分词表，每个元素为str，表示一个分词
    """
    raw_tokens = pu.findall(pu.tokenize_pattern, text)
    seg_tokens = list()
    for token in raw_tokens:
        seg_tokens.extend(_segment_long_word(token)) if len(token) > 16 else seg_tokens.append(token)
    return seg_tokens


def tokens2n_grams(tokens, n):
    """
    从分词表中构造n-gram(bigram,trigram, ...)；n=1时进行额外处理，包括去除停用词，判断单词是否含字母等
    :param tokens: list，分词表，每个元素为str，表示一个分词
    :param n: gram中分词的数量
    :return: list，gram词表，每个元素为str，表示一个gram
    """
    if len(tokens) < n:
        return list()
    grams = list()
    if n <= 1:
        for token in tokens:
            if pu.findall(r'[a-zA-Z0-9]+', token) and not pu.is_stop_word(token, pu.my_stop_words):
                grams.append(token)
    else:
        for i in range(len(tokens) - n + 1):
            cur_tokens = tokens[i: i + n]
            if not _is_tokens_all_stopwords(cur_tokens):
                words = ' '.join(cur_tokens)
                grams.append(words)
    return grams


def _get_n2grams(tokens_list, n_range):
    """
    给定多个分词表，根据 n_range 指定的n的范围构造 grams；即找出一组文本中能构造的所有gram，并进行计数
    :param tokens_list: list，每个元素为list，即分词表
    :param n_range: (iterable)，需要构建 gram 的n的范围
    :return:
    """
    n2grams = dict()
    for n in n_range:
        ngrams = list()
        for tokens in tokens_list:
            ngrams.extend(tokens2n_grams(tokens, n))
        n2grams[n] = Counter(ngrams).most_common()
    return n2grams


def _rescore_grams(n2grams, len_thres, corpus_len):
    """
    重排序所有 grams，根据当前使用的文本数量、n的大小对不同n的grams进行加权；用于提升关键词提取质量
    :param n2grams: list，每个元素为str，即一个gram
    :param len_thres: int，任何gram的最小长度阈值，低于该阈值的gram将被过滤
    :param corpus_len: int，构成 n2grams 所使用的文本的数量
    :return:
    """
    combine_list = list()
    for n, grams in n2grams.items():
        weight_of_n = np.tanh(corpus_len / 50) * np.sqrt(n)
        for word, freq in grams:
            if len(word) >= len_thres:
                combine_list.append((word, freq * weight_of_n))
    word_score_list = sorted(combine_list, key=lambda item: item[1], reverse=True)
    return word_score_list


def get_quality_keywords(tokens_list, n_range, len_thres):
    """
    从分词表中构造 grams，并进行优化过滤，获得质量较高的 grams
    :param tokens_list: list，每个元素为list，即分词表
    :param n_range: (iterable)，需要构建 gram 的n的范围
    :param len_thres: int，任何gram的最小长度阈值，低于该阈值的gram将被过滤
    :return: list，每个元素为str，即一个gram，视其为返回的关键词
    """
    n2grams = _get_n2grams(tokens_list, n_range)
    gram_score_list = _rescore_grams(n2grams, len_thres, len(tokens_list))
    keywords = [gram for gram, score in gram_score_list]
    return keywords


def get_quality_n_gram(textarr, n_range, len_thres):
    """
    包装 get_quality_keywords，使得调用者能够仅传递文本列表而获得关键词提取结果，简化调用
    :param textarr: list，每个元素为str，即一条文本
    :param n_range: (iterable)，需要构建 gram 的n的范围
    :param len_thres: int，任何gram的最小长度阈值，低于该阈值的gram将被过滤
    :return: list，每个元素为str，即一个gram，视其为返回的关键词
    """
    posttextarr = [text.lower().strip() for text in textarr]
    tokens_list = [valid_tokens_of_text(text) for text in posttextarr]
    keywords = get_quality_keywords(tokens_list, n_range, len_thres)
    return tokens_list, keywords


if __name__ == "__main__":
    # from utils.array_utils import group_textarr_similar_index
    # file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-03-26_suicide-bomb_Lahore.json"
    # twarr = fu.load_array(file)
    # textarr = [tw[tk.key_text] for tw in twarr]
    # tmu.check_time()
    file = "/home/nfs/cdong/tw/src/calling/tmp/0.txt"
    _textarr = fu.read_lines(file)
    _tokens_list = [valid_tokens_of_text(text.lower().strip()) for text in _textarr]
    tmu.check_time()
    _keywords = get_quality_keywords(_tokens_list, n_range=range(1, 5), len_thres=20)
    tmu.check_time()
    print(_keywords)
    # _ngrams = get_ngrams_from_textarr(_textarr, 4)
    # _reorder_list = reorder_grams(_ngrams, 100)
    # _keywords = [w for w, f in _reorder_list]
    # print(_keywords)
    # idx_g, word_g = group_textarr_similar_index(_keywords, 0.2)
    # for g in word_g:
    #     print(g)
