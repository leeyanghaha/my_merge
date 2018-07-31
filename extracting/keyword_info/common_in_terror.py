from utils.utils_loader import *
from extracting.keyword_info.n_gram_keyword import valid_tokens_of_text, tokens2n_grams


def grams_of_textarr(textarr):
    grams = list()
    for text in textarr:
        text = text.lower().strip()
        tokens = valid_tokens_of_text(text)
        grams.extend(tokens2n_grams(tokens, 1))
        grams.extend(tokens2n_grams(tokens, 2))
        grams.extend(tokens2n_grams(tokens, 3))
    return Counter(grams)


def grams_of_files(file_list):
    grams = Counter()
    doc_num = 0
    for file in file_list:
        twarr = fu.load_array(file)
        textarr = [tw[tk.key_text] for tw in twarr]
        doc_num += len(textarr)
        grams += grams_of_textarr(textarr)
    return doc_num, grams


def word_freq_thres(counter, freq_thres):
    loc = -1
    word_freq_list = counter.most_common()
    for idx, (word, freq) in enumerate(word_freq_list):
        if freq < freq_thres:
            loc = idx + 1
            break
    return word_freq_list[:loc]


if __name__ == "__main__":
    pos_base = "/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/"
    neg_base = "/home/nfs/cdong/tw/origin/"
    # pos_file_list = fi.listchildren(pos_base, fi.TYPE_FILE, concat=True)
    # pos_doc_num, pos_grams = grams_of_files(pos_file_list)
    tmu.check_time()
    neg_file_list = fi.listchildren(neg_base, fi.TYPE_FILE, concat=True)[-1000:]
    neg_file_blocks = mu.split_multi_format(neg_file_list, 20)
    res_list = mu.multi_process(grams_of_files, args_list=[(file_list, ) for file_list in neg_file_blocks])
    neg_doc_total = sum([res[0] for res in res_list])
    neg_grams_total = Counter()
    for res in res_list:
        neg_grams_total += res[1]
    tmu.check_time()
    fu.dump_array('neg_common.json', word_freq_thres(neg_grams_total, 100))
    print(neg_doc_total)
