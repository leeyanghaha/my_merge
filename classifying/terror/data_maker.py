from pathlib import Path

import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.timer_utils as tmu

import classifying.fast_text_utils as ftu


def twarr2textarr(twarr):
    textarr = list()
    for tw in twarr:
        text = tw.get(tk.key_text).strip()
        if tk.key_orgntext not in tw:
            text = pu.text_normalization(text)
        if pu.is_empty_string(text):
            continue
        textarr.append(text)
    return textarr


def split_train_test(array):
    split = int(len(array) * 0.8)
    return array[:split], array[split:]


pos_event_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/{}'
neg_event_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/queried/negative/{}'
pos_files = fi.listchildren(pos_event_pattern.format(''), fi.TYPE_FILE, concat=True)
neg_files = fi.listchildren(neg_event_pattern.format(''), fi.TYPE_FILE, concat=True)


""" -------- for fasttext -------- """
label_t, label_f = ftu.label_t, ftu.label_f
ft_data_pattern = "/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/{}"
fasttext_train = ft_data_pattern.format("train")
fasttext_test = ft_data_pattern.format("test")

neg_2012_full_pattern = "/home/nfs/cdong/tw/seeding/Terrorist/queried/negative/neg_2012_full_1/{}"
neg_2012_full_files = fi.listchildren(neg_2012_full_pattern.format(''), concat=True)


def prefix_textarr(label, textarr):
    label_text_arr = list()
    for text in textarr:
        if pu.is_empty_string(text):
            continue
        label_text_arr.append('{} {}'.format(label, text.strip()))
    return label_text_arr


def make_train_test():
    p_file = ft_data_pattern.format("pos_2016.txt")
    n_bad_files = fi.listchildren(ft_data_pattern.format(''), fi.TYPE_FILE, concat=True, pattern='2016_bad')
    n_2017_files = fi.listchildren(ft_data_pattern.format(''), fi.TYPE_FILE, concat=True, pattern='2017')
    # n_2012_fulls = fi.listchildren(ft_data_pattern.format(''), fi.TYPE_FILE, concat=True, pattern='2012_full')[:12]
    n_2012_fulls = fi.listchildren(ft_data_pattern.format(''), fi.TYPE_FILE, concat=True, pattern='2012_full')
    n_2016_files = fi.listchildren(ft_data_pattern.format(''), fi.TYPE_FILE, concat=True, pattern='2016_queried')
    print(len(n_bad_files), len(n_2017_files), len(n_2012_fulls), len(n_2016_files))
    
    n_files = n_bad_files + n_2017_files + n_2012_fulls + n_2016_files
    
    p_txtarr = fu.read_lines(p_file)
    p_prefix_txtarr = prefix_textarr(label_t, p_txtarr)
    n_txtarr_blocks = [fu.read_lines(file) for file in n_files]
    n_prefix_txtarr_blocks = [prefix_textarr(label_f, txtarr) for txtarr in n_txtarr_blocks]
    
    train_test = list()
    bad = len(n_bad_files)
    bad_blocks, n_blocks = n_prefix_txtarr_blocks[:bad], n_prefix_txtarr_blocks[bad:]
    train_test.append(split_train_test(p_prefix_txtarr))
    train_test.extend([split_train_test(block) for block in n_blocks])
    print("len(train_test)", len(train_test))
    train_list, test_list = zip(*train_test)
    train_list = list(train_list) + bad_blocks
    
    train_txtarr = au.merge_array(train_list)
    test_txtarr = au.merge_array(test_list)
    fu.write_lines(fasttext_train, train_txtarr)
    fu.write_lines(fasttext_test, test_txtarr)
    print("len(train_list)", len(train_list), "len(train_txtarr)", len(train_txtarr),
          "len(test_txtarr)", len(test_txtarr))


def make_text_files():
    for idx, file in enumerate(neg_2012_full_files):
        twarr = fu.load_array(file)
        txtarr = list()
        for tw in twarr:
            text = pu.text_normalization(tw[tk.key_text])
            if pu.is_empty_string(text) or len(text) < 20:
                continue
            txtarr.append(text)
        print('len delta', len(twarr) - len(txtarr))
        path = Path(file)
        out_file_name = '_'.join([path.parent.name, path.name]).replace('json', 'txt')
        out_file = ft_data_pattern.format(out_file_name)
        print(out_file)
        fu.write_lines(out_file, txtarr)
    return
    p_twarr_blocks = map(fu.load_array, pos_files)
    p_txtarr_blocks = map(twarr2textarr, p_twarr_blocks)
    p_txtarr = au.merge_array(list(p_txtarr_blocks))
    p_out_file = ft_data_pattern.format('pos_2016.txt')
    fu.write_lines(p_out_file, p_txtarr)
    
    for f in neg_files:
        in_file = neg_event_pattern.format(f)
        out_file = ft_data_pattern.format(f.replace("json", "txt"))
        twarr = fu.load_array(in_file)
        txtarr = twarr2textarr(twarr)
        print(len(twarr), '->', len(txtarr), len(twarr) - len(txtarr))
        fu.write_lines(out_file, txtarr)


def test_train_pos_neg_portion(train_file, test_file):
    def portion_of_file(file):
        p_cnt = n_cnt = 0
        with open(file) as fp:
            lines = fp.readlines()
        for idx, line in enumerate(lines):
            label = line.split(' ', 1)[0]
            if label == label_t:
                p_cnt += 1
            elif label == label_f:
                n_cnt += 1
            else:
                print(idx)
        return p_cnt, n_cnt, len(lines)
    train_p, train_n, total_train = portion_of_file(train_file)
    print('train {}/{}={}'.format(train_p, train_n, round(train_p / train_n, 4)))
    test_p, test_n, total_test = portion_of_file(test_file)
    print('test {}/{}={}'.format(test_p, test_n, round(test_p / test_n, 4)))


""" not for anyone """


def make_neg_event_bad_text_2016():
    files = fi.listchildren("/home/nfs/cdong/tw/origin/", fi.TYPE_FILE, concat=True)
    files_blocks = mu.split_multi_format(files, 4)
    output_file = neg_event_pattern.format("neg_2016_bad_text_{}.json")
    args_list = [(block, output_file.format(idx)) for idx, block in enumerate(files_blocks)]
    res_list = mu.multi_process(extract_bad_tweets_into, args_list)
    n_num_list, tw_num_list = zip(*res_list)
    total_n, total_tw = sum(n_num_list), sum(tw_num_list)
    print(n_num_list, tw_num_list, total_n, total_tw, round(total_n / total_tw, 6))


def extract_bad_tweets_into(files, output_file):
    total_tw_num = 0
    neg_twarr = list()
    for file in files:
        twarr = fu.load_array(file)
        total_tw_num += len(twarr)
        for tw in twarr:
            text = tw[tk.key_text]
            if len(text) < 20 or not pu.has_enough_alpha(text, 0.6):
                neg_twarr.append(tw)
    fu.dump_array(output_file, neg_twarr)
    return len(neg_twarr), total_tw_num


if __name__ == '__main__':
    """ fasttext """
    tmu.check_time()
    # make_train_test()
    test_train_pos_neg_portion(fasttext_train, fasttext_test)
    tmu.check_time()
