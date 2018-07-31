from classifying.terror.data_maker import au, fi, fu, ftu, tk, prefix_textarr, split_train_test, test_train_pos_neg_portion


label_t, label_f = ftu.label_t, ftu.label_f
value_t, value_f = ftu.value_t, ftu.value_f

neg_pattern = "/home/nfs/cdong/tw/seeding/NEGATIVE/{}"
k_data_pattern = "/home/nfs/cdong/tw/seeding/Korea/data/{}"
train_file = k_data_pattern.format("train")
test_file = k_data_pattern.format("test")
pos_twarr_file = "/home/nfs/cdong/tw/seeding/Korea/korea.json"
pos_text_file = k_data_pattern.format("k_pos.txt")


def make_train_test():
    neg_path = neg_pattern.format('')
    # n_2012_fulls = fi.listchildren(neg_path, fi.TYPE_FILE, concat=True, pattern='2012_f')[:3]
    n_2012_fulls = fi.listchildren(neg_path, fi.TYPE_FILE, concat=True, pattern='2012_f')
    n_2016_bad_s = fi.listchildren(neg_path, fi.TYPE_FILE, concat=True, pattern='2016_b')
    n_2016_files = fi.listchildren(neg_path, fi.TYPE_FILE, concat=True, pattern='2016_q')
    n_2017_files = fi.listchildren(neg_path, fi.TYPE_FILE, concat=True, pattern='2017')
    print(len(n_2012_fulls), len(n_2016_bad_s), len(n_2016_files), len(n_2017_files))
    n_files = n_2016_bad_s + n_2012_fulls + n_2016_files + n_2017_files
    
    p_txtarr = fu.read_lines(pos_text_file)
    p_prefix_txtarr = prefix_textarr(label_t, p_txtarr)
    n_txtarr_blocks = [fu.read_lines(file) for file in n_files]
    n_prefix_txtarr_blocks = [prefix_textarr(label_f, txtarr) for txtarr in n_txtarr_blocks]
    
    train_test = list()
    bad = len(n_2016_bad_s)
    bad_blocks, n_blocks = n_prefix_txtarr_blocks[:bad], n_prefix_txtarr_blocks[bad:]
    train_test.append(split_train_test(p_prefix_txtarr))
    train_test.extend([split_train_test(block) for block in n_blocks])
    print("len(train_test)", len(train_test))
    train_list, test_list = zip(*train_test)
    train_list = list(train_list) + bad_blocks
    
    train_txtarr = au.merge_array(train_list)
    test_txtarr = au.merge_array(test_list)
    fu.write_lines(train_file, train_txtarr)
    fu.write_lines(test_file, test_txtarr)
    print("len(train_list)", len(train_list), "len(train_txtarr)", len(train_txtarr),
          "len(test_txtarr)", len(test_txtarr))


def make_positive():
    twarr_blocks = fu.load_array(pos_twarr_file)
    twarr = au.merge_array(twarr_blocks)
    print(type(twarr[0]), len(twarr))
    txtarr = [tw[tk.key_text] for tw in twarr]
    fu.write_lines(pos_text_file, txtarr)


if __name__ == '__main__':
    make_positive()
    make_train_test()
    test_train_pos_neg_portion(train_file, test_file)
