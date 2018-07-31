from utils.utils_loader import *


def tweet_num_multi(file_list, func):
    file_list_blocks = mu.split_multi_format(file_list, 20)
    args_list = [(file_list_block, ) for file_list_block in file_list_blocks]
    res_list = mu.multi_process(func, args_list)
    return sum([sum(res) for res in res_list])


def tweet_num(file_list):
    count = list()
    for idx, file in enumerate(file_list):
        twarr_len = len(fu.read_lines(file))
        if idx % int(len(file_list) / 10) == 0:
            print(int(idx / len(file_list)) * 10, '%')
        count.append(twarr_len)
    return count


def tweet_num_bz2(file_list):
    import bz2file
    count = list()
    for file in file_list:
        fp = bz2file.open(file, 'r')
        count.append(len(fp.readlines()))
        fp.close()
    return count


if __name__ == '__main__':
    # disaster2016 = "/home/nfs/cdong/tw/seeding/NaturalDisaster/queried/NaturalDisaster.sum"
    # array = fu.load_array(disaster2016)
    # print(len(array))
    # exit()
    
    base2012_neg = "/home/nfs/yying/sync/data{}/tweets"
    file_list = au.merge_array([fi.listchildren(base2012_neg.format(idx), concat=True) for idx in range(1, 7)])
    print(len(file_list))
    exit()
    # base2012_pos = "/home/nfs/yangl/event_detection/testdata/event2012/relevant_tweets"
    # base2016_terror = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive"
    all2016 = "/home/nfs/cdong/tw/origin/"
    file_list = fi.listchildren(all2016, concat=True)
    print(len(file_list))
    terror_num = tweet_num_multi(file_list, tweet_num)
    print(tweet_num_multi(file_list, tweet_num))
