import os
import json
from json import JSONDecodeError
import bz2file


def read_lines(file, mode='r', newline='\n'):
    """
    从文本文件中读取各行文本
    :param file: str，待读取文件路径
    :param mode: str，参数含义同open的参数列表中的mode
    :param newline: str，参数含义同open的参数列表中的newline
    :return: list，每个元素为str，即文件的一行
    """
    with open(file, mode=mode, newline=newline) as fp:
        lines = fp.readlines()
    lines = [line.strip(newline) for line in lines]
    return lines


def write_lines(file, lines, mode='w', newline='\n'):
    """
    给定字符串列表，将其写入文件
    :param file: str，待写入文件路径
    :param lines: list，每个元素为str，即文件的一行
    :param mode: str，参数含义同open的参数列表中的mode
    :param newline: str，每行的结束符
    :return:
    """
    lines = [line.strip(newline) + newline for line in lines]
    with open(file, mode=mode, encoding='utf8') as fp:
        fp.writelines(lines)


def dumps(obj, *args, **kwargs):
    return json.dumps(obj, *args, **kwargs)


def loads(s, *args, **kwargs):
    return json.loads(s, *args, **kwargs)


def dump_array(file, array, overwrite=True, sort_keys=False):
    """
    给定列表，将每个元素用json包转化为字符串，向文件写入
    :param file: str，待写入文件路径
    :param array: list，每行的元素要能够接受 json.dumps 的处理
    :param overwrite: bool，写入是否覆盖
    :param sort_keys: bool，参数含义同json.dumps参数列表中的sort_keys
    :return:
    """
    if type(array) not in {list, tuple}:
        raise TypeError("Input should be a list or a tuple.")
    lines = list()
    for item in array:
        json_str = json.dumps(item, sort_keys=sort_keys)
        lines.append(json_str + '\n')
    with open(file, 'w' if overwrite else 'a') as fp:
        fp.writelines(lines)


def load_array(file):
    """
    从文件中读取json可转换的字符串为各个对象
    :param file: str，待读取文件路径
    :return: list，每行为json.loads转换得到的对象
    """
    array = list()
    with open(file, 'r') as fp:
        for line in fp.readlines():
            array.append(json.loads(line.strip()))
    return array


def load_array_catch(file):
    array = list()
    with open(file, 'r') as fp:
        for idx, line in enumerate(fp.readlines()):
            try:
                array.append(json.loads(line.strip()))
            except JSONDecodeError as e:
                print('file:{}, line:{}, col:{}'.format(file,  e.lineno, e.colno, ))
                continue
    return array


def load_twarr_from_bz2(bz2_file):
    fp = bz2file.open(bz2_file, 'r')
    twarr = list()
    for line in fp.readlines():
        try:
            json_obj = json.loads(line.decode('utf8'))
            twarr.append(json_obj)
        except:
            print('Error when parsing ' + bz2_file + ': ' + line)
            continue
    fp.close()
    return twarr


def change_format(lxp_tweet, my_tweet):
    import copy
    from dateutil import parser
    new_tweet = copy.deepcopy(my_tweet)
    # my_tweet['id'] = lxp_tweet['_id']
    # my_tweet['event_id'] = lxp_tweet['event_id']

    # my_tweet['is_reply'] = lxp_tweet['is_reply']
    # TODO created_at 可能存在问题
    new_tweet['created_at'] = lxp_tweet['tweet']['created_at'].strftime('%a %b %d %H:%M:%S %z %Y')
    # user
    new_tweet['id'] = lxp_tweet['tweet']['id']

    # text
    new_tweet['entities']['hashtags'] = lxp_tweet['tweet']['hashtags']
    new_tweet['entities']['user_mentions'] = lxp_tweet['tweet']['mentions']
    new_tweet['orgn'] = lxp_tweet['tweet']['raw_text']
    my_tweet['lang'] = lxp_tweet['tweet']['lang']
    new_tweet['text'] = lxp_tweet['tweet']['standard_text']

    # action
    new_tweet['retweet_count'] = lxp_tweet['tweet']['action']['retweets']
    new_tweet['favorite_count'] = lxp_tweet['tweet']['action']['favorites']
    # my_tweet['in_reply_to_status_id'] = lxp_tweet['action']['retweet_id']
    # my_tweet['is_quote_status'] = lxp_tweet['action']['is_retweet'].lower()

    # geo and time
    return new_tweet


def clear(my_tweet):
    for key, value in my_tweet.items():
        if isinstance(value, dict):
            clear(value)
        else:
            my_tweet[key] = ''
    return my_tweet


def change_from_lxp_format(lxp_twarr, my_tweet=None):
    res = []
    with open('/home/nfs/yangl/merge/format.json') as f:
        my_tweet = json.loads(f.readline())
        for lxp_tw in lxp_twarr:
           res.append(change_format(lxp_tw, my_tweet))
    return res


if __name__ == '__main__':
    # file = '/home/nfs/cdong/tw/origin/2016_03_08_08.sum'
    # tw = load_array(file)[342]
    # with open('/home/nfs/yangl/merge/format.json','a') as f:
    #     print(json.dumps(clear(tw)), file=f)

    file = '/home/nfs/yangl/merge/lxp_data/lxp-test-500.json'
    lxp_twarr = load_array(file)[:3]
    with open('/home/nfs/yangl/merge/format.json') as f:
        data = f.readline()
        api_format = json.loads(data)
        res = change_from_lxp_format(lxp_twarr)
        for tw in res:
            print(tw['text'])
