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


def change_tweet_format():
    pass