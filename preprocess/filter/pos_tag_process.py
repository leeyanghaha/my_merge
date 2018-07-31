from config.configure import getcfg


# TODO 先用绝对路径标识文件名，我这里在配置文件中写成绝对路径了
afinn_file = getcfg().afinn_file


ark_marks = {'N': 0, 'O': 1, '^': 2, 'S': 3, 'Z': 4, 'V': 5, 'A': 6, 'R': 7, '!': 8, 'D': 9, 'P': 10,
             '&': 11, 'T': 12, 'X': 13, '$': 14, ',': 15, 'G': 16, 'L': 17, 'M': 18, 'Y': 19}
affinn_dict = {}


with open(afinn_file, 'r') as fp:
    for line in fp.readlines():
        l1 = line.split('\t')
        affinn_dict[l1[0].strip()] = int(l1[1].strip())


def count_ark_mark(ark_list):
    mark_list = [0] * 20
    for pos_mark in ark_list:
        if pos_mark[1] in ark_marks:
            mark_list[ark_marks[pos_mark[1]]] += 1
    return mark_list


def count_sentiment(text):
    words = text.split()
    sentiment = 0
    for word in words:
        if word in affinn_dict:
            sentiment += affinn_dict[word]
    return sentiment
