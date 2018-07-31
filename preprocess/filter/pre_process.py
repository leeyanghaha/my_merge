import numpy
import re
import os
import json
import sys
from os import listdir
from os.path import isfile, join
from enum import Enum
from multiprocessing import Pool
import timeit
import utils.pattern_utils as pu

totalPatternsMatch = 0
totalRatioCount = 0
reason = None
import gensim


# inputpath = '../chatModel/chat_'
# dictionary = gensim.corpora.Dictionary.load_from_text(inputpath + 'wordids.txt')
# corpus = gensim.corpora.MmCorpus(inputpath + 'corpus.mm')
# lda = gensim.models.ldamodel.LdaModel.load(inputpath + 'lda.lda')
# labeled_topic = dict()
# fp = open('../labeled_chat_topics.txt', 'r')
# for line in fp.readlines():
#     two_num = line.split()
#     labeled_topic[int(two_num[0])] = int(two_num[1])
#
# one = 0
# two = 0
# three = 0
# for topic in labeled_topic:
#     if labeled_topic[topic] == 1:
#         one += 1
#     elif labeled_topic[topic] == 2:
#         two += 1
#     elif labeled_topic[topic] == 3:
#         three += 1
#     else:
#         print(labeled_topic[topic])
#
# print(one, two, three)


# 2 susupicious
class res(Enum):
    NORMAL = 1
    SUSPICIOUS = 2
    NOISY = 3


def mergeResults(f1, f2, f3):
    res = f1.value + f2.value + f3.value - 3
    if res >= 2:
        return True
    else:
        return False


# if run into this func flag will be false
# return true if entity indicate the tweet should be filtered
def entityClassify(entity):
    global reason
    urls = entity['urls']
    hashTags = entity['hashtags']
    if urls:
        urlPatterns = list()
        urlPatterns.append(re.compile(r'.*?fllwrs.*?'))
        # urlPatterns.append(re.compile(r'.*?fb\.me.*?'))
        urlPatterns.append(re.compile(r'.*?amazon.*?'))
        urlPatterns.append(re.compile(r'.*?ebay.*?'))
        urlPatterns.append(re.compile(r'.*?amzn.*?'))
        try:
            for urlEntry in urls:
                url = urlEntry['expanded_url']
                for pattern in urlPatterns:
                    match = pattern.fullmatch(url)
                    if match:
                        reason = "url pattern " + str(pattern)
                        return res.NOISY
        except:
            pass
    if hashTags:
        noisePatterns = list()
        noisePatterns.append(re.compile(r'.*?sale.*?', re.I))
        noisePatterns.append(re.compile(r'.*?game.*?', re.I))
        susPatterns = list()
        susPatterns.append(re.compile(r'.*?porn.*?', re.I))
        susPatterns.append(re.compile(r'.*?sex.*?', re.I))
        for hashTag in hashTags:
            text = hashTag['text']
            for pattern in noisePatterns:
                match = pattern.fullmatch(text)
                if match:
                    reason = "hashtag pattern " + str(pattern)
                    return res.NOISY
            for pattern in susPatterns:
                match = pattern.fullmatch(text)
                if match:
                    return res.SUSPICIOUS

    return res.NORMAL


def chatFilter(orgn):
    global reason
    c_f = res.NORMAL
    corpus = pu.text_normalization(orgn).lower().split()
    vec_bow = dictionary.doc2bow(corpus)
    vec_lda = lda[vec_bow]
    maxSim = 0.35
    topicNum = -1
    for sim in vec_lda:
        if sim[1] > maxSim:
            topicNum = sim[0]
            maxSim = sim[1]
    #         print(topicNum)
    if topicNum != -1:
        #         print(nd_text[count])
        #         print (nd_corpus[count])
        if labeled_topic[topicNum] == 3:
            c_f = res.NOISY
            reason = "topic: " + str(topicNum)
        elif labeled_topic[topicNum] == 2:
            c_f = res.SUSPICIOUS
        else:
            c_f = res.NORMAL
        pass
    return c_f


'''
return a result list 
0 for spam
1 for not spam 
'''


def filterArray(array):
    resultList = list()
    patterns = list()
    patterns.append(re.compile(r'.*?I posted a new photo to Facebook.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?poste?d?\sa\sphoto.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?Get Weather Updates from The Weather Channel.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?New Song.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?My week on Twitter.*?', re.IGNORECASE))
    patterns.append(re.compile(r'^(?=.*\btemp)(?=.*\bhum).*$', re.I))
    patterns.append(re.compile(r'.*?[0-9]?[0-9]%\s?off.*?', re.I))
    patterns.append(re.compile(r'.*?happy\s?new\s?year.*?', re.I))

    i = 0
    ratio_count = 0
    entity_count = 0
    for jsonObject in array:
        # spam ad chat
        s_f = res.NORMAL
        a_f = res.NORMAL
        c_f = res.NORMAL
        text = jsonObject['orgn']
        if jsonObject['user']['friends_count'] != 0:
            follower_ratio = jsonObject['user']['followers_count'] / jsonObject['user']['friends_count']
        elif jsonObject['user']['followers_count'] != 0:
            follower_ratio = 1
        else:
            follower_ratio = 0
            if jsonObject['user']['friends_count'] == 0 and jsonObject['user']['followers_count'] == 0:
                follower_ratio = 0.09
        if follower_ratio < 0.01:
            # print("ratio<0.01")
            # print(text)
            s_f = res.NOISY
            reason = "follower_ratio less than 0.01"
            ratio_count += 1
        elif follower_ratio < 0.1:
            s_f = res.SUSPICIOUS
        for pattern in patterns:
            match = pattern.fullmatch(text)
            if match:
                s_f = res.NOISY
                i = i + 1
                # print('pattern ' + pattern.pattern + ' detected \nnumber' +str(i) + text + ' is now out of dataset')
            if s_f == res.NOISY:
                break
        if a_f == res.NOISY:
            pass
        else:
            a_f = entityClassify(jsonObject['entities'])
            if a_f == res.NOISY:
                entity_count += 1
        if s_f == res.NOISY:
            pass
        elif jsonObject['user']['description']:
            pass
        else:
            s_f = res.SUSPICIOUS
        # c_f = chatFilter(text)
        if mergeResults(s_f, a_f, c_f):
            # 0 for neg & spam
            resultList.append(0)
        else:
            # 1 for pos
            resultList.append(1)

    return resultList


def filterStr(filepath, outputPath):
    global totalPatternsMatch, totalRatioCount, reason
    reason = None
    s_noi = 0
    s_sus = 0
    a_noi = 0
    a_sus = 0
    c_noi = 0
    c_sus = 0
    array = list()
    filetereddata = list()
    try:
        fp = open(filepath, 'r')
        for line in fp.readlines():
            try:
                array.append(json.loads(line.strip()))
            except:
                print(filepath)
                print(line)
    except:
        print("error while read", filepath)
        return
    patterns = list()
    patterns.append(re.compile(r'.*?I posted a new photo to Facebook.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?poste?d?\sa\sphoto.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?Get Weather Updates from The Weather Channel.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?New Song.*?', re.IGNORECASE))
    patterns.append(re.compile(r'.*?My week on Twitter.*?', re.IGNORECASE))
    patterns.append(re.compile(r'^(?=.*\btemp)(?=.*\bhum).*$', re.I))
    patterns.append(re.compile(r'.*?[0-9]?[0-9]%\s?off.*?', re.I))
    patterns.append(re.compile(r'.*?happy\s?new\s?year.*?', re.I))

    i = 0
    ratio_count = 0
    entity_count = 0
    for jsonObject in array:
        # spam ad chat
        s_f = res.NORMAL
        a_f = res.NORMAL
        c_f = res.NORMAL
        text = jsonObject['orgn']
        if jsonObject['user']['friends_count'] != 0:
            follower_ratio = jsonObject['user']['followers_count'] / jsonObject['user']['friends_count']
        elif jsonObject['user']['followers_count'] != 0:
            follower_ratio = 1
        else:
            follower_ratio = 0
            if jsonObject['user']['friends_count'] == 0 and jsonObject['user']['followers_count'] == 0:
                follower_ratio = 0.09
        if follower_ratio < 0.01:
            # print("ratio<0.01")
            # print(text)
            s_f = res.NOISY
            reason = "follower_ratio less than 0.01"
            ratio_count += 1
        elif follower_ratio < 0.1:
            s_f = res.SUSPICIOUS
        for pattern in patterns:
            match = pattern.fullmatch(text)
            if match:
                s_f = res.NOISY
                i = i + 1
                # print('pattern ' + pattern.pattern + ' detected \nnumber' +str(i) + text + ' is now out of dataset')
            if s_f == res.NOISY:
                break
        if a_f == res.NOISY:
            pass
        else:
            a_f = entityClassify(jsonObject['entities'])
            if a_f == res.NOISY:
                entity_count += 1
        if s_f == res.NOISY:
            pass
        elif jsonObject['user']['description']:
            pass
        else:
            s_f = res.SUSPICIOUS
        # c_f = chatFilter(text)
        if mergeResults(s_f, a_f, c_f):
            s_v = s_f.value
            a_v = a_f.value
            c_v = c_f.value
            if s_v == 3:
                s_noi += 1
            elif a_v == 3:
                a_noi += 1
            elif c_v == 3:
                c_noi += 1
            elif s_v == 2:
                s_sus += 1
                if a_v == 2:
                    a_sus += 1
                if c_v == 2:
                    c_sus += 1
            # print("'''''''''")
            # print(jsonObject['orgn'])
            # print(reason)
            pass
        else:
            filetereddata.append(jsonObject)
    # print(filepath + "  " + str(i) + " texts is now out of dataset")
    # print("ratio exclude count " + str(ratio_count))
    # print("entity count",str(entity_count))
    # totalPatternsMatch += i
    # totalRatioCount += ratio_count
    # for i in range(5):
    #     print("*******************")
    print(filepath)
    print("total origin tweets :", len(array), "  tweets after filtered: ", len(filetereddata), "filterRate",
          (len(array) - len(filetereddata)) / len(array) if len(array) != 0 else 'divide by zero')
    print("s noise", s_noi, "s sus", s_sus, "a noise", a_noi, "a sus", a_sus, "c noise", c_noi, "c sus", c_sus)
    with open(outputPath, mode='wt', encoding='utf-8') as myfile:
        for element in filetereddata:
            myfile.write(json.dumps(element) + '\n')
    myfile.close
