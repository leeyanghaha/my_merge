import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import datetime
from bson import objectid
import time
from multiprocessing import Process,Queue


# mongo2_address = '54.242.147.30:27017'
# client = pymongo.MongoClient(mongo2_address)
#
# # db = client.test
# tweet_db = client.tweet
# positive_16 = 'positive_16'
# clu_db = client.cluster
# lxp_db = client.lxp_data
# lxp_test = 'test'

mongo3_address = '54.161.160.206:27017'
client = pymongo.MongoClient(mongo3_address)
nd_db = client.natural_disaster
nd = 'dataset'

def insert(db, collection, tweet):
    db[collection].insert_one(tweet)


def insert_many(db, collection, twarr):
    db[collection].insert_many(twarr)


def find_one(db, collection):
    return db[collection].find_one()


def find_all(db, collection, kwargs=None, limit_num=None):
    res = []
    if kwargs is None:
        for tweet in db[collection].find():
            res.append(tweet)
    else:
        for tweet in db[collection].find({}, kwargs):
            res.append(tweet)
    res = res[:limit_num if min(len(res), limit_num) == limit_num else len(res)]
    return res


def find_by_condition(db, collection, query):
    res = []
    try:
        for tweet in db[collection].find(query):
            res.append(tweet)
    except:
        raise ValueError('kwargs does not exits')
    # finally:
    #     if limit_num is not None:
    #         return res[:limit_num if min(len(res), limit_num) == limit_num else len(res)]


def update_many(db, collection, query, new_values):
    db[collection].update_many(query, new_values)
    print('update over.')


def delete_many(db, collection, kwargs):
    db[collection].delete_many(kwargs)
    print('delete over.')


def delete_collection(db, collection, name):
    if name in db.collection_names():
        db[collection].drop()
        print('delete collection {} over.'.format(db[collection]))
    else:
        raise ValueError('no such collection in {} '.format(db))


def read_after_last_check(db, collection, last_check_time, limit=None):
    res = []
    for idx, tw in enumerate(db[collection].find({'tweet.created_at': {'$lt': last_check_time}})):
        if limit is not None and idx > limit:
            break
        res.append(tw)
    return res


def start_read(outq, time_interval, last_check_time):
    while(True):
        time.sleep(time_interval)
        # print(last_check_time)
        interval_tweet, new_time = read_after_last_check(last_check_time)
        outq.put(interval_tweet)
        # print(len(interval_tweet))
        last_check_time = new_time


def read_from_db(time_interval, last_check_time):
    q = Queue()
    pw = Process(target=start_read, args=(q, time_interval, last_check_time))
    pw.start()
    time.sleep(time_interval)
    return q.get()


if __name__ == '__main__':
     tweet = (nd_db, nd)
     import utils.function_utils as fu
     import json
     start_time = find_one(nd_db, nd)['tweet']['created_at']
     my = find_one(nd_db, nd)['tweet']['created_at'].strftime('%a %b %d %H:%M:%S %z %Y')
     print(start_time, my)
     # from dateutil import parser
     # my = parser.parse(my).strftime('%a %b %d %H:%M:%S %z %Y')
     # print(my)






