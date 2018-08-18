import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import datetime
from bson.objectid import ObjectId
import time
from multiprocessing import Process,Queue


# mongo2_address = '54.242.147.30:27017'
# client = pymongo.MongoClient(mongo2_address)
#
# # db = client.test
# tweet_db = client.tweet
# positive_16 = 'positive_16'
# clu_db = client.cluster


mongo3_address = '54.161.160.206:27017'
client = pymongo.MongoClient(mongo3_address)
nd_db = client.natural_disaster
format_db = client.format_db
nd = 'dataset'
api_format = 'api_format'
lxp_db = client.lxp_data
lxp_test = 'test'


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


# def read_after_last_check(db, collection, last_check_time, limit=None):
#     res = []
#     for idx, tw in enumerate(db[collection].find({'tweet.created_at': {'$lt': last_check_time}})):
#         if limit is not None and idx > limit:
#             break
#         res.append(tw)
#     return res


def read_after_last_check(db, collection, last_check_time, limit=None):
    res = []
    beijing_time = datetime.datetime.fromtimestamp(last_check_time.timestamp() - 8*3600)
    dummy_id = ObjectId.from_datetime(beijing_time)
    for idx, tw in enumerate(db[collection].find({'_id': {'$gt': dummy_id}})):
        if limit is not None and idx > limit:
            break
        res.append(tw)
    return res, datetime.datetime.now()


if __name__ == '__main__':
    last_check_time = datetime.datetime.now()
    while(True):
        time.sleep(10)
        twarr, now = read_after_last_check(lxp_db, lxp_test, last_check_time)
        print('{} tweets read from db'.format(len(twarr)))
        last_check_time = now





