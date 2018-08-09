import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import datetime
from bson import objectid
import time
from multiprocessing import Process,Queue


mongo2_address = '54.242.147.30:27017'
client = pymongo.MongoClient(mongo2_address)

# db = client.test
tweet_db = client.tweet
positive_16 = 'positive_16'
clu_db = client.cluster
lxp_dataset = 'dataset'


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
        for tweet in db[collection].find({},kwargs):
            res.append(tweet)
    res = res[:limit_num if min(len(res), limit_num) == limit_num else len(res)]
    return res


def find_by_condition(db, collection, query, limit_num=None):
    res = []
    try:
        for tweet in db[collection].find(query):
            res.append(tweet)
    except:
        raise ValueError('kwargs does not exits')
    finally:
        if limit_num is not None:
            return res[:limit_num if min(len(res), limit_num) == limit_num else len(res)]


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


def read_after_last_check(db, collection, last_check_time):
    last_time = datetime.datetime.strptime(last_check_time, "%Y-%m-%d %H:%M:%S").timestamp()
    new = datetime.datetime.fromtimestamp(last_time - 8*60*60)
    gen_time = datetime.datetime(new.year,new.month,new.day,new.hour,new.minute,new.second)
    obj_id = objectid.ObjectId.from_datetime(gen_time)
    # sec = int(str(obj_id)[:8],16)
    # print(datetime.datetime.fromtimestamp(sec))
    cursor = db[collection].find({'_id': {'$gt': obj_id}})
    res = []
    for tw in cursor:
        res.append(tw)
    return res, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


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
    print(find_one(tweet_db, lxp_dataset))
    # last_check_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print(read_from_db(10,last_check_time))

    # while(True):
    #     time.sleep(10)
    #     print(last_check_time)
    #     res_len, new_time = read_after_last_check(last_check_time)
    #     print(res_len)
    #     print(new_time)
    #     read_after_last_check(new_time)
    #     last_check_time = new_time






