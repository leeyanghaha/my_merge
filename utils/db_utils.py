import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
mongo2_address = '54.242.147.30:27017'
client = pymongo.MongoClient(mongo2_address)
db = client.test
post = db.test_collection1


def insert(tweet):
    post.insert_one(tweet)


def insert_many(twarr):
    post.insert_many(twarr)


def find_all(kwargs = None,limit_num = None):
    res = []
    if kwargs is None:
        for tweet in post.find():
            res.append(tweet)
    else:
        for tweet in post.find({},kwargs):
            res.append(tweet)
    res = res[:limit_num if min(len(res), limit_num) == limit_num else len(res)]
    return res




def find_by_condition(query,limit_num = None):
    res = []
    try:
        for tweet in post.find(query):
            res.append(tweet)
    except:
        raise ValueError('kwargs does not exits')
    finally:
        if limit_num is not None:
            return res[:limit_num if min(len(res),limit_num) == limit_num else len(res)]


def update_many(query,new_values):
    post.update_many(query,new_values)
    print('update over.')


def delete_many(kwargs):
    post.delete_many(kwargs)
    print('delete over.')


def delete_collection(name):
    if name in db.collection_names():
        post.drop()
        print('delete collection {} over.'.format(post))
    else:
        raise ValueError('no such collection in {} '.format(db))


















