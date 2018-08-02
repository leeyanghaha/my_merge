from config.configure import getcfg
import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.db_utils as dbu
import multiprocessing as mp
import datetime
from bson import objectid
dir = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/'
def read(time_interval):
    last_check_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return dbu.read_from_db(time_interval,last_check_time)


def chane_format(lxp_tweet):
    my_tweet = {}
    my_tweet['id'] = lxp_tweet['id']
    my_tweet['conversation_id'] = lxp_tweet['conversation_id']
    my_tweet['is_reply'] = lxp_tweet['is_reply']
    my_tweet['created_at'] = lxp_tweet['created_at']

    #user
    my_tweet['user']['id'] = lxp_tweet['user']['user_id']

    # text
    my_tweet['entities']['hashtags'] = lxp_tweet['hashtags']
    my_tweet['entities']['user_mentions'] = lxp_tweet['mentions']
    my_tweet['orgn'] = lxp_tweet['raw_text']
    my_tweet['lang'] = lxp_tweet['lang']
    my_tweet['text'] = lxp_tweet['standard_text']

    #action
    my_tweet['retweet_count'] = lxp_tweet['action']['retweets']
    my_tweet['favorite_count'] = lxp_tweet['action']['favorites']
    my_tweet['in_reply_to_status_id'] = lxp_tweet['action']['retweet_id']
    my_tweet['is_quote_status'] = lxp_tweet['action']['is_retweet'].lower()

    #geo and time
    print(my_tweet)
time_interval = 20
if __name__ == '__main__':

    # gen_time = datetime.datetime(2018, 4, 10)
    # dummy_id = objectid.ObjectId.from_datetime(gen_time)
    # read_iter_num = 10
    # for i in range(read_iter_num):
    #     read(20)
    # read_process = mp.Process(target=read,args=('leeyang',))
    # read_process.start()
    # read_process.join()
    # dbu.delete_collection('positive_16')
    # condition = {'in_reply_to_status_id':None}
    # for tweet in dbu.find_by_condition(condition,limit_num=5):
    #     print(tweet)
    file = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-01-01_shoot_Aviv.json'
    # files = fi.listchildren(dir,concat=True)
    # for file in files:
    #     twarr = fu.load_array(file)
    tweet = {}
    new_tweet ={}
    twarr = fu.load_array(file)
    print(twarr[1])



