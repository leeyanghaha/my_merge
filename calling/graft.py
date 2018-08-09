from config.configure import getcfg
import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.db_utils as dbu
import multiprocessing as mp
import datetime
from bson import objectid
dir = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/'




time_interval = 20
if __name__ == '__main__':
    time = datetime.datetime.strptime('2010-03-08 06:01:28.0000Z',"%Y-%m-%d %H:%M:%S.0000Z")
    print(time.timestamp())
    # print(change_format(lxp_twarr,my_tweet))


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
    # file = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/2016-01-01_shoot_Aviv.json'
    # # files = fi.listchildren(dir,concat=True)
    # # for file in files:
    # #     twarr = fu.load_array(file)
    # tweet = {}
    # new_tweet ={}
    # twarr = fu.load_array(file)
    # print(twarr[1])
