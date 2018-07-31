from config.configure import getcfg
import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.db_utils as dbu

dir = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive'
if __name__ == '__main__':
    files = fi.listchildren(dir,concat=True)
    # for file in files:
    #     twarr = fu.load_array(file)
    #     for tweet in twarr:
    #         dbu.insert(tweet)
    # dbu.delete_collection('test_collection1')
    condition = {'in_reply_to_status_id':None}
    for tweet in dbu.find_by_condition(condition,limit_num=5):
        print(tweet)