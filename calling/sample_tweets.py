from config.configure import getcfg
import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.db_utils as dbu
import multiprocessing as mp
import datetime
dir = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive'
files = fi.listchildren(dir,concat=True)
for file in files:
    twarr = fu.load_array(file)
    for tw in twarr:
        dbu.insert(dbu.tweet_db,dbu.positive_16,tw)
