from config.configure import getcfg
import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.db_utils as dbu
import multiprocessing as mp
import datetime
# dir = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive'
# dir = '/home/nfs/yangl/merge/lxp_data'
# files = fi.listchildren(dir, concat=True)
# count = 0
# for file in files:
#     twarr = fu.load_array(file)
#     for tw in twarr:
#         count += 1
#         dbu.insert(dbu.lxp_db, dbu.lxp_test, tw)
#         print('insert {} tweets to lxp_db.lxp_test.'.format(count))

file = '/home/nfs/yangl/merge/lxp_test.json'
twarr = fu.load_array(file)
count = 0
import json
for tw in twarr:
    count += 1
    dbu.insert(dbu.lxp_db, dbu.lxp_test, tw)
    print('insert {} tweets to lxp_db.lxp_test.'.format(count))