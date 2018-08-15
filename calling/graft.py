from config.configure import getcfg
import pymongo
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.db_utils as dbu
import multiprocessing as mp
import datetime
import json
from bson import objectid
dir = '/home/nfs/cdong/tw/seeding/Terrorist/queried/positive/'




time_interval = 20
if __name__ == '__main__':
    new_file = '/home/nfs/yangl/merge/change_lxp_test.json'
    file = '/home/nfs/yangl/merge/lxp_test.json'
    twarr = fu.load_array(file)
    with open(new_file, 'a') as f:
        for tw in twarr:
            time = tw['tweet']['date']['$date']
            new_tw = tw
            new_tw['tweet']['date'] = time
            f.write(json.dumps(new_tw) + '\n')

