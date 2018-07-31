from utils.utils_loader import *
from preprocess.filter.yying_non_event_filter import EffectCheck


base = "/home/nfs/cdong/tw/origin/"
files = fi.listchildren(base, fi.TYPE_FILE, concat=True)
file = files[571]
my_filter = EffectCheck()
twarr = fu.load_array(file)
print(len(twarr), '->', len(my_filter.filter(twarr, 0.4)))
