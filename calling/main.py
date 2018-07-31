import calling.back_filter as bflt
import calling.back_cluster as bclu
import calling.back_extractor as bext

import classifying.classifier_class_info as cci
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.timer_utils as tmu
import utils.email_utils as emu
import time
import warnings


event_type = cci.event_t
# filter & classify
flt_pool_size = 12
ne_threshold = 0.4
clf_threshold = 0.13
# cluster
check_every_sec = 180
# check_every_sec =50
max_window_size = 10
full_interval = 8
alpha = 30
beta = 0.01
# extractorS
ext_pool_size = 10


def twarr2filter(twarr):
    """
    将同一组推特列表拆分为多个 batch，传入过滤&分类模块中
    :param twarr: list，推特列表，每个元素为dict(即json格式字符串转换而来的对象)
    :return:
    """
    tw_batch = mu.split_multi_format(twarr, flt_pool_size)
    bflt.input_twarr_batch(tw_batch)
# --del--
#     fu.dump_array(history_twarr_len_file, [len(twarr)], overwrite=False)
# output_base = './last_4000-ne_thres={}-clf_thres={}-hold_batch={}-batch_size={}-alpha={}-beta={}'.\
#     format(ne_threshold, clf_threshold, hold_batch_num, batch_size, alpha, beta)
# fi.mkdir(output_base, remove_previous=True)
# filtered_twarr_file = fi.join(output_base, "filtered_twarr.json")
# history_twarr_len_file = fi.join(output_base, "history_twarr_len.txt")
# history_filter_len_file = fi.join(output_base, "history_filter_len.txt")
# --del--


def filter2cluster(remain_workload=None):
    """
    从过滤&分类模块读取其返回的推特列表，合并后输入聚类模块
    :param remain_workload: int/None，控制从过滤&分类模块读取结果的行为
            若为None，则调用 bflt.try_get_unread_batch_output 获取返回结果；
            否则以其为参数调用 bflt.wait_get_unread_batch_output 获取返回结果
    :return:
    """
    if remain_workload is None:
        batches_of_batches = bflt.try_get_unread_batch_output()
    else:
        batches_of_batches = bflt.wait_get_unread_batch_output(remain_workload)
    if not batches_of_batches:
        return
    filtered_batches = au.merge_array(batches_of_batches)
    filtered_twarr = au.merge_array(filtered_batches)
    bclu.input_twarr(filtered_twarr)
# --del--
#     fu.dump_array(filtered_twarr_file, filtered_twarr, overwrite=False)
#     fu.dump_array(history_filter_len_file, [len(filtered_twarr)], overwrite=False)
# --del--


def cluster2extractor():
    """
    从聚类模块读取返回结果，输入聚类信息提取模块
    :return:
    """
    cluid_twarr_list = bclu.try_get_cluid_twarr_list()
    bext.input_cluid_twarr_list(cluid_twarr_list)


def end_it():
    """
    等待各模块执行完剩余的过程
    :return:
    """
    print('reaching end')
    filter2cluster(0)
    bclu.execute_cluster()
    cluid_twarr_list = bclu.wait_get_cluid_twarr_list()
    bext.input_cluid_twarr_list(cluid_twarr_list)
    bext.end_pool()


def main():
    """
    启动各进程（池），遍历 _sub_files 中的文件名，逐个读取文件内容，
    每读取一个文件，输入过滤&聚类模块，并尝试从分类器读取返回结果后输入聚类模块；
    每过指定时间，向聚类模块发送聚类指令；
    随后尝试从聚类模块读取返回结果，并输入聚类信息提取模块
    :return:
    """
    tmu.check_time('qwertyui')
    tmu.check_time('main line 116', print_func=None)
    
    bflt.start_pool(flt_pool_size, ne_threshold, clf_threshold, event_type)
    bclu.start_pool(max_window_size, full_interval, alpha, beta)
    bext.start_pool(ext_pool_size, event_type)

    alarm = tmu.Alarm()
    _sub_files = fi.listchildren("/home/nfs/cdong/tw/origin/", fi.TYPE_FILE, concat=True)[-4000:]
    # positive_twarr = fu.load_array('/home/nfs/yangl/dc/calling/filtered_twarr.json')[:5000]
    # _sub_files = fi.listchildren("/home/nfs/yangl/dc/input", fi.TYPE_FILE, concat=True)
    for _idx, _file in enumerate(_sub_files):
        _twarr = fu.load_array(_file)
        if (_idx + 1) % 1000 == 0:
            dt = tmu.check_time('main line 116', print_func=None)
            emu.send_email('file {}/{}'.format(_idx + 1, len(_sub_files)), '{}s from last 1000 file'.format(dt))
        if _idx > 0 and _idx % 10 == 0:
            print("main: {} th twarr to filter, len: {}".format(_idx, len(_twarr)))
        twarr2filter(_twarr)
        filter2cluster()
        if alarm.is_time_elapse_bigger_than(check_every_sec):
            alarm.initialize_timestamp()
            filter2cluster(5)
            bclu.execute_cluster()
        cluster2extractor()

    time.sleep(120)
    end_it()
    tmu.check_time('qwertyui')


if __name__ == '__main__':
    tmu.check_time()
    main()
    tmu.check_time(print_func=lambda dt: print("total time elapsed {}s".format(dt)))
    # 100  hours: 3914407  -> 93128  in 5695 s  (1.58 h),  threshold 0.1, thread 5
    # 1000 hours: 40110901 -> 988585 in 60769 s (16.88 h), threshold 0.1, thread 5
    # last 1000 hours: 37232413 -> 48193  in 7818 s (2.17 h), thread 15
    
    # 17 files/min, 4000/17=235 min=4 h,
