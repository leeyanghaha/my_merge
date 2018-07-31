from clustering.gsdpmm.gsdpmm_stream_ifd_dynamic import GSDPMMStreamIFDDynamic as GSDPMM
from utils.multiprocess_utils import DaemonProcess
import multiprocessing as mp


class ClusterDaemonProcess(DaemonProcess):
    def __init__(self, pidx=0):
        DaemonProcess.__init__(self, pidx)
        self.outq2 = mp.Queue()
    
    def start(self, func):
        self.process = mp.Process(target=func, args=(self.inq, self.outq, self.outq2))
        self.process.daemon = True
        self.process.start()
        # print('ClusterDaemonProcess', self.process.pid)


cluster_daemon = ClusterDaemonProcess()
END_PROCESS = -1
SET_PARAMS = 0
INPUT_TWARR = 1
EXE_CLUSTER = 2
OUTPUT_LIST = 3
OUTPUT_STATE = 4


def cluster_main(inq, outq, outq2):
    """
    聚类模块子进程的主函数，从inq读入推特列表，用outq2返回聚类器的输出；
    总是首先尝试读取一条指令，判断指令内容后读取其附带的参数列表；
    目前指令包括：输入超参（SET_PARAMS）、输入推特列表（INPUT_TWARR）以及进行聚类（EXE_CLUSTER）；
    每次进行聚类都会伴随一次向outq2写入输出的操作
    :param inq: mp.Queue，主进程向子进程的输入队列
    :param outq: mp.Queue，子进程向主进程的输出队列
    :param outq2: mp.Queue，子进程向主进程的输出队列
    :return:
    """
    clusterer = BackCluster()
    while True:
        command = inq.get()
        if command == SET_PARAMS:
            params = inq.get()
            clusterer.set_parameters(*params)
        elif command == INPUT_TWARR:
            tw_batch = inq.get()
            clusterer.input_twarr(tw_batch)
        elif command == EXE_CLUSTER:
            iter_num = inq.get()
            clusterer.exec_cluster(iter_num)
            clusterer.output(outq2)
        elif command == END_PROCESS:
            print('ending cluster_main')
            outq.put('ending cluster_main')
            return


def start_pool(*args):
    """
    启动聚类模块的子进程，并输入相应参数
    :param args: list，位置参数列表
    """
    cluster_daemon.start(cluster_main)
    cluster_daemon.set_input(SET_PARAMS)
    cluster_daemon.set_input(args)


def end_pool():
    """
    发送终止进程的指令，子进程一旦处理完该指令之前所有的输入指令，则在收到该指令后退出
    :return:
    """
    cluster_daemon.set_input(END_PROCESS)
    cluster_daemon.outq.get()


def input_twarr(twarr):
    """
    向子进程输入推特列表，并且不等待返回结果
    :param twarr: list，推特列表
    """
    if twarr:
        cluster_daemon.set_input(INPUT_TWARR)
        cluster_daemon.set_input(twarr)


def execute_cluster(iter_num=25):
    """
    向子进程发送执行聚类的指令
    :param iter_num: int，当子进程判断到达重新进行聚类的条件时，以多大的迭代次数进行聚类
    :return:
    """
    cluster_daemon.set_input(EXE_CLUSTER)
    cluster_daemon.set_input(iter_num)


def wait_get_cluid_twarr_list():
    """
    阻塞地等待子进程在上一次执行聚类操作后输出的聚类结果，并将其返回
    :return: 子进程的输出（见clustering函数中outq的输入）
    """
    return cluster_daemon.outq2.get()


def try_get_cluid_twarr_list():
    """
    循环地判断子进程是否产生了输出，是则调用 wait_get_cluid_twarr_list，直至获得最新的结果
    :return: list/None，若子进程有输出则返回值为列表，见BackCluster.output()；否则返回None
    """
    cluid_twarr_list = None
    while cluster_daemon.outq2.qsize() > 0:
        cluid_twarr_list = wait_get_cluid_twarr_list()
    return cluid_twarr_list


class Countdown:
    def __init__(self, ref_val):
        self.c = self.ref_val = ref_val
    
    def has_reach_ref(self):
        return self.c == self.ref_val
    
    def count(self):
        self.c += 1
        if self.c > self.ref_val:
            self.c = 0


class BackCluster:
    G_CLASS = GSDPMM
    
    def __init__(self):
        """
        聚类器包装对象，负责记录窗口、根据记录值调用不同聚类过程、返回聚类结果
        """
        self.max_window_size = self.countdown = None
        self.read_batch_num = self.hist_len = 0
        self.g = self.G_CLASS()
        self.window_record = list()
    
    def set_parameters(self, max_window_size, full_interval, alpha, beta):
        """
        设置参数，部分参数用于调整窗口大小以及batch大小，另一部分则用于调整概率表达式的先验参数
        :param max_window_size: int，记录窗的窗口大小；相邻记录点之间的时间间隔由外部调用者确定
        :param full_interval: int，进行完全聚类的记录点间隔数
        :param alpha: float，概率参数1
        :param beta: float，概率参数2
        :return:
        """
        self.max_window_size = max_window_size
        self.countdown = Countdown(full_interval)
        self.g.set_hyperparams(alpha, beta)
    
    def exec_cluster(self, iter_num):
        """
        self.window_record 记录了每次被要求执行聚类时，聚类器中推特列表的长度；
        在 self.window_record 中的记录个数达到 self.max_window_size 之前，
        调用该方法仅会在 self.window_record 中进行一次记录；
        达到 self.max_window_size 之后，每次调用该方法，除记录长度之外，
        还判断 self.countdown 的计数值是否达到阈值，若达到则进行一次完全聚类；
        此后，self.countdown 计一次数，并调用 self.pop_oldest_record
        :param iter_num: 进行完全聚类时迭代的次数
        :return:
        """
        window_record = self.window_record
        cur_tw_num = self.g.get_current_tw_num()
        window_record.append(cur_tw_num)
        print('  bclu: ** checkpoint **, record={}'.format(window_record))
        if len(window_record) > self.max_window_size:
            # if self.countdown == 0:
            #     self.g.full_cluster(iter_num)
            #     self.countdown += 1
            # elif self.countdown < self.full_interval:
            #     self.countdown += 1
            # elif self.countdown == self.full_interval:
            #     self.countdown = 0
            if self.countdown.has_reach_ref():
                print('  do full')
                self.g.full_cluster(iter_num)
            self.countdown.count()
            self.pop_oldest_record()
    
    def pop_oldest_record(self):
        """
        self.window_record 中最早的一条记录指示了聚类器当前持有的推特列表中，
        最早的一组推特列表的长度 oldest_k，该方法将这些列表从聚类器中弹出，
        并将 self.window_record 中其余元素均减去 oldest_k
        :return:
        """
        window_record = self.window_record
        if not window_record:
            return
        oldest_k = window_record.pop(0)
        self.g.pop_oldest(oldest_k)
        for i in range(len(window_record)):
            window_record[i] -= oldest_k
        print('  bclu: pop oldest {} tws, remain tw={}, record={}'.
              format(oldest_k, self.g.get_current_tw_num(), window_record))
    
    def input_twarr(self, twarr):
        """
        对输入的推特列表进行id去重，将去重结果添加进入聚类器待命，
        若聚类器中已形成聚类管理器，则对输入的推特列表进行采样，为其分配已有或新聚类；
        否则，仅进行添加操作
        :param twarr: list，推特列表
        :return:
        """
        g = self.g
        twarr = g.filter_dup_id(twarr)
        n = len(twarr)
        self.hist_len += n
        self.read_batch_num += 1
        print('  bclu -> new {} tw, #hist: {}, #batch: {}'.format(n, self.hist_len, self.read_batch_num))
        param = [
            dict(tw_batch=twarr, action=g.ACT_STORE, iter_num=None),
            dict(tw_batch=twarr, action=g.ACT_SAMPLE, iter_num=4),
        ][0 if g.cludict is None else 1]
        g.input_batch(**param)
    
    def output(self, out_channel):
        """
        将 clustering.gsdpmm.gsdpmm_stream_ifd_dynamic.GSDPMMStreamIFDDynamic#get_cluid_twarr_list
        的返回结果送入输出到 out_channel 中
        :param out_channel: mp.Queue，即子进程的outq2，用于跨进程传输数据
        :return:
        """
        twnum_thres = 4
        cluid_twarr_list = self.g.get_cluid_twarr_list(twnum_thres)
        print('cluid_twarr_list len: ', len(cluid_twarr_list) if cluid_twarr_list is not None else 0)
        out_channel.put(cluid_twarr_list)


def test1():
    import utils.tweet_keys as tk
    import utils.array_utils as au
    import utils.pattern_utils as pu
    import utils.timer_utils as tmu
    import calling.back_extractor as bext
    import utils.file_iterator as fi
    import utils.function_utils as fu
    from extracting.cluster_infomation import merge_cic_list2cluid_twarr_list
    
    # C_NAME = BackCluster.G_CLASS.__name__
    # _base = '/home/nfs/cdong/tw/src/calling/tmp_{}'.format(C_NAME)
    # fi.mkdir(_base, remove_previous=True)
    # _cluid_cluster_list_file = '_cluid_cluster_list_{}'.format(C_NAME)
    
    _twarr = fu.load_array("./filtered_twarr.json")[:5000]
    _batches = au.array_partition(_twarr, [1] * 43, random=False)
    
    _max_window_size, _full_interval = 4, 2
    _alpha, _beta = 30, 0.01
    start_pool(_max_window_size, _full_interval, _alpha, _beta)
    for idx, twarr in enumerate(_batches):
        input_twarr(twarr)
        if idx > 0 and idx % 2 == 0:
            execute_cluster(10)
            ctl = wait_get_cluid_twarr_list()
            print('len(ctl)={}'.format(len(ctl)) if ctl else 'ctl is None')
    
    end_pool()


if __name__ == '__main__':
    import utils.timer_utils as tmu
    tmu.check_time()
    test1()
    tmu.check_time()
    # import utils.tweet_keys as tk
    # import utils.array_utils as au
    # import utils.pattern_utils as pu
    # import utils.timer_utils as tmu
    # import calling.back_extractor as bext
    # import utils.file_iterator as fi
    # import utils.function_utils as fu
    # from extracting.cluster_infomation import merge_cic_list2cluid_twarr_list
    #
    # C_NAME = BackCluster.G_CLASS.__name__
    # _base = '/home/nfs/cdong/tw/src/calling/tmp_{}'.format(C_NAME)
    # fi.mkdir(_base, remove_previous=True)
    # _cluid_cluster_list_file = '_cluid_cluster_list_{}'.format(C_NAME)
    #
    #
    # _twarr = fu.load_array("./filtered_twarr.json")[:10000]
    # _batches = au.array_partition(_twarr, [1] * 10, random=False)
    #
    #
    # exit()
    # tmu.check_time()
    # if not fi.exists(_cluid_cluster_list_file):
    #     _batch_size = 100
    #     _hold_batch_num = 100
    #     _alpha, _beta = 30, 0.01
    #     start_pool(_hold_batch_num, _batch_size, _alpha, _beta)
    #     _file = "./filtered_twarr.json"
    #     _twarr = fu.load_array(_file)[:10100]
    #     input_twarr(_twarr)
    #     print('---> waiting for _cluid_cluster_list')
    #     _cluid_cluster_list = wait_until_get_cluid_twarr_list()
    #     print('---> get _cluid_cluster_list, len:{}'.format(len(_cluid_cluster_list)))
    #     fu.dump_array(_cluid_cluster_list_file, _cluid_cluster_list)
    #
    # tmu.check_time()
    # _cluid_cluster_list = fu.load_array(_cluid_cluster_list_file)
    # _ext_pool_size = 10
    # bext.start_pool(_ext_pool_size)
    #
    # bext.input_cluid_twarr_list(_cluid_cluster_list)
    # print('waiting for cic outputs')
    # _cic_list = bext.get_batch_output()
    # print('len(_cic_list)', len(_cic_list))
    # tmu.check_time()
    #
    # _new_cluid_twarr_list = merge_cic_list2cluid_twarr_list(_cic_list)
    # bext.input_cluid_twarr_list(_new_cluid_twarr_list)
    # _new_cic_list = bext.get_batch_output()
    # print(len(_new_cluid_twarr_list), len(_new_cic_list))
    # _new_cic_list = sorted(_new_cic_list, key=lambda item: len(item.twarr), reverse=True)
    #
    # for _cic in _new_cic_list:
    #     cluid, clu_twarr = _cic.cluid, _cic.twarr
    #     small_geo_table = _cic.small_geo_table
    #     twnum = len(clu_twarr)
    #
    #     print('cluid: {}'.format(cluid))
    #     print('twnum: {}'.format(twnum))
    #     for row in small_geo_table:
    #         print(row['address'], row['freq'])
    #     print(_cic.keywords)
    #     print('\n\n')
    #
    #     if len(small_geo_table) == 0:
    #         title = 'NOGPE'
    #     else:
    #         title = '*'.join([row['address'] for row in small_geo_table[:8]])
    #         if len(small_geo_table) >= 8:
    #             title += '+++'
    #     _out_file = _base + '/id={}_twnum={}_{}.txt'.format(cluid, twnum, title)
    #     _txtarr = [pu.text_normalization(tw[tk.key_text]) for tw in _cic.twarr]
    #     _idx_g, _txt_g = au.group_similar_items(_txtarr, score_thres=0.3, process_num=20)
    #     _txt_g = [sorted(g, key=lambda t: len(t), reverse=True) for g in _txt_g]
    #     _txtarr = au.merge_array(_txt_g)
    #     fu.write_lines(_out_file, _txtarr)
    #
    # tmu.check_time()
    # exit()
    # # geo2group_id, cluid2group = dict(), dict()
    # # group_counter = Counter()
    # # for _cic in _cic_list:
    # #     cluid, clu_twarr = _cic.cluid, _cic.twarr
    # #     clu_summary, clu_geo_list = _cic.od['summary'], _cic.od['geo_infer']
    # #     not_country_geo = [geo['address'] for geo in clu_geo_list if geo['quality'] != 'country']
    # #     # record every cluster
    # #     twnum = len(_cic.twarr)
    # #     print('cluid:{}'.format(cluid))
    # #     print(clu_summary['keywords'])
    # #     print('twarr len:{}'.format(twnum))
    # #     print('all gpe', [(geo['address'], geo['freq'], '{}%'.format(round(geo['freq']/len(clu_twarr)*100, 2)))
    # #                       for geo in clu_geo_list])
    # #     print('not country gpe', [geo_name for geo_name in not_country_geo])
    # #     print()
    # #     if len(not_country_geo) == 0:
    # #         title = 'NOGPE'
    # #     else:
    # #         title = '*'.join(not_country_geo[:8])
    # #         if len(not_country_geo) >= 8:
    # #             title += '+++'
    # #     _out_file = _base + '/id={}_twnum={}_{}.txt'.format(cluid, twnum, title)
    # #     _txtarr = [pu.text_normalization(tw[tk.key_text]) for tw in _cic.twarr]
    # #     _idx_g, _txt_g = au.group_similar_items(_txtarr, score_thres=0.3, process_num=20)
    # #     _txt_g = [sorted(g, key=lambda t: len(t), reverse=True) for g in _txt_g]
    # #     _txtarr = au.merge_array(_txt_g)
    # #     fu.write_lines(_out_file, _txtarr)
    # #     # try to group clusters
    # #     if len(not_country_geo) == 0:
    # #         cluid2group[cluid] = [_cic]
    # #         group_counter['--NOGPE--'] += 1
    # #     else:
    # #         clu_top_geo = not_country_geo[0]
    # #         group_counter[clu_top_geo] += 1
    # #         if clu_top_geo not in geo2group_id:
    # #             geo2group_id[clu_top_geo] = cluid
    # #             cluid2group[cluid] = [_cic]
    # #         else:
    # #             group_id = geo2group_id[clu_top_geo]
    # #             cluid2group[group_id].append(_cic)
    # #
    # # no_gpe_count = group_counter.pop('--NOGPE--')
    # # count_list = [('--NOGPE--', no_gpe_count)]
    # # count_list.extend(group_counter.most_common())
    # # df = pd.DataFrame(columns=['GEO NAME', 'COUNT'], data=count_list)
    # # df.to_csv('geo_count_{}.csv'.format(C_NAME))
    # # # print(df)
    # # # L = len(_cic_list)
    # # # no_gpe_count = group_counter.pop('--NOGPE--')
    # # # print('{: <3}/{: <3}={: <6}%, {}\n'.format(no_gpe_count, L, round(no_gpe_count/L*100, 2), '--NOGPE--'))
    # # # df.append()
    # # # for gpe, count in group_counter.most_common():
    # # #     print('{: <3}/{: <3}={: <6}%, {}'.format(count, L, round(count/L*100, 2), gpe))
    # # # print('\n')
    # #
    # # print('gnum, clunum', len(cluid2group), len(_cic_list))
    # # # merge clusters by geo name
    # # for group_id, group in cluid2group.items():
    # #     print('--- g ---')
    # #     for _cic in group:
    # #         cluid, clu_twarr = _cic.cluid, _cic.twarr
    # #         clu_summary, clu_geo_list = _cic.od['summary'], _cic.od['geo_infer']
    # #         not_country_geo = [(geo['address'], '{}%'.format(round(geo['freq']/len(clu_twarr)*100, 2)))
    # #                            for geo in clu_geo_list if geo['quality'] != 'country']
    # #         print('id:{: <4} tw num:{: <4}'.format(cluid, len(clu_twarr)), not_country_geo)
    # #     print('\n')
    # #
    # # tmu.check_time()
