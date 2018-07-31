from extracting.cluster_infomation import ClusterInfoGetter as CIG, merge_cic_list2cluid_twarr_list, write_cic_list
from utils.multiprocess_utils import mp, CustomDaemonPool, DaemonProcess

import utils.function_utils as fu
import utils.file_iterator as fi
from config.configure import getcfg


class ExtractSubProcess(DaemonProcess):
    def start(self, func):
        self.process = mp.Process(target=func, args=(self.inq, self.outq))
        self.process.daemon = False
        self.process.start()
        # print('ExtractSubProcess', self.process.pid)


extract_sub_process = ExtractSubProcess()


END_PROCESS = -1
SET_PARAMS = 0
INPUT_LIST = 1
OUTPUT_LIST = 3
OUT_BASE = getcfg().output_path
fi.mkdir(OUT_BASE, remove_previous=True)


def extract_sub_main(inq, outq):
    """
    聚类信息提取模块子进程的主函数，该进程还有下属的若干子进程用于执行实际操作，
    负责读取主进程输入，调用子进程组进行一次地点提取，根据返回结果，对多个聚类依地点进行合并操作，
    并再次调用子进程组对合并结果进行完整处理，以该结果为最终结果输出到文件
    :param inq: mp.Queue，主进程向子进程的输入队列
    :param outq: mp.Queue，子进程向主进程的输出队列
    :return:
    """
    pool_size, event_type = inq.get()
    extract_pool = CustomDaemonPool()
    extract_pool.start(extract_pool_main, pool_size)
    extract_pool.set_batch_input([(pidx, event_type) for pidx in range(pool_size)])
    extract_pool.get_batch_output()
    
    counter = 0
    while True:
        command = inq.get()
        if command == INPUT_LIST:
            cluid_twarr_list = inq.get()
            print('    bext: remain task={}, len(cluid_twarr_list)={}'.format(inq.qsize(), len(cluid_twarr_list)))
            if inq.qsize() > 6:
                print('    bext: too many tasks undone, jumping')
                continue
            # tmu.check_time('extract_sub_main', print_func=lambda dt: print('lline 41', dt))
            # print("    bext: get new cluid_twarr_list, len {}".format(len(cluid_twarr_list)))
            extract_pool.set_batch_input(cluid_twarr_list)
            cic_list = extract_pool.get_batch_output()
            # tmu.check_time('extract_sub_main', print_func=lambda dt: print('lline 43', dt))
            new_cluid_twarr_list = merge_cic_list2cluid_twarr_list(cic_list)
            new_cluid_twarr_list = [(cluid, twarr, CIG.TAR_FULL) for cluid, twarr in new_cluid_twarr_list]
            extract_pool.set_batch_input(new_cluid_twarr_list)
            n_cic_list = extract_pool.get_batch_output()
            # tmu.check_time('extract_sub_main', print_func=lambda dt: print('lline 49', dt))
            # print("    bext: get merged cic list, len {}".format(len(n_cic_list)))
            write_cic_list(fi.join(OUT_BASE, "{}_{}clusters/").format(counter, len(n_cic_list)), n_cic_list)
            counter += 1
        elif command == END_PROCESS:
            print('ending extract_sub_main')
            outq.put('ending extract_sub_main')
            return


def extract_pool_main(inq, outq):
    """
    提取子进程组中各子进程的主函数，每次从inq读入单一聚类的编号以及推特列表，
    调用ClusterInfoGetter对象对上述输入进行处理，用outq返回提取结果的输出
    :param inq: mp.Queue，主进程向子进程的输入队列
    :param outq: mp.Queue，子进程向主进程的输出队列
    :return:
    """
    pidx, event_type = inq.get()
    cig = CIG(event_type)
    outq.put(pidx)
    while True:
        arg = inq.get()
        cic = cig.cluid_twarr2cic(*arg)
        outq.put(cic)


def start_pool(pool_size, event_type):
    """
    启动提取模块的进程池，为多个以 per_cluid_twarr 为主的子进程分配资源，并待命
    :param pool_size: int，进程池中的进程个数
    :param event_type: str，分类器所针对的事件类型
    :return:
    """
    extract_sub_process.start(extract_sub_main)
    extract_sub_process.set_input((pool_size, event_type))


def end_pool():
    """
    发送终止进程的指令，子进程一旦处理完该指令之前所有的输入指令，则在收到该指令后退出
    :return:
    """
    extract_sub_process.set_input(END_PROCESS)
    extract_sub_process.get_output()


def input_cluid_twarr_list(cluid_twarr_list):
    """
    向子进程输入要求其处理的聚类列表，不等待返回结果
    :param cluid_twarr_list: list，每个元素为tuple
        见 clustering.gsdpmm.gsdpmm_stream_ifd_dynamic.GSDPMMStreamIFDDynamic#get_cluid_twarr_list
    :return:
    """
    if cluid_twarr_list:
        extract_sub_process.set_input(INPUT_LIST)
        extract_sub_process.set_input(cluid_twarr_list)


if __name__ == '__main__':
    import utils.timer_utils as tmu
    _base = "/home/nfs/cdong/tw/seeding/Terrorist/queried/positive"
    files = fi.listchildren(_base, concat=True)
    _cluid_twarr_list = [(idx, fu.load_array(file)[:1000]) for idx, file in enumerate(files)]
    
    start_pool(10, 'terrorist_attack')
    tmu.check_time()
    for i in range(2):
        input_cluid_twarr_list(_cluid_twarr_list)
    end_pool()
    tmu.check_time()
    exit()
