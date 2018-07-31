import math
import multiprocessing as mp
from collections import Counter


def split_multi_format(array, process_num):
    block_size = math.ceil(len(array) / process_num)
    formatted_array = list()
    for i in range(process_num):
        arr_slice = array[i * block_size: (i + 1) * block_size]
        if arr_slice:
            formatted_array.append(arr_slice)
    return formatted_array


def multi_process(func, args_list=None, kwargs_list=None):
    """
    Do func in multiprocess way.
    :param func: To be executed within every process
    :param args_list: default () as param for apply_async if not given
    :param kwargs_list:
    :return:
    """
    process_num = len(args_list) if args_list is not None else len(kwargs_list)
    pool = mp.Pool(processes=process_num)
    res_getter = list()
    for i in range(process_num):
        res_holder = pool.apply_async(func=func,
                                      args=args_list[i] if args_list else (),
                                      kwds=kwargs_list[i] if kwargs_list else {})
        res_getter.append(res_holder)
    pool.close()
    pool.join()
    results = [rh.get() for rh in res_getter]
    return results


def multi_process_batch(func, batch_size=8, args_list=None, kwargs_list=None):
    total_num = len(args_list)
    if kwargs_list is None:
        kwargs_list = [{} for _ in range(total_num)]
    input_num = 0
    results = list()
    while input_num < total_num:
        since, until = input_num, input_num + batch_size
        res_batch = multi_process(func, args_list[since: until], kwargs_list[since: until])
        results.extend(res_batch)
        input_num += batch_size
    return results


class DaemonPool:
    def __init__(self):
        self.service_on = False
        self.daemon_class = None
        self.daemon_pool = list()
        self._last_task_len = list()
        self._last_batch_len = list()
    
    def is_service_on(self):
        return self.service_on
    
    def start(self, func, pool_size):
        if self.is_service_on():
            return
        for i in range(pool_size):
            daemon = self.daemon_class(i)
            daemon.start(func)
            self.daemon_pool.append(daemon)
        self.service_on = True
    
    def end(self):
        if not self.service_on:
            return
        for daeprocess in self.daemon_pool:
            daeprocess.end()
        self.daemon_pool.clear()
        self.service_on = False


class DaemonProcess:
    def __init__(self, pidx=0):
        self.process = None
        self.pidx = pidx
        self.inq = mp.Queue()
        self.outq = mp.Queue()
    
    def set_input(self, args):
        self.inq.put(args)
    
    def get_output(self):
        return self.outq.get()
    
    def start(self, func):
        self.process = mp.Process(target=func, args=(self.inq, self.outq))
        self.process.daemon = True
        self.process.start()
    
    def end(self):
        self.process.terminate()


class CustomDaemonPool(DaemonPool):
    def __init__(self, daemon_class=DaemonProcess):
        DaemonPool.__init__(self)
        self.daemon_class = daemon_class
    
    def set_input(self, arg_list):
        arg_len = len(arg_list)
        self._last_task_len.append(arg_len)
        for idx in range(arg_len):
            daemon = self.daemon_pool[idx]
            args = arg_list[idx]
            daemon.set_input(args)
    
    def get_output(self):
        arg_len = self._last_task_len.pop(0)
        res_list = list()
        for idx in range(arg_len):
            daemon = self.daemon_pool[idx]
            res_list.append(daemon.get_output())
        return res_list
    
    def set_batch_input(self, arg_list):
        # one item in arg_list will be input to a process
        dpnum = len(self.daemon_pool)
        innum = count = 0
        while innum < len(arg_list):
            self.set_input(arg_list[innum: innum + dpnum])
            innum += dpnum
            count += 1
        self._last_batch_len.append(count)
    
    def get_batch_output(self):
        batch_len = self._last_batch_len.pop(0)
        res_list = list()
        for i in range(batch_len):
            res_list.extend(self.get_output())
        return res_list
    
    """ state of the pool """
    def has_unread_batch_output(self):
        return len(self._last_batch_len) > 0
    
    def get_unread_batch_output_num(self):
        return len(self._last_batch_len)
    
    def last_batch_task_num_requirement(self):
        p2require = Counter()
        last_batch_size = self._last_batch_len[0]
        for task_idx in range(last_batch_size):
            task_num = self._last_task_len[task_idx]
            for pidx in range(task_num):
                p2require[pidx] += 1
        return p2require
    
    def process_outq_ready_size(self):
        p2qsize = dict([(pidx, daemon.outq.qsize()) for pidx, daemon in enumerate(self.daemon_pool)])
        return p2qsize
    
    def can_read_batch_output(self):
        if not self.has_unread_batch_output():
            return False
        p2require = self.last_batch_task_num_requirement()
        p2qsize = self.process_outq_ready_size()
        # print(p2require)
        # print(p2qsize)
        for pidx in p2require.keys():
            ready_size, requirement = p2qsize[pidx], p2require[pidx]
            if ready_size < requirement:
                # print('cannot output batch')
                return False
        # print('output batch')
        return True


""" process & pool for general purpose """


class GeneralPurposeProcess:
    def __init__(self):
        self.process = None
        self.inq_dict = self.outq_dict = None
    
    @staticmethod
    def _get_queue_dict_with_titles(titles):
        return dict([(title, mp.Queue()) for title in titles])
    
    def set_inq_entries(self, in_entries):
        self.inq_dict = self._get_queue_dict_with_titles(in_entries)
    
    def set_outq_entries(self, out_entries):
        self.outq_dict = self._get_queue_dict_with_titles(out_entries)
    
    def set_input(self, kv_iter):
        for k, v in kv_iter:
            if k in self.inq_dict:
                self.inq_dict[k].put(v)
            else:
                print('key {} is not in inq'.format(k))
    
    def get_output(self, k_iter):
        out_list = list()
        for k in k_iter:
            if k in self.outq_dict:
                v = self.outq_dict[k].get()
                out_list.append((k, v))
            else:
                print('key {} is not in outq'.format(k))
        return out_list
    
    def start(self, func, is_daemon):
        self.process = mp.Process(target=func, args=(self.inq_dict, self.outq_dict))
        self.process.daemon = is_daemon
        self.process.start()
    
    def end(self):
        self.process.terminate()


class GeneralPurposePool:
    def __init__(self, process_class=GeneralPurposeProcess):
        self.service_on = False
        self.process_pool = list()
        self._alloc_hist = list()
        self.process_class = process_class
    
    def is_service_on(self):
        return self.service_on
    
    def start(self, func, is_daemon, pool_size):
        if self.is_service_on():
            return
        for i in range(pool_size):
            process = self.process_class()
            process.start(func, is_daemon)
            self.process_pool.append(process)
        self.service_on = True
    
    def end(self):
        if not self.is_service_on():
            return
        for daeprocess in self.process_pool:
            daeprocess.end()
        self.process_pool.clear()
        self.service_on = False
    
    def allocate_processes(self, arg_len):
        pool_size = len(self.process_pool)
        return [i for i in range(min(arg_len, pool_size))]
    
    def set_input(self, arg_list):
        arg_len = len(arg_list)
        self._last_task_len.append(arg_len)
        for idx in range(arg_len):
            daemon = self.process_pool[idx]
            args = arg_list[idx]
            daemon.set_input(args)
    
    def get_output(self):
        arg_len = self._last_task_len.pop(0)
        res_list = list()
        for idx in range(arg_len):
            daemon = self.process_pool[idx]
            res_list.append(daemon.get_output())
        return res_list
    
    # def set_batch_input(self, arg_list):
    #     # one item in arg_list will be input to a process
    #     dpnum = len(self.daemon_pool)
    #     innum = count = 0
    #     while innum < len(arg_list):
    #         self.set_input(arg_list[innum: innum + dpnum])
    #         innum += dpnum
    #         count += 1
    #     self._alloc_hist_batch.append(count)
    #
    # def get_batch_output(self):
    #     batch_len = self._alloc_hist_batch.pop(0)
    #     res_list = list()
    #     for i in range(batch_len):
    #         res_list.extend(self.get_output())
        return res_list
    
    """ state of the pool """
    
    def has_unread_batch_output(self):
        return len(self._alloc_hist_batch) > 0
    
    def get_unread_batch_output_num(self):
        return len(self._alloc_hist_batch)
    
    def last_batch_task_num_requirement(self):
        p2require = Counter()
        last_batch_size = self._alloc_hist_batch[0]
        for task_idx in range(last_batch_size):
            task_num = self._last_task_len[task_idx]
            for pidx in range(task_num):
                p2require[pidx] += 1
        return p2require
    
    def process_outq_ready_size(self):
        p2qsize = dict([(pidx, daemon.outq.qsize()) for pidx, daemon in enumerate(self.process_pool)])
        return p2qsize
    
    def can_read_batch_output(self):
        if not self.has_unread_batch_output():
            return False
        p2require = self.last_batch_task_num_requirement()
        p2qsize = self.process_outq_ready_size()
        # print(p2require)
        # print(p2qsize)
        for pidx in p2require.keys():
            ready_size, requirement = p2qsize[pidx], p2require[pidx]
            if ready_size < requirement:
                # print('cannot output batch')
                return False
        # print('output batch')
        return True







import utils.function_utils as fu
import utils.file_iterator as fi
import utils.timer_utils as tmu


def read(inq, outq):
    while True:
        idx, file = inq.get()
        twarr = fu.load_array(file)
        outq.put([idx, len(twarr)])


def read2(idx, file, nothing='p'):
    twarr = fu.load_array(file)
    return [idx, len(twarr)]


if __name__ == '__main__':
    # dp = CustomDaemonPool()
    # dp = ProxyDaemonPool()
    # dp.set_parameters(read2, 8)
    # base = '/home/nfs/cdong/tw/testdata/yying/2016_04/'
    # files = [base + sub for sub in subs][:40]
    base = '/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/'
    subs = fi.listchildren(base, children_type=fi.TYPE_FILE)
    files = [base + sub for sub in subs]
    
    tmu.check_time()
    res = multi_process_batch(read2, args_list=[(idx, file) for idx, file in enumerate(files)])
    # dp.set_batch_input([(idx, file) for idx, file in enumerate(files)],
    #                    [{'nothing': str(idx)+file} for idx, file in enumerate(files)])
    # res = dp.get_batch_output()
    # print(sum(([length for idx, length in res])))
    # print(res)
    tmu.check_time()
    print([[idx, len(fu.load_array(file))] for idx, file in enumerate(files)])
    tmu.check_time()
