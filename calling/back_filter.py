import classifying.classifier_class_info as cci
from preprocess.filter.yying_non_event_filter import EffectCheck
from utils.multiprocess_utils import CustomDaemonPool
import utils.tweet_keys as tk
import utils.pattern_utils as pu


flt_clf_pool = CustomDaemonPool()
class_table = {
    cci.event_t: cci.ClassifierTerror,
    cci.event_nd: cci.ClassifierNaturalDisaster,
    cci.event_k: cci.ClassifierK,
}


def filter_twarr_text(twarr):
    """
    对输入的推特列表，对所有推特的文本进行预处理，抛弃预处理结果不合规的推特；
    每条推特 tk.key_orgntext 字段保留原始文本， tk.key_text 字段保留预处理结果
    :param twarr: list，推特列表
    :return: list，经过文本预处理以及筛选的推特列表
    """
    flt_twarr = list()
    for tw in twarr:
        # TODO text_orgn = tw.get(tk.key_text, '').strip()
        text_orgn = tw.get(tk.key_orgntext, tw.get(tk.key_text, None)).strip()
        if not text_orgn:
            continue
        text_norm = pu.text_normalization(text_orgn).strip()
        if pu.is_empty_string(text_norm) or not pu.has_enough_alpha(text_norm, 0.65):
            continue
        tw[tk.key_orgntext] = text_orgn
        tw[tk.key_text] = text_norm
        flt_twarr.append(tw)
    return flt_twarr


def filter_and_classify_main(inq, outq):
    """
    过滤&分类模块进程池中各子进程的主函数；从inq读取主进程输入，包括分类阈值、推特列表等；
    首先进行文本预处理，并根据文本进行过滤；随后去除对话、骚扰、广告推特（ne_filter），
    再分类出特定类型事件的推特列表（clf_filter）；最终通过outq向主进程返回过滤后的推特列表；
    ※注：每次从inq读取的是一个 batch 中的某个子列表，返回的也是对该子列表的处理结果
    :param inq: mp.Queue，子进程的输入队列
    :param outq: mp.Queue，子进程的输出队列
    :return:
    """
    pidx, ne_threshold, clf_threshold, event_type = inq.get()
    outq.put(pidx)
    ne_filter = EffectCheck()
    clf_filter = class_table[event_type]()
    while True:
        twarr = inq.get()
        # len1 = len(twarr)
        # print(len1)
        twarr = filter_twarr_text(twarr)
        flt_twarr = ne_filter.filter(twarr, ne_threshold)
        flt_twarr = clf_filter.filter(flt_twarr, clf_threshold)
        # len4 = len(twarr)
        # print('{}->{}'.format(len1, len4))

        outq.put(flt_twarr)


def start_pool(pool_size, ne_threshold, clf_threshold, event_type):
    """
    启动过滤&分类模块的进程池，为多个以 filter_and_classify_main 为主函数的子进程分配资源；
    阈值取得越高，过滤返回的推特数越多，越容易找出所有的正例，但同时会引入更多的负例
    :param pool_size: int，进程池中的进程个数
    :param ne_threshold: float，过滤器的过滤阈值
    :param clf_threshold: float，分类器的过滤阈值
    :param event_type: str，分类器所针对的事件类型
    :return:
    """
    flt_clf_pool.start(filter_and_classify_main, pool_size)
    flt_clf_pool.set_batch_input([(idx, ne_threshold, clf_threshold, event_type) for idx in range(pool_size)])
    flt_clf_pool.get_batch_output()


def input_twarr_batch(tw_batch):
    """
    将接收到的 batch 中各个子列表分别分配给各子进程；不等待返回结果，batch 结构如下
    batch
        + - 子列表1 - 子进程1处理并返回 +
        + - 子列表2 - 子进程2处理并返回 +
        +           ...                +
        + - 子列表n - 子进程n处理并返回 +
    n个进程的处理结果捆绑起来，即为该batch的处理结果
    :param tw_batch: list，每个元素为推特列表(list)
    :return:
    """
    if tw_batch:
        flt_clf_pool.set_batch_input(tw_batch)


def get_batch_output():
    """
    阻塞地从各子进程读取上一个输入 batch 的过滤&分类结果
    :return: list，子进程对一个 batch 的处理结果，每个元素为推特列表(list)
    """
    return flt_clf_pool.get_batch_output()


def can_read_batch_output():
    """
    判断各子进程是否已完成最早输入的一个 batch 的处理
    :return: bool，判断结果；
        若判断为是，则此时调用 get_batch_output ，可以直接从各子进程读取一个返回结果；
        若判断为否，则此时调用 get_batch_output ，将阻塞直至最早的 batch 处理完成饼返回
    """
    return flt_clf_pool.can_read_batch_output()


def is_unread_batch_num_over(num_thres):
    """
    判断子进程池目前滞留的未读取结果的 batch 的数量是否超出了某阈值
    :param num_thres: int，判定阈值
    :return: bool，判断结果；
        若判断为是，则意味着至少存在 num_thres 个 batch 的输出未被读取；
        若判断为否，则意味着当前子进程池的残存工作量至少不大于 num_thres
    """
    return flt_clf_pool.get_unread_batch_output_num() > num_thres


def try_get_unread_batch_output():
    """
    非阻塞地从子进程池读取任何未读取的返回结果
    :return: list，长度表示读取了多少个batch的输出；
        其元素为list，长度表示多少个子进程参与了该batch的处理
    """
    res_batches = list()
    while can_read_batch_output():
        batches = get_batch_output()
        res_batches.append(batches)
    return res_batches


def wait_get_unread_batch_output(remain_workload):
    """
    阻塞地从子进程池读取任何未读取的返回结果，直到未读取处理结果的 batch 个数少于 remain_workload
    :param remain_workload: int，指示直到负载降到多少之前一直需要阻塞等待
    :return: list，长度表示读取了多少个batch的输出；
        其元素为list，长度表示多少个子进程参与了该batch的处理
    """
    res_batches = list()
    while is_unread_batch_num_over(remain_workload):
        batches = get_batch_output()
        res_batches.append(batches)
    return res_batches
