import numpy as np
from sklearn import metrics
import utils.multiprocess_utils as mu
import Levenshtein


def merge_array(array):
    """
    将array中iterable的元素拼接到一个新的list中
    :param array: list，含有iterable的元素，如list，tuple等
    :return: list，拼接结果
    """
    res_list = list()
    for arr in array:
        res_list.extend(arr)
    return res_list


def random_array_items(array, item_num, keep_order=True):
    """ Get random items from array.
    :param array: From which items are extracted.
    :param item_num: Number of items to be extracted.
    :param keep_order: If remains the relative order within array in the result.
    :return:
    """
    array_len = len(array)
    item_num = array_len if item_num >= array_len else item_num
    item_index = np.random.choice(array_len, item_num, replace=False)
    item_index = sorted(item_index) if keep_order else item_index
    return [array[i] for i in item_index]


def array_partition(array, partition_arr=(1, 1, 1), random=True, ordered=False):
    indexes = index_partition(array, partition_arr, random)
    return [[array[j] for j in indexes[i]] for i in range(len(indexes))]


def index_partition(array, partition_arr=(1, 1, 1), random=True, ordered=False):
    item_num = len(array)
    indexes = np.array([i for i in range(item_num)])
    indexes = shuffle(indexes) if random else indexes
    normed_portion = np.array(partition_arr) / np.sum(partition_arr)
    sum_idx = [0]
    sum_normed_por = [0]
    for portion in normed_portion:
        sum_normed_por.append(sum_normed_por[-1] + portion)
        sum_idx.append(int(sum_normed_por[-1] * item_num))
    sum_idx[-1] = item_num
    return [indexes[sum_idx[i]: sum_idx[i + 1]] for i in range(len(partition_arr))]


def shuffle(array, inplace=True):
    array = array if inplace else array[:]
    np.random.shuffle(array)
    return array


def score(y_true, y_score, score_type):
    score_type = score_type.lower()
    if score_type == 'auc':
        return metrics.roc_auc_score(y_true, y_score)
    elif score_type == 'nmi':
        return metrics.normalized_mutual_info_score(y_true, y_score)
    elif score_type == 'homo':
        return metrics.homogeneity_score(y_true, y_score)
    elif score_type == 'cmplt':
        return metrics.completeness_score(y_true, y_score)
    elif score_type == 'ari':
        return metrics.adjusted_rand_score(y_true, y_score)


def precision_recall_threshold(labels_true, labels_pred, thres_range=[i/10 for i in range(1, 10)], file=None):
    import pandas as pd
    auc = score(labels_true, labels_pred, 'auc')
    print("auc:", round(auc, 6))
    
    precision, recall, thresholds = metrics.precision_recall_curve(labels_true, labels_pred)
    last_idx = 0
    columns = ['threshold', 'precision', 'recall']
    df = pd.DataFrame(columns=columns)
    for thres in thres_range:
        for idx in range(last_idx, len(thresholds)):
            if thresholds[idx] >= thres:
                data = [[round(thresholds[idx], 2), round(precision[idx], 4), round(recall[idx], 4)]]
                df = df.append(pd.DataFrame(columns=columns, data=data), ignore_index=True)
                last_idx = idx
                break
    print(df)
    if file:
        df.to_csv(file)


def permutation_generator(max_of_each_v, generator):
    if 0 in max_of_each_v:
        raise ValueError('0 value should not appear in max_of_each_v')
    v_num = len(max_of_each_v)
    vec = [0] * v_num
    while True:
        yield generator(vec)
        idx = 0
        while idx < v_num and vec[idx] == max_of_each_v[idx] - 1:
            vec[idx] = 0
            idx += 1
        if idx < v_num:
            vec[idx] += 1
        else:
            break


def sample_index(array):
    return np.random.choice(a=[i for i in range(len(array))], p=np.array(array) / np.sum(array))


def choice(array):
    return np.random.choice(array)


def cosine_similarity(vecarr1, vecarr2=None):
    return metrics.pairwise.cosine_similarity(vecarr1, vecarr2)


def cohesion_score(vecarr):
    vecnum = len(vecarr)
    if vecnum <= 1:
        return None
    cos_sim_mtx = cosine_similarity(vecarr)
    cohesion = 0
    for i in range(0, vecnum - 1):
        for j in range(i + 1, vecnum):
            cohesion += cos_sim_mtx[i, j]
    pair_count = (vecnum - 1) * vecnum / 2
    return cohesion / pair_count


def text_dist_proportion(text1, text2):
    edit_dist = Levenshtein.distance(text1, text2)
    max_len = max(len(text1) + 1, len(text2) + 1)
    dist = edit_dist / max_len
    weighted_dist = np.power(dist, np.tanh(max_len / 6))
    return weighted_dist


def group_and_reduce(array, dist_thres, process_num, pair_func=text_dist_proportion):
    idx_groups, item_groups = group_similar_items(array, dist_thres, process_num, pair_func)
    idx_reduce, item_reduce = [g[0] for g in idx_groups], [g[0] for g in item_groups]
    return idx_reduce, item_reduce


def group_similar_items(array, score_thres, process_num, pair_func=text_dist_proportion):
    idx_pair_score_list = pairwise_score_multi(array, process_num, pair_func)
    total = len(array)
    union_find_set = [idx for idx in range(total)]
    def find_root(num):
        while union_find_set[num] != num:
            num = union_find_set[num]
        return num
    for (x, y), s in idx_pair_score_list:
        if s < score_thres:
            union_find_set[y] = find_root(x)
    groups = [[] for _ in range(total)]
    for idx, root in enumerate(union_find_set):
        groups[root].append(idx)
    idx_groups = [idx_group for idx_group in groups if idx_group]
    item_groups = [[array[idx] for idx in idx_group] for idx_group in idx_groups]
    return idx_groups, item_groups


def pairwise_score_multi(array, process_num, pair_func):
    total = len(array)
    pairs = [(i, j) for i in range(total - 1) for j in range(i + 1, total)]
    if process_num <= 1 or total < 40:
        idx_pair_score_list = pairwise_score(array, pairs, pair_func)
    else:
        pair_blocks = mu.split_multi_format(pairs, process_num)
        arg_list = [(array, idx_pairs, pair_func) for idx_pairs in pair_blocks]
        score_pairs_blocks = mu.multi_process(pairwise_score, arg_list)
        idx_pair_score_list = merge_array(score_pairs_blocks)
    return idx_pair_score_list


def pairwise_score(array, idx_pairs, pair_func):
    idx_pairs_score_list = list()
    for x, y in idx_pairs:
        s = pair_func(array[x], array[y])
        idx_pairs_score_list.append(((x, y), s))
    return idx_pairs_score_list
