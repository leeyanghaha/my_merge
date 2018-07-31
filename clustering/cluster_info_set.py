from os import path
from collections import Counter

from classifying.terror.classifier_terror import ClassifierTerror

import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.tweet_utils as tu
import utils.tweet_keys as tk
import utils.timer_utils as tmu
import utils.spacy_utils as su
import utils.pattern_utils as pu
import preprocess.tweet_filter as tflt

from spacy.tokens import Doc


# def twarr2spacy_files(twarr, base):
#     tmu.check_time("twarr2spacy_files")
#     twarr = tu.twarr_nlp(twarr)
#     tmu.check_time("twarr2spacy_files")
#     print("{} tweets ".format(len(twarr)))
#     for tw in twarr:
#         doc, twid = tw[tk.key_spacy], str(tw[tk.key_id])
#         doc.to_disk(path.join(base, twid))
#     files = fi.listchildren(base, fi.TYPE_FILE)
#     print("{} files under {}".format(len(files), base))
#
#
# def twid2spacy_file(base):
#     files = fi.listchildren(base, fi.TYPE_FILE, concat=False)
#     return dict([(file, path.join(base, file)) for file in files])
#
#
# def get_docarr_of_twarr_from_base(twarr, base):
#     tmu.check_time("get_docarr_of_twarr")
#     vocab = su.get_nlp().vocab
#     id2file_dict = twid2spacy_file(base)
#     for idx, tw in enumerate(twarr):
#         if idx % int(len(twarr) / 10) == 0:
#             print("{} % docs restored".format(round(idx / len(twarr) * 100, 2)))
#         twid = str(tw[tk.key_id])
#         if twid in id2file_dict:
#             doc = Doc(vocab)
#             doc.from_disk(id2file_dict[twid])
#         else:
#             print("new tweet encountered")
#             doc = su.text_nlp(tw[tk.key_text])
#             doc.to_disk(path.join(base, twid))
#         tw[tk.key_spacy] = doc
#     tmu.check_time("get_docarr_of_twarr", print_func=lambda dt: print("get_docarr in {}s".format(dt)))


def split_array_into_batches(array, batch_size):
    index = 0
    batches = list()
    while index < len(array):
        batches.append(array[index: index + batch_size])
        index += batch_size
    return batches


def refilter_twarr(in_file, out_file):
    twarr = fu.load_array(in_file)[:200000]
    origin_len = len(twarr)
    print(origin_len)
    clf_filter = ClassifierTerror()
    
    # for idx in range(len(twarr) - 1, -1, -1):
    #     text = twarr[idx][tk.key_text]
    #     if not pu.has_enough_alpha(text, 0.6):
    #         print(text)
    #         twarr.pop(idx)
    # text_filter_len = len(twarr)
    # print("delta by text =", origin_len - text_filter_len)
    
    tmu.check_time("refilter_twarr")
    twarr = clf_filter.filter(twarr, 0.2)
    tmu.check_time("refilter_twarr")
    print(len(twarr))
    fu.dump_array(out_file, twarr[:100000])
    # 300000 -> 197350


class HandpickedInfoSet:
    def __init__(self):
        base = "/home/nfs/cdong/tw/src/clustering/data/"
        self.labelled_batch_file = base + "batches.txt"
        self.cluid_batch_file = base + "cluid_evo.txt"
        self.batch_size = 100
        self.hold_batch_num = 50
    
    def lbarr_of_twarr(self, twarr):
        return [tw[tk.key_event_label] for tw in twarr]
    
    def order_twarr_through_time(self):
        print("data source : normal")
        event_blocks = fu.load_array("./data/events2016.txt")
        false_event_twarr = fu.load_array("./data/false_pos_events.txt")
        event_blocks.append(false_event_twarr)
        for block_idx, block in enumerate(event_blocks):
            for tw in block:
                tw[tk.key_event_label] = block_idx
        twarr = au.merge_array(event_blocks)
        tflt.filter_twarr_dup_id(twarr)
        
        def random_idx_for_item(item_arr, dest_item):
            from numpy import random
            def sample(prob):
                return random.rand() < prob
        
            non_dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] not in dest_item]
            dest_item_idx = [idx for idx in range(len(item_arr)) if item_arr[idx] in dest_item]
            non_dest_cnt = dest_cnt = 0
            res = list()
            while len(non_dest_item_idx) > non_dest_cnt and len(dest_item_idx) > dest_cnt:
                if sample((len(dest_item_idx) - dest_cnt) /
                          (len(dest_item_idx) - dest_cnt + len(non_dest_item_idx) - non_dest_cnt)):
                    res.append(dest_item_idx[dest_cnt])
                    dest_cnt += 1
                else:
                    res.append(non_dest_item_idx[non_dest_cnt])
                    non_dest_cnt += 1
            while len(non_dest_item_idx) > non_dest_cnt:
                res.append(non_dest_item_idx[non_dest_cnt])
                non_dest_cnt += 1
            while len(dest_item_idx) > dest_cnt:
                res.append(dest_item_idx[dest_cnt])
                dest_cnt += 1
            return res
    
        idx_time_order = tu.rearrange_idx_by_time(twarr)
        twarr = [twarr[idx] for idx in idx_time_order]
        lbarr = self.lbarr_of_twarr(twarr)
        idx_random_item = random_idx_for_item(lbarr, {max(lbarr)})
        twarr = [twarr[idx] for idx in idx_random_item]
        return twarr
    
    def twarr_info(self, twarr):
        lbarr = self.lbarr_of_twarr(twarr)
        label_distrb = Counter(lbarr)
        for idx, cluid in enumerate(sorted(label_distrb.keys())):
            print("{:<3}:{:<6}".format(cluid, label_distrb[cluid]), end="\n" if (idx + 1) % 10 == 0 else "")
        print("\nTopic num: {}, total tw: {}".format(len(label_distrb), len(twarr)))
    
    def make_tw_batches(self, batch_size):
        ordered_twarr = self.order_twarr_through_time()
        tw_batches = split_array_into_batches(ordered_twarr, batch_size)
        self.twarr_info(au.merge_array(tw_batches))
        fu.dump_array(self.labelled_batch_file, tw_batches)
    
    def load_tw_batches(self, load_cluid_arr):
        tw_batches = fu.load_array(self.labelled_batch_file)
        tu.twarr_nlp(au.merge_array(tw_batches))
        print("twarr nlp over")
        if load_cluid_arr:
            cluid_batches = fu.load_array(self.cluid_batch_file)
            assert len(tw_batches) == len(cluid_batches)
            for b_idx in range(len(tw_batches)):
                tw_batch, cluid_batch = tw_batches[b_idx], cluid_batches[b_idx]
                assert len(tw_batch) == len(cluid_batch)
                for idx in range(len(tw_batch)):
                    tw, cluid = tw_batch[idx], cluid_batch[idx]
                    tw[tk.key_event_cluid] = cluid
        return tw_batches


class FilteredInfoSet:
    def __init__(self):
        base = "/home/nfs/cdong/tw/src/clustering/data/"
        self.origin_filtered_twarr_file = base + "filtered.json"
        self.filtered_twarr_file = base + "filtered_twarr.json"
        self.filtered_cluidarr_file = base + "filtered_cluidarr.json"
        self.batch_size = 500
        self.hold_batch_num = 100
    
    def dump_cluidarr(self, cluidarr):
        fu.dump_array(self.filtered_cluidarr_file, cluidarr)
    
    def load_tw_batches(self, load_cluid_arr):
        temp_len = 60000
        twarr = fu.load_array(self.filtered_twarr_file)[:temp_len]
        print("load_tw_batches, len(twarr)=", len(twarr))
        if load_cluid_arr:
            cluidarr = fu.load_array(self.filtered_cluidarr_file)[:temp_len]
            assert len(twarr) == len(cluidarr)
            for idx in range(len(twarr)):
                tw, twid = twarr[idx], twarr[idx][tk.key_id]
                origin_id, cluid = cluidarr[idx]
                assert twid == origin_id
                tw[tk.key_event_cluid] = cluid
        twarr = tu.twarr_nlp(twarr)
        tw_batches = split_array_into_batches(twarr, self.batch_size)
        print("batch distrb {}, {} batches, total {} tweets".format(
            [len(b) for b in tw_batches], len(tw_batches), len(twarr)))
        return tw_batches


handpicked_info_set = HandpickedInfoSet()
filtered_info_set = FilteredInfoSet()


if __name__ == '__main__':
    import random
    # filtered_info_set.load_tw_batches(load_cluid_arr=True)
    # refilter_twarr(filtered_info_set.origin_filtered_twarr_file, filtered_info_set.filtered_twarr_file)
    twarr = fu.load_array(filtered_info_set.origin_filtered_twarr_file)
    len_pre = len(twarr)
    for idx in range(len(twarr) - 1, -1, -1):
        text = twarr[idx][tk.key_text]
        if not pu.has_enough_alpha(text, 0.6):
            twarr.pop(idx)
            print(text)
    print(len_pre, '->', len(twarr), len(twarr) - len_pre)
    # fu.dump_array(filtered_info_set.origin_filtered_twarr_file, twarr)
