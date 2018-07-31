from utils.utils_loader import *


en_nlp = spacy.load('en_core_web_sm', vector=False)
process_num = 5


def merge_geonames2counter(geonames2counter_list):
    merge = dict()
    for geonames2counter in geonames2counter_list:
        for geoname, counter in geonames2counter.items():
            if geoname not in merge:
                merge[geoname] = counter
            else:
                merge[geoname] += counter
    for geoname, counter in merge.items():
        merge[geoname] = dict(counter)
    return merge


def extract_geo_names(file_list):
    file_list_blocks = mu.split_multi_format(file_list, process_num)
    res_list = mu.multi_process(extract_geo_names_from_files, [[flist] for flist in file_list_blocks])
    return merge_geonames2counter(au.merge_array(res_list))


def extract_geo_names_from_files(file_list):
    res_list = list()
    for file in file_list:
        twarr = fu.load_array(file)
        geonames2counter = extract_geo_names_from_twarr(twarr)
        res_list.append(geonames2counter)
    return res_list


def extract_geo_names_from_twarr(twarr):
    textarr = [tw[tk.key_text] for tw in twarr if len(tw[tk.key_text]) > 20][:5000]
    docarr = su.textarr_nlp(textarr, nlp=en_nlp)
    geonames2counter = dict()
    for doc in docarr:
        for ent in doc.ents:
            ent_name = pu.capitalize(ent.text.strip())
            if not(ent.label_ in su.LABEL_LOCATION and pu.has_azAZ(ent_name) and len(ent_name) > 1):
                continue
            if ent_name not in geonames2counter:
                geonames2counter[ent_name] = Counter()
            geonames2counter[ent_name][ent.label_] += 1
    return geonames2counter


# def merge_geo_names():
#     from extracting.geo_and_time.extract_geo_loction import get_geo_by_loc_name
#     base = "/home/nfs/cdong/tw/src/extracting/geo_and_time/"
#     files = fi.listchildren(base, fi.TYPE_FILE, ".json$")
#     ifd = IdFreqDict()
#     for file in files:
#         word_freq_list = fu.load_array(file)
#         for word, freq in word_freq_list:
#             if len(word) <= 1:
#                 continue
#             if len(word) == 2:
#                 word = word.upper()
#             if get_geo_by_loc_name(word) is not None:
#                 continue
#             ifd.count_word(word, freq)
#     fu.dump_array("geo_freq.json", ifd.most_common())


if __name__ == "__main__":
    base = "/home/nfs/cdong/tw/origin/"
    sub_files = fi.listchildren(base, fi.TYPE_FILE, concat=True)
    tmu.check_time()
    geonames2counter = extract_geo_names(sub_files)
    tmu.check_time()
    res_list = [(geo_name, counter) for geo_name, counter in geonames2counter.items()]
    res_list = sorted(res_list, key=lambda item: sum([n for g, n in item[1].items()]), reverse=True)
    fu.dump_array("geonames2counter.json", res_list)
