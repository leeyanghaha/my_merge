from os import listdir
from os.path import isfile, join

import pickle
import traceback
import pandas as pd

from config.configure import getcfg
from preprocess.filter.ChatFilter import ChatFilter
from preprocess.filter.filter_utils import readFilesAsJsonList
import utils.pattern_utils as pu
import utils.tweet_keys as tk


# TODO 先用绝对路径标识文件名，我这里在配置文件中写成绝对路径了
class_dist_file = getcfg().class_dist_file
chat_filter_file = getcfg().chat_filter_file
is_noise_dict_file = getcfg().is_noise_dict_file
orgn_predict_label_file = getcfg().orgn_predict_label_file


class UseGSDMM:
    def __init__(self, trainning=None):
        self.c = ChatFilter()
        self.orgn_predict_label = None
        self.class_dist = None
        self.is_noise_dict = None

        if trainning is None:
            try:
                with open(chat_filter_file, 'rb') as f:
                    self.c = pickle.load(f)
                with open(orgn_predict_label_file, 'rb') as f:
                    self.orgn_predict_label = pickle.load(f)
                with open(class_dist_file, 'rb') as f:
                    self.class_dist = pickle.load(f)
                with open(is_noise_dict_file, 'rb') as f:
                    self.is_noise_dict = set(pickle.load(f))
            except:
                print('load error')
                traceback.print_exc()
        else:
            # prepare data
            mypath = '../data/'
            onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
            print(onlyfiles)
            twarrF = readFilesAsJsonList(onlyfiles)
            twarrT = trainning
            for idx, tw in enumerate(twarrF):
                twarrF[idx]['label'] = 0

            for idx, tw in enumerate(twarrT):
                twarrT[idx]['label'] = 1
            twarr = list()
            twarr.extend(twarrT)
            twarr.extend(twarrF)
            i = 0
            for tw in twarr:
                tw['text'] = pu.text_normalization(tw['orgn'])
                twarr[i] = tw
                i += 1

            # train
            self.c = ChatFilter()
            self.c.set_twarr(twarr)
            self.c.set_hyperparams(0.9, 0.01, 55)  # 推荐超参，论文里用的是alpha=0.1 * len(twarr), beta=0.02
            class_dist, orgn_predict_label = self.c.recluster_using_GSDMM()

            try:
                with open(chat_filter_file, 'wb') as f:
                    pickle.dump(self.c, f)
                with open(orgn_predict_label_file, 'wb') as f:
                    pickle.dump(orgn_predict_label, f)
                with open(class_dist_file, 'wb') as f:
                    pickle.dump(class_dist, f)
            except:
                print('save error')
                traceback.print_exc()

            # get isNoiseDict
            label = [tw['label'] for tw in twarr]
            table = pd.DataFrame(index=set(orgn_predict_label), columns=set(label), data=0)
            for i in range(len(label)):
                table.loc[orgn_predict_label[i], label[i]] += 1
            print(table)
            multiple_times = 30
            self.is_noise_dict = []
            zero_total = float(table[0].sum())
            one_total = float(table[1].sum())
            for index, row in table.iterrows():
                if row[1] == 0:
                    if row[0] > multiple_times:
                        self.is_noise_dict.append(index)
                    else:
                        continue
                elif (row[0] / zero_total) / (row[1] / one_total) > multiple_times:
                    self.is_noise_dict.append(index)
            with open(is_noise_dict_file, 'wb') as f:
                pickle.dump(self.is_noise_dict, f)

    def get_GSDMM_Feature(self, json):
        """ is tw a chat or not """
        if tk.key_orgntext in json:
            json[tk.key_text] = pu.text_normalization(json[tk.key_orgntext])
        else:
            text = json[tk.key_text]
            json[tk.key_orgntext] = text
            json[tk.key_text] = pu.text_normalization(text)
        topic_num = self.c.sample_cluster(json)
        if topic_num in self.is_noise_dict:
            return 1
        return 0
