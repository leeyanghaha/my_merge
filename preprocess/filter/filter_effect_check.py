import datetime
import numpy as np
import re
import traceback
import pickle
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from preprocess.filter.filter_utils import readFilesAsJsonList, get_all_substrings
from preprocess.filter.pos_tag_process import count_ark_mark, count_sentiment
from preprocess.filter.pre_process import filterArray
from preprocess.filter.use_GSDMM import UseGSDMM
from utils.ark_service_proxy import twarr_ark
import utils.tweet_keys as tk
import utils.pattern_utils as pu
from config.configure import getcfg


# TODO 先用绝对路径标识文件名，我这里在配置文件中写成绝对路径了
clf_model_file = getcfg().clf_model_file
black_list_file = getcfg().black_list_file


class EffectCheck:
    def __init__(self, T_dir=None, F_dir=None):
        if T_dir is not None or F_dir is not None:
            self.T_corpus = readFilesAsJsonList(T_dir)
            self.F_corpus = readFilesAsJsonList(F_dir)
        self.gsdmm = None
        with open(black_list_file, 'r') as fp:
            self.spam_words = set([line.strip() for line in fp.readlines()])

    def run_a_function_list(self, function_list, print_pos_matchcase=False):
        for function in function_list:
            T_filtered = 0
            F_filtered = 0
            for T_element in self.T_corpus:
                if function(T_element):
                    if print_pos_matchcase:
                        print(T_element['text'])
                    T_filtered += 1
            for F_element in self.F_corpus:
                if function(F_element):
                    F_filtered += 1
            T_filter_rate = T_filtered / float(len(self.T_corpus))
            F_filter_rate = F_filtered / float(len(self.F_corpus))
            if T_filter_rate != 0:
                ratio = F_filter_rate / T_filter_rate
            else:
                ratio = 'inf'
            print('method {}, T filter rate = {}, F filter rate = {}'.format(
                function.__name__, round(T_filter_rate, 2), round(F_filter_rate, 2), round(ratio, 2)
            ))

    def test_get_feature(self):
        for json in self.T_corpus:
            self.get_features(json)
        for json in self.F_corpus:
            self.get_features(json)

    def filter(self, twarr):
        twarr = twarr_ark(twarr)
        self.gsdmm = UseGSDMM()
        data = [self.get_features(json) for json in twarr]
        try:
            with open(clf_model_file, 'rb') as f:
                clf = pickle.load(f)
        except:
            traceback.print_exc()
            return
        predict = clf.predict_proba(data)
        flt_twarr = list()
        for idx, p in enumerate(predict):
            if p:
                flt_twarr.append(twarr[idx])
        return flt_twarr

    # need to do if word list empty then split
    def get_features(self, json):
        user = json[tk.key_user]
        # l_profile_name = len(user['name'])
        # l_profile_description = 0
        if tk.key_description in user and user[tk.key_description] is not None:
            l_profile_description = len(user[tk.key_description])
        else:
            l_profile_description = 0
        FI = user[tk.key_friends_count]
        FE = user[tk.key_followers_count]
        num_tweet_posted = user[tk.key_statuses_count]

        tw_time = json[tk.key_created_at]
        user_born_time = json[tk.key_user][tk.key_created_at]
        # TODO 有些推文时间字段有误，需要判断处理，比如缺了分秒信息
        tw_d = datetime.datetime.strptime(tw_time, '%a %b %d %H:%M:%S %z %Y')
        user_d = datetime.datetime.strptime(user_born_time, '%a %b %d %H:%M:%S %z %Y')
        time_delta = tw_d - user_d
        AU = time_delta.seconds / 60.0 + time_delta.days * 24
        FE_FI_ratio = 0
        if FI != 0:
            FE_FI_ratio = FE / float(FI)
        reputation = 0
        if (FI + FE) != 0:
            reputation = FE / float(FI + FE)

        following_rate = FI / float(AU)
        tweets_per_day = num_tweet_posted / (AU / 24)
        tweets_per_week = num_tweet_posted / (AU / (24 * 7))

        user_features = [l_profile_description, FI, FE, num_tweet_posted, AU, FE_FI_ratio,
                         reputation, following_rate, tweets_per_day, tweets_per_week]
        """ content features """
        if tk.key_orgntext not in json:
            json[tk.key_orgntext] = json[tk.key_text]
        orgn = json[tk.key_orgntext]
        text = json[tk.key_text]
        words = text.split()
        num_words = len(words)
        # is this number of characters
        num_charater = len(text)
        num_white_space = len(re.findall(r'(\s)', text))
        num_capitalization_word = len(re.findall(r'(\b[A-Z]([a-z])*\b)', text))
        num_capital_per_word = num_capitalization_word / num_words

        max_word_length = 0
        mean_word_length = 0
        assert (len(words) > 0)
        for word in words:
            if len(word) > max_word_length:
                max_word_length = len(word)
                mean_word_length += len(word)

        mean_word_length /= len(words)
        num_exclamation_marks = orgn.count('!')
        num_question_marks = orgn.count('?')
        num_urls = len(json['entities']['urls'])
        num_urls_per_word = num_urls / num_words
        num_hashtags = len(json['entities']['hashtags'])
        num_hashtags_per_word = num_hashtags / num_words
        num_mentions = len(json['entities']['user_mentions'])
        num_mentions_per_word = num_mentions / num_words

        substrings = get_all_substrings(text)
        num_spam_words = 0
        for sub in substrings:
            if sub in self.spam_words:
                num_spam_words += 1
        num_spam_words_per_word = num_spam_words / num_words
        # pos_tags = count_ark_mark(json['ark'])
        content_features = [num_words, num_charater, num_white_space, num_capitalization_word,
                            num_capital_per_word, max_word_length, mean_word_length,
                            num_exclamation_marks, num_question_marks, num_urls, num_urls_per_word,
                            num_hashtags, num_hashtags_per_word, num_mentions,
                            num_mentions_per_word, num_spam_words, num_spam_words_per_word]
        sentiment_frature = count_sentiment(text)
        chat_feature = self.gsdmm.get_GSDMM_Feature(json)
        # content_features.extend(pos_tags)
        total_features = list()
        total_features.extend(user_features)
        total_features.extend(content_features)
        total_features.append(sentiment_frature)
        total_features.append(chat_feature)
        # lda_flag = getLDA().get_chat_LDA_feature(orgn)
        # total_features.append(lda_flag)
        return total_features

    def get_filter_res(self, fileDir):
        twarr = readFilesAsJsonList(fileDir)
        twarr = twarr_ark(twarr)
        self.gsdmm = UseGSDMM()
        data = [self.get_features(tw) for tw in twarr]
        try:
            with open(clf_model_file, 'rb') as fp:
                clf = pickle.load(fp)
        except:
            traceback.print_exc()
            return
        predict = clf.predict(data)
        table = pd.DataFrame(index={"data"}, columns={'保留', '被过滤'}, data=0)
        for i in range(len(predict)):
            if predict[i] == 1:
                table.loc["data", '保留'] += 1
            else:
                table.loc["data", '被过滤'] += 1
        print(table)
        print('总数：', len(predict), '过滤比例：', table.loc["data"]['被过滤'] / len(predict))

    def get_sets(self):
        x_t = [json for json in self.T_corpus]
        y_t = [1 for json in x_t]
        # x_t.extend(x_t)
        # x_t.extend(x_t)
        x_f = [json for json in self.F_corpus]
        x_f = x_f[:11000]
        y_f = [0 for json in x_f]
        x_t.extend(x_f)
        y_t.extend(y_f)
        # add pos feature to X with ark
        x_t = twarr_ark(x_t)

        X_train, X_test, y_train, y_test = train_test_split(x_t, y_t, test_size=0.2, random_state=42)

        gsdmm_train = []
        for i in range(len(X_train)):
            if y_train[i] == 1:
                gsdmm_train.append(X_train[i])
        # train GSDMM
        self.gsdmm = UseGSDMM()

        # get feature
        X_test_save = X_test
        X_train = [self.get_features(json) for json in X_train]
        X_test = [self.get_features(json) for json in X_test]

        clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=200, min_impurity_decrease=0.06)
        # tuned_parameter = [{'min_samples_leaf': [1, 2, 3], 'n_estimators': [10, 50, 100, 200],
        #                     'min_impurity_decrease': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]}]
        # gsearchcv = GridSearchCV(estimator=clf, param_grid=tuned_parameter, cv=5, n_jobs=5, scoring='recall')
        # gsearchcv.fit(x_t, y_t)
        # print('best params', gsearchcv.best_params_, 'best score', gsearchcv.best_score_)

        clf.fit(X_train, y_train)
        try:
            filepath = 'clf_N_T_041018'
            with open('../clf/' + filepath, 'wb') as f:
                pickle.dump(clf, f)
        except:
            traceback.print_exc()
        print("Training Score:")
        predict = clf.predict(X_train)
        print("accuracy ", metrics.accuracy_score(y_train, predict))
        print("recall", metrics.recall_score(y_train, predict))
        print("f1", metrics.f1_score(y_train, predict))
        print("roc auc", metrics.roc_auc_score(y_train, predict))

        print("Test Score:")
        predict = clf.predict(X_test)
        print("accuracy ", metrics.accuracy_score(y_test, predict))
        print("recall", metrics.recall_score(y_test, predict))
        print("f1", metrics.f1_score(y_test, predict))
        print("roc auc", metrics.roc_auc_score(y_test, predict))

        print("After spam filter")
        # 0 for spam
        filter_result = filterArray(X_test_save)
        if len(filter_result) == len(predict):
            for i in range(len(filter_result)):
                if filter_result[i] == 0:
                    predict[i] = 0
            print("accuracy ", metrics.accuracy_score(y_test, predict))
            print("recall", metrics.recall_score(y_test, predict))
            print("f1", metrics.f1_score(y_test, predict))
            print("roc auc", metrics.roc_auc_score(y_test, predict))

        table = pd.DataFrame(index=set(["test", "predict"]), columns=set([0, 1]), data=0)
        for i in range(len(y_test)):
            table.loc["test"][y_test[i]] += 1
        for i in range(len(predict)):
            table.loc["predict"][predict[i]] += 1
        print(table)
