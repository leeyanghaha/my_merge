import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from config.configure import getcfg
import classifying.fast_text_utils as ftu
import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.spacy_utils as su
import utils.tweet_keys as tk
import utils.timer_utils as tmu

from classifying.terror.classifier_terror import ClassifierTerror


# value_t, value_f = ftu.value_t, ftu.value_f
# terror_ft_model_file = getcfg().terror_ft_model_file
# terror_clf_model_file = getcfg().terror_lr_model_file
k_clf_model_file = ''


class ClassifierTerrorK(ClassifierTerror):
    nlp = None
    sensitive_words = {'shooting', 'wounded', 'shots', 'attack', 'shooter', 'wounds', 'dead',
                       'terrorist', 'hurt', 'terror', 'police', 'killed', 'gunman', 'weapon',
                       'injured', 'attacked', 'bomb', 'bombed', 'attacker'}
    
    @staticmethod
    def get_nlp():
        if ClassifierTerrorK.nlp is None:
            ClassifierTerrorK.nlp = su.get_nlp_disable_for_ner()
    
    def __init__(self, ft_model_file=terror_ft_model_file, clf_model_file=terror_clf_model_file):
        ClassifierTerror.__init__(self, )
        self.ft_model = self.clf_model = None
        if ft_model_file:
            self.load_ft_model(ft_model_file)
        if clf_model_file:
            self.load_clf_model(clf_model_file)
    
    def save_ft_model(self, ft_model_file): ftu.save_model(ft_model_file, self.ft_model)
    
    def load_ft_model(self, ft_model_file): self.ft_model = ftu.load_model(ft_model_file)
    
    def save_clf_model(self, clf_model_file): joblib.dump(self.clf_model, clf_model_file)
    
    def load_clf_model(self, clf_model_file): self.clf_model = joblib.load(clf_model_file)
    
    def get_ft_vector(self, text): return self.ft_model.get_sentence_vector(text)
    
    def has_locate_feature(self, doc):
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                return 1
        return -1
    
    def has_keyword_feature(self, text):
        count = 0
        for word in text.lower().split():
            if word.strip() in ClassifierTerrorK.sensitive_words:
                count += 1
        return count
    
    def train_ft_model(self, train_file, **kwargs):
        self.ft_model = ftu.FastText()
        self.ft_model.train_supervised(train_file, **kwargs)
    
    def train_clf_model(self, x, y, **kwargs):
        self.clf_model = LogisticRegression(**kwargs)
        self.clf_model.fit(x, y)
    
    def textarr2featurearr_no_gpe(self, textarr):
        vecarr = list()
        for text in textarr:
            try:
                ft_vec = self.get_ft_vector(text)
            except:
                text = pu.text_normalization(text)
                ft_vec = self.get_ft_vector(text)
            ft_vec = np.append(ft_vec, self.has_keyword_feature(text))
            vecarr.append(ft_vec)
        return np.array(vecarr)
    
    # def textarr2featurearr(self, textarr, docarr):
    #     assert len(textarr) == len(docarr)
    #     vecarr = list()
    #     for idx in range(len(textarr)):
    #         text, doc = textarr[idx], docarr[idx]
    #         try:
    #             ft_vec = self.get_ft_vector(text)
    #         except:
    #             text = pu.text_normalization(text)
    #             ft_vec = self.get_ft_vector(text)
    #         ft_vec = np.append(ft_vec, self.has_locate_feature(doc))
    #         ft_vec = np.append(ft_vec, self.has_keyword_feature(text))
    #         vecarr.append(ft_vec)
    #     assert len(docarr) == len(vecarr)
    #     return np.array(vecarr)
    
    def predict_proba(self, featurearr):
        return list(self.clf_model.predict_proba(featurearr)[:, 1])
    
    def predict_mean_proba(self, twarr):
        textarr = [tw.get(tk.key_text) for tw in twarr]
        featurearr = self.textarr2featurearr_no_gpe(textarr)
        mean_feature = np.mean(featurearr, axis=0).reshape([1, -1])
        return self.clf_model.predict_proba(mean_feature)[0, 1]
    
    def filter(self, twarr, threshold):
        textarr = [tw.get(tk.key_text) for tw in twarr]
        """"""
        # docarr = su.textarr_nlp(textarr, self.get_nlp())
        # featurearr = self.textarr2featurearr(textarr, docarr)
        featurearr = self.textarr2featurearr_no_gpe(textarr)
        """"""
        probarr = self.predict_proba(featurearr)
        assert len(twarr) == len(probarr)
        return [tw for idx, tw in enumerate(twarr) if probarr[idx] >= threshold]
    
    def train_ft(self, train_file, ft_args, ft_model_file):
        self.train_ft_model(train_file, **ft_args)
        self.save_ft_model(ft_model_file)
    
    def train_clf(self, featurearr, labelarr, clf_args, clf_model_file):
        print(featurearr.shape, labelarr.shape, np.mean(labelarr))
        self.train_clf_model(featurearr, labelarr, **clf_args)
        self.save_clf_model(clf_model_file)
    
    def test(self, test_file):
        textarr, labelarr = file2label_text_array(test_file)
        """"""
        # docarr = su.textarr_nlp(textarr, self.get_nlp())
        # featurearr = self.textarr2featurearr(textarr, docarr)
        featurearr = self.textarr2featurearr_no_gpe(textarr)
        """"""
        probarr = self.predict_proba(featurearr)
        au.precision_recall_threshold(labelarr, probarr,
            thres_range=[i / 100 for i in range(1, 10)] + [i / 20 for i in range(2, 20)])
        fu.dump_array("result.json", (labelarr, probarr))


def file2label_text_array(file):
    lines = fu.read_lines(file)
    return text2label_text_array(lines)


def text2label_text_array(txt_lbl_arr):
    label2value = ftu.binary_label2value
    textarr, labelarr = list(), list()
    for line in txt_lbl_arr:
        try:
            label, text = line.strip().split(' ', maxsplit=1)
            label = label2value[label]
        except:
            raise ValueError('Label wrong, check the text file.')
        textarr.append(text)
        labelarr.append(label)
    print('total text length: {}'.format(len(textarr)))
    return textarr, labelarr


def generate_train_matrices(ft_model_file, lbl_txt_file, mtx_lbl_file_list):
    lbl_txt_arr = fu.read_lines(lbl_txt_file)
    lbl_txt_blocks = mu.split_multi_format(lbl_txt_arr, len(mtx_lbl_file_list))
    args_list = [(ft_model_file, lbl_txt_blocks[idx], mtx_file, lbl_file)
                 for idx, (mtx_file, lbl_file) in enumerate(mtx_lbl_file_list)]
    print([len(b) for b in lbl_txt_blocks])
    mu.multi_process_batch(_generate_matrices, 10, args_list)


def _generate_matrices(ft_model_file, lbl_txt_arr, mtx_file, lbl_file):
    print(len(lbl_txt_arr), mtx_file, lbl_file)
    textarr, labelarr = text2label_text_array(lbl_txt_arr)
    clf = ClassifierTerrorK(ft_model_file, None)
    """"""
    # docarr = su.textarr_nlp(textarr, clf.get_nlp())
    # tmu.check_time('_generate_matrices')
    # featurearr = clf.textarr2featurearr(textarr, docarr)
    featurearr = clf.textarr2featurearr_no_gpe(textarr)
    """"""
    np.save(mtx_file, featurearr)
    np.save(lbl_file, labelarr)


def recover_train_matrix(mtx_lbl_file_list):
    mtx_list, lbl_list = list(), list()
    for idx, (mtx_file, lbl_file) in enumerate(mtx_lbl_file_list):
        print("recovering {} / {}".format(idx + 1, len(mtx_lbl_file_list)))
        mtx_list.append(np.load(mtx_file))
        lbl_list.append(np.load(lbl_file))
    featurearr = np.concatenate(mtx_list, axis=0)
    labelarr = np.concatenate(lbl_list, axis=0)
    return featurearr, labelarr


def coef_of_lr_model(file):
    from sklearn.externals import joblib
    lr_model = joblib.load(file)
    print(len(lr_model.coef_[0]))
    print(lr_model.coef_)


if __name__ == "__main__":
    from classifying.terror.data_maker import fasttext_train, fasttext_test, ft_data_pattern
    ft_model = "/home/nfs/cdong/tw/src/models/classify/terror/ft_no_gpe_model"
    lr_model = "/home/nfs/cdong/tw/src/models/classify/terror/lr_no_gpe_model"
    clf_model = lr_model
    tmu.check_time('all')
    tmu.check_time()
    
    # coef_of_lr_model("/home/nfs/cdong/tw/src/models/classify/terror/lr_no_gpe_model")
    # clf_filter = ClassifierAddFeature(None, None)
    # for file in fi.listchildren("/home/nfs/cdong/tw/seeding/Terrorist/queried/positive", concat=True):
    #     twarr = fu.load_array(file)
    #     print(file, clf_filter.predict_mean_proba(twarr))
    # tmu.check_time()
    # exit()
    
    batch_num = 20
    fi.mkdir(ft_data_pattern.format('matrices_no_add'), remove_previous=True)
    train_mtx_ptn = ft_data_pattern.format('matrices_no_add/train_feature_mtx_{}.npy')
    train_lbl_ptn = ft_data_pattern.format('matrices_no_add/train_lblarr_mtx_{}.npy')
    train_file_list = [(train_mtx_ptn.format(idx), train_lbl_ptn.format(idx)) for idx in range(batch_num)]
    
    # _clf = ClassifierAddFeature(None, None)
    # _ft_args = dict(epoch=150, lr=1.5, wordNgrams=2, verbose=2, minCount=2, thread=20, dim=300)
    # tmu.check_time()
    
    # _clf.train_ft(fasttext_train, _ft_args, ft_model)
    # tmu.check_time(print_func=lambda dt: print('train ft time: {}s'.format(dt)))
    # generate_train_matrices(ft_model, fasttext_train, train_file_list)
    # tmu.check_time()
    # _featurearr, _labelarr = recover_train_matrix(train_file_list)
    # tmu.check_time()
    #
    # _lr_args = dict(n_jobs=20, max_iter=300, tol=1e-6, class_weight={value_f: 1, value_t: 10})
    # _clf.train_clf(_featurearr, _labelarr, _lr_args, clf_model)
    # tmu.check_time(print_func=lambda dt: print('train lr time: {}s'.format(dt)))
    
    _clf = ClassifierTerrorK(ft_model, clf_model)
    _clf.test(fasttext_test)
    tmu.check_time(print_func=lambda dt: print('test time: {}s'.format(dt)))
    tmu.check_time('all')
